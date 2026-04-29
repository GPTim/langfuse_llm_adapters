"""
Entry point del plugin: definisce gli hook di lifecycle della richiesta
(before_cat_reads_message / before_cat_sends_message) e l'endpoint
/metrics per lo scraping Prometheus.

Hook lifecycle
--------------
1) before_cat_reads_message  -> start_request(): nuovo RequestMetrics nel ContextVar
2) ...esecuzione pipeline (LLM + embedder strumentati via Mixin)...
3) before_cat_sends_message  -> osserva i 5 histogram per-richiesta + clear

Endpoint /metrics
-----------------
Esposto via @endpoint del Cat, restituisce il payload in formato text-based
Prometheus exposition. In modalita' multi-worker (PROMETHEUS_MULTIPROC_DIR
settato) l'endpoint aggrega automaticamente tutti i worker.
"""

from cat.mad_hatter.decorators import hook, endpoint
from cat.log import log
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

from .prometheus_observability import request_context
from .prometheus_observability.registry import (
    get_registry,
    REQUEST_DURATION,
    # ACTIVE_REQUESTS,
    LLM_TIME_PER_REQUEST,
    EMBEDDER_TIME_PER_REQUEST,
    LLM_CALLS_PER_REQUEST,
    EMBEDDER_CALLS_PER_REQUEST,
    ACTIVE_SESSIONS
)


# ---------- Lifecycle hooks ----------

@hook(priority=100)
def before_cat_reads_message(user_message_json, cat):
    """Inizializza l'accumulatore per-richiesta.

    priority=100 per assicurarci di girare PRIMA di altri hook (es. quello
    di Langfuse nel plugin custom_llm), cosi' eventuali chiamate LLM/embedder
    fatte da altri hook nel stesso turn vengono comunque accumulate.
    """
    try:
        request_context.start_request()
        # ACTIVE_REQUESTS.inc()
    except Exception as e:
        log.warning(f"[prometheus_observability] start_request failed: {e}")
    return user_message_json

@hook(priority=-100)
def before_cat_sends_message(message, cat):
    """Emette le metriche aggregate sulla richiesta e resetta il contesto.

    priority=-100 per girare DOPO eventuali altri hook che potrebbero ancora
    fare chiamate LLM/embedder (es. arricchimento del messaggio).
    """
    metrics = request_context.get_current()
    if metrics is None:
        return message

    try:
        elapsed = metrics.elapsed()
        REQUEST_DURATION.observe(elapsed)
        LLM_TIME_PER_REQUEST.observe(metrics.llm_total_time)
        EMBEDDER_TIME_PER_REQUEST.observe(metrics.embedder_total_time)
        LLM_CALLS_PER_REQUEST.observe(metrics.llm_call_count)
        EMBEDDER_CALLS_PER_REQUEST.observe(metrics.embedder_call_count)
        # ACTIVE_REQUESTS.dec()

        log.debug(
            f"[prometheus_observability] request: total={elapsed:.2f}s "
            f"llm={metrics.llm_total_time:.2f}s ({metrics.llm_call_count} calls) "
            f"emb={metrics.embedder_total_time:.2f}s ({metrics.embedder_call_count} calls)"
        )
    except Exception as e:
        log.warning(f"[prometheus_observability] failed to emit per-request metrics: {e}")
    finally:
        request_context.end_request()

    return message

# ---------- Metrics endpoint ----------

@endpoint.get("/metrics", tags=["Observability"])
def prometheus_metrics():
    """Endpoint di scraping Prometheus.

    Configurazione minimale lato Prometheus (prometheus.yml):

        scrape_configs:
          - job_name: 'gptim'
            metrics_path: /metrics
            static_configs:
              - targets: ['cheshire-cat:80']

    Note:
    - L'endpoint e' pubblico per default, dato che non puo' usare check_permissions
      (Prometheus non porta credenziali utente). Se serve protezione, mettere
      il Cat dietro un reverse proxy che restringe /metrics per IP/network.
    - In multi-worker l'aggregazione e' automatica via MultiProcessCollector.
    """
    from cat.looking_glass.cheshire_cat import CheshireCat
    app = CheshireCat().fastapi_app
    ws_manager = app.state.websocket_manager
    ACTIVE_SESSIONS.set(float(len(ws_manager.connections)))
    registry = get_registry()
    payload = generate_latest(registry)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

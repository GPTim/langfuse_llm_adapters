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
from datetime import datetime, timezone
import os

from cat.mad_hatter.decorators import hook, endpoint
from cat.log import log
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import subprocess

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

from prometheus_client import Gauge
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
)


FILES_TO_PROCESS = Gauge(
    "rag_documents_to_be_processed",
    "Documents with supported extensions in files directory",
    multiprocess_mode="max"
)

FILES_IN_PROCESS = Gauge(
    "rag_documents_on_going",
    "Documents with on_going status in Qdrant",
    multiprocess_mode="max"
)

PROCESSING_FILE_AGE_SECONDS = Gauge(
    "rag_document_on_going_age_seconds",
    "Age in seconds of each document currently in ON_GOING",
    ["file_name", "point_id"],
    multiprocess_mode="max"
)

_PROCESSING_FILE_LABELS = set()

def update_files_to_process_gauge(files_path: str = "/app/cat/data/files"):
    result = subprocess.run(
        [
            "find", files_path, "-type", "f",
            "(", "-name", "*.docx",
            "-o", "-name", "*.pdf",
            "-o", "-name", "*.pptx",
            "-o", "-name", "*.zip",
            "-o", "-name", "*.md",
            "-o", "-name", "*.txt",
            ")"
        ],
        capture_output=True, text=True
    )
    count = len(result.stdout.strip().splitlines())
    FILES_TO_PROCESS.set(count)

def update_files_in_process_gauge(cat, collection: str = "documents"):
    """
    Recupera tutti i documenti ON_GOING da Qdrant tramite scroll,
    aggiorna FILES_IN_PROCESS e ritorna la lista dei punti trovati.
    """
    client = cat.memory.vectors.vector_db

    on_going_filter = Filter(
        must=[
            FieldCondition(
                key="status",
                match=MatchValue(value="ON_GOING")
            )
        ]
    )

    points = []
    offset = None

    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=on_going_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        points.extend(batch)

        if next_offset is None:
            break

        offset = next_offset

    FILES_IN_PROCESS.set(len(points))

    return points

def _parse_datetime(value):
    """
    Converte processing_started_at in datetime timezone-aware.
    Supporta formato ISO/RFC3339 tipo:
    2026-07-07T10:30:00Z
    """
    if not value:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    return None

def update_processing_file_age_seconds(on_going_points) -> None:
    """
    Prende i punti ON_GOING restituiti da update_files_in_process_gauge()
    e aggiorna la metrica per-file:

    rag_document_on_going_age_seconds{file_name="...", point_id="..."} 12345
    """
    global _PROCESSING_FILE_LABELS

    now = datetime.now(timezone.utc)
    current_labels = set()

    for point in on_going_points:
        payload = point.payload or {}

        point_id = str(point.id)
        file_name = _get_file_name(payload, point_id)

        started_at_raw = payload.get("updatedAt")
        started_at = _parse_datetime(started_at_raw)

        if started_at is None:
            age_seconds = 0
        else:
            age_seconds = max(0, int((now - started_at).total_seconds()))

        PROCESSING_FILE_AGE_SECONDS.labels(
            file_name=file_name,
            point_id=point_id,
        ).set(age_seconds)

        current_labels.add((file_name, point_id))

    # Pulizia label vecchie, utile quando un file non è più ON_GOING.
    # Nota: in modalità multiprocess Prometheus questa pulizia può non essere perfetta.
    stale_labels = _PROCESSING_FILE_LABELS - current_labels

    for file_name, point_id in stale_labels:
        try:
            PROCESSING_FILE_AGE_SECONDS.remove(file_name, point_id)
        except Exception as e:
            log.debug(
                f"[prometheus_observability] failed to remove stale processing file metric "
                f"file_name={file_name}, point_id={point_id}: {e}"
            )

    _PROCESSING_FILE_LABELS = current_labels

def _get_file_name(payload: dict, point_id) -> str:
    """
    Adatta qui le chiavi in base a come salvi il nome file nel payload Qdrant.
    """
    raw_name = (
        payload.get("filename")
        or str(point_id)
    )

    return os.path.basename(str(raw_name))

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
    cat = CheshireCat()
    update_files_to_process_gauge()
    on_going_points = update_files_in_process_gauge(cat=cat)
    update_processing_file_age_seconds(on_going_points)
    app = cat.fastapi_app
    ws_manager = app.state.websocket_manager
    ACTIVE_SESSIONS.set(float(len(ws_manager.connections)))
    registry = get_registry()
    payload = generate_latest(registry)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

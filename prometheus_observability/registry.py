"""
Registry Prometheus per il monitoraggio del Cheshire Cat.

Definisce tutte le metriche esposte dal plugin e gestisce la modalita'
multiprocess in modo trasparente: se la variabile d'ambiente
PROMETHEUS_MULTIPROC_DIR e' valorizzata, ogni worker scrive su file mmap
condivisi e l'endpoint /metrics aggrega tutto a scrape time. Altrimenti
si usa il registry di default (caso single-worker / dev).

Tutte le costanti definite a livello di modulo sono parte dell'API
pubblica del package e vengono importate dai Mixin e dagli hook.

Buckets degli histogram
-----------------------
I bucket sono calibrati sui range tipici osservati su questo deployment:
- richieste end-to-end: da pochi secondi (small talk) a minuti (deep search)
- chiamate LLM singole: da centinaia di ms a decine di secondi
- chiamate embedder: ordine dei millisecondi/decine di ms
- TTFT: tipicamente 100ms - qualche secondo
- refresh OAuth Vertex: decine di ms - 1-2s
"""

import os
from typing import Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    Gauge,
    multiprocess,
)


__all__ = [
    "get_registry",
    "cleanup_dead_worker",
    "REQUEST_DURATION",
    "LLM_TIME_PER_REQUEST",
    "EMBEDDER_TIME_PER_REQUEST",
    "LLM_CALLS_PER_REQUEST",
    "EMBEDDER_CALLS_PER_REQUEST",
    "LLM_CALLS_TOTAL",
    "EMBEDDER_CALLS_TOTAL",
    "LLM_TTFT",
    "VERTEX_TOKEN_REFRESH_TOTAL",
    "VERTEX_TOKEN_REFRESH_DURATION",
    "ACTIVE_SESSIONS",
]


def get_registry() -> CollectorRegistry:
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    from prometheus_client import REGISTRY
    return REGISTRY


# --- Durata richieste ------------------------------------------------------
# Range tipico: 0.5s (small talk) - 5min (deep search). Granularita' fine
# soprattutto tra 1-30s dove cade la maggior parte delle richieste reali.

REQUEST_DURATION = Histogram(
    "gptim_request_duration_seconds",
    "Durata totale di una richiesta utente, dall'arrivo del messaggio "
    "(before_cat_reads_message) all'invio della risposta "
    "(before_cat_sends_message).",
    buckets=(
        0.1, 0.25, 0.5, 0.75,
        1, 1.5, 2, 3, 4, 5, 7.5,
        10, 15, 20, 25, 30, 45,
        60, 70, 80, 90, 120, 180, 300, 600,
    ),
)

# --- Tempo cumulativo LLM/embedder per richiesta ---------------------------

LLM_TIME_PER_REQUEST = Histogram(
    "gptim_llm_time_per_request_seconds",
    "Wall-time cumulativo speso in chiamate LLM in una singola richiesta. "
    "In presenza di parallelismo (deep search) puo' superare la durata "
    "totale della richiesta.",
    buckets=(
        0.1, 0.25, 0.5, 0.75,
        1, 1.5, 2, 3, 4, 5, 7.5,
        10, 15, 20, 25, 30, 40, 45,
        60, 70, 80, 90, 120, 180, 300, 600,
    ),
)

EMBEDDER_TIME_PER_REQUEST = Histogram(
    "gptim_embedder_time_per_request_seconds",
    "Wall-time cumulativo speso in chiamate embedder in una singola richiesta.",
    buckets=(
        0.005, 0.01, 0.025, 0.05, 0.075,
        0.1, 0.15, 0.25, 0.5, 0.75,
        1, 1.5, 2, 3, 5, 7.5, 10, 12.5, 15,
    ),
)

# --- Numero di chiamate ----------------------------------------------------

LLM_CALLS_TOTAL = Counter(
    "gptim_llm_calls_total",
    "Numero totale di chiamate LLM effettuate, etichettate per modalita' "
    "(invoke = chiamata sincrona non-stream, stream = chiamata in streaming) "
    "e per status (success | error). Le chiamate effettuate fuori da una "
    "richiesta tracciata (es. ingestion) non vengono conteggiate.",
    ["mode", "status"],
)

LLM_CALLS_PER_REQUEST = Histogram(
    "gptim_llm_calls_per_request",
    "Numero di chiamate LLM effettuate in una singola richiesta. Utile "
    "per identificare richieste in loop di deep search.",
    buckets=(1, 2, 3, 4, 5, 6),
)

EMBEDDER_CALLS_TOTAL = Counter(
    "gptim_embedder_calls_total",
    "Numero totale di chiamate embedder, etichettate per metodo "
    "(query | documents) e per esito cache (hit | miss). I cache hit "
    "non comportano chiamate HTTP all'embedder remoto. Le chiamate "
    "effettuate fuori da una richiesta tracciata (es. ingestion) non "
    "vengono conteggiate.",
    ["method", "cache"],
)

EMBEDDER_CALLS_PER_REQUEST = Histogram(
    "gptim_embedder_calls_per_request",
    "Numero di chiamate embedder per singola richiesta (incluse le cache hit).",
    buckets=(1, 2, 3, 4, 5, 6),
)

# --- Time To First Token (streaming) ---------------------------------------

LLM_TTFT = Histogram(
    "gptim_llm_ttft_seconds",
    "Time To First Token: tempo tra l'invio della richiesta all'LLM e "
    "la ricezione del primo chunk con contenuto non vuoto. Misurato solo "
    "nelle chiamate in streaming dentro a una richiesta tracciata.",
    buckets=(
        0.05, 0.1, 0.15, 0.25, 0.4,
        0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 12.5, 15,
    ),
)

# --- Refresh OAuth Vertex --------------------------------------------------

VERTEX_TOKEN_REFRESH_TOTAL = Counter(
    "gptim_vertex_token_refresh_total",
    "Numero totale di refresh del token OAuth Vertex AI, etichettato per "
    "esito (success | error). Conta solo i refresh effettivi, non le "
    "verifiche di validita' che terminano senza refresh.",
    ["status"],
)

VERTEX_TOKEN_REFRESH_DURATION = Histogram(
    "gptim_vertex_token_refresh_duration_seconds",
    "Durata di un refresh del token OAuth Vertex AI (rete + ricreazione "
    "dei client OpenAI).",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5),
)


ACTIVE_SESSIONS = Gauge(
    "gptim_active_sessions",
    "Numero di sessioni utente attualmente aperte (websocket connessi).",
    multiprocess_mode="mostrecent",
)


def cleanup_dead_worker(pid: Optional[int] = None) -> None:
    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        return
    if pid is None:
        pid = os.getpid()
    try:
        multiprocess.mark_process_dead(pid)
    except Exception:
        pass
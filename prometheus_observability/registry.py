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
    # Funzioni
    "get_registry",
    "cleanup_dead_worker",
    # Histogram per-richiesta
    "REQUEST_DURATION",
    "LLM_TIME_PER_REQUEST",
    "EMBEDDER_TIME_PER_REQUEST",
    "LLM_CALLS_PER_REQUEST",
    "EMBEDDER_CALLS_PER_REQUEST",
    # Counter globali
    "LLM_CALLS_TOTAL",
    "EMBEDDER_CALLS_TOTAL",
    # TTFT streaming
    "LLM_TTFT",
    # Vertex OAuth
    "VERTEX_TOKEN_REFRESH_TOTAL",
    "VERTEX_TOKEN_REFRESH_DURATION",
    "ACTIVE_SESSIONS",
    # "ACTIVE_REQUESTS",
]


# ---------------------------------------------------------------------------
# Registry selection
# ---------------------------------------------------------------------------

def get_registry() -> CollectorRegistry:
    """Restituisce il registry corretto in base alla modalita' di deploy.

    In modalita' multiprocess (PROMETHEUS_MULTIPROC_DIR settata) crea un
    registry vuoto e ci attacca un MultiProcessCollector che a scrape time
    legge i file mmap di tutti i worker. In modalita' single-process usa
    il registry globale di default.
    """
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry

    from prometheus_client import REGISTRY
    return REGISTRY


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------
#
# Le metriche sono definite a livello di modulo: vengono create una volta
# all'import e riusate da tutti gli hook/mixin. In modalita' multiprocess
# il client gestisce in automatico la scrittura su file mmap condivisi.
#
# Naming convention: gptim_<dominio>_<unita>
# - _seconds per durate
# - _total per counter cumulativi
# - histogram senza suffisso quando misurano "quanto" qualcosa per richiesta

# --- Durata richieste ------------------------------------------------------

REQUEST_DURATION = Histogram(
    "gptim_request_duration_seconds",
    "Durata totale di una richiesta utente, dall'arrivo del messaggio "
    "(before_cat_reads_message) all'invio della risposta "
    "(before_cat_sends_message).",
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)

# --- Tempo cumulativo LLM/embedder per richiesta ---------------------------
#
# NOTA IMPORTANTE: queste metriche misurano il *wall-time accumulato* di
# tutte le chiamate effettuate durante una singola richiesta. Per via del
# parallelismo nel deep search (ThreadPoolExecutor che chiama l'LLM in
# parallelo sui chunk), il valore puo' eccedere la durata della richiesta
# stessa: e' atteso e rappresenta le "ore-LLM" consumate dal turno.

LLM_TIME_PER_REQUEST = Histogram(
    "gptim_llm_time_per_request_seconds",
    "Wall-time cumulativo speso in chiamate LLM in una singola richiesta. "
    "In presenza di parallelismo (deep search) puo' superare la durata "
    "totale della richiesta.",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)

EMBEDDER_TIME_PER_REQUEST = Histogram(
    "gptim_embedder_time_per_request_seconds",
    "Wall-time cumulativo speso in chiamate embedder in una singola richiesta.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
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
    buckets=(1, 2, 3, 5, 10, 20, 50, 100),
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
    buckets=(1, 2, 5, 10, 20, 50, 100),
)

# --- Time To First Token (streaming) ---------------------------------------

LLM_TTFT = Histogram(
    "gptim_llm_ttft_seconds",
    "Time To First Token: tempo tra l'invio della richiesta all'LLM e "
    "la ricezione del primo chunk con contenuto non vuoto. Misurato solo "
    "nelle chiamate in streaming dentro a una richiesta tracciata.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
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
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)


ACTIVE_SESSIONS = Gauge(
    "gptim_active_sessions",
    "Numero di sessioni utente attualmente aperte (websocket connessi).",
    multiprocess_mode="mostrecent",  # importante in multiproc
)

# ACTIVE_REQUESTS = Gauge(
#     "gptim_active_requests",
#     "Numero di richieste attualmente in elaborazione.",
#     multiprocess_mode="livesum",
# )

# ---------------------------------------------------------------------------
# Multiprocess cleanup
# ---------------------------------------------------------------------------

def cleanup_dead_worker(pid: Optional[int] = None) -> None:
    """Rimuove i file mmap di un worker terminato.

    Va chiamato dall'hook child_exit di gunicorn/uvicorn. Senza questo
    cleanup i file dei worker morti restano nel multiproc dir e
    contaminano le aggregazioni a scrape time (counter "fantasma" che
    non aumentano mai piu' ma vengono comunque sommati).
    """
    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        return
    if pid is None:
        pid = os.getpid()
    try:
        multiprocess.mark_process_dead(pid)
    except Exception:
        # In caso di errore non vogliamo bloccare lo shutdown del worker.
        pass
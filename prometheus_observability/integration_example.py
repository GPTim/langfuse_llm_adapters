# """
# Esempio di integrazione del monitoring Prometheus.

# ASSUNZIONE: il package `prometheus_observability/` è collocato come
# sotto-cartella DENTRO il plugin che definisce le classi LLM ed embedder
# custom (chiamiamolo qui `gptim_llm_embedder` come esempio). NON è un
# plugin Cheshire Cat indipendente.

# Layout sul filesystem:

#     cat/plugins/gptim_llm_embedder/
#     ├── plugin.json
#     ├── llm.py                       # Custom*WithLangfuse + LLMConfig*
#     ├── embedder.py                  # CachedOpenAICompatibleEmbeddings
#     ├── deep_search_hooks.py         # eventuali hook di deep search
#     └── prometheus_observability/    # <-- sotto-package
#         ├── __init__.py
#         ├── registry.py
#         ├── request_context.py
#         ├── monitored_llm.py
#         ├── monitored_embedder.py
#         └── monitored_vertex.py

# Conseguenze pratiche:
# - gli import sono relativi: `from .prometheus_observability.monitored_llm
#   import MonitoredLLMMixin`
# - gli hook `before_cat_reads_message`, `before_cat_sends_message` e
#   l'endpoint `/metrics` vengono dichiarati nel plugin host (es. in un
#   `monitoring_hooks.py` accanto agli altri file del plugin), perché il
#   Cat scopre `@hook` e `@endpoint` solo nei moduli top-level del plugin.
# - il sotto-package non ha bisogno di `plugin.json`: è codice di libreria,
#   non un plugin.

# Questo file mostra concretamente come modificare i file esistenti del
# plugin host per agganciare il monitoring.
# """


# # =============================================================================
# # 1. MODIFICA AL FILE LLM (es. llm.py del plugin host)
# # =============================================================================
# #
# # La classe ReasoningLLMMixin esiste già e va MANTENUTA.
# # Aggiungere MonitoredLLMMixin in cima alla MRO.
# #
# # IMPORTANTE: l'ordine dei Mixin nella MRO conta. MonitoredLLMMixin va
# # messo PRIMA di ReasoningLLMMixin, che a sua volta è prima di CustomOllama
# # o CustomOpenAI:
# #
# #   class CustomOpenaiLikeWithLangfuse(
# #       MonitoredLLMMixin,    # nuovo: prima di tutti
# #       ReasoningLLMMixin,
# #       CustomOpenAI,
# #   ):
# #       ...
# #
# # Questo garantisce che super()._generate / super()._stream del mixin
# # di monitoring risolvano correttamente verso le classi sottostanti
# # (ReasoningLLMMixin non override _generate/_stream, quindi la MRO
# # prosegue fino a CustomOpenAI).

# # Import relativo al sotto-package interno:
# from .prometheus_observability.monitored_llm import MonitoredLLMMixin
# from .prometheus_observability.monitored_vertex import (
#     instrument_token_refresh,
# )


# # Ollama -----------------------------------------------------------------

# class CustomOllamaWithLangfuse(
#     MonitoredLLMMixin,
#     # ReasoningLLMMixin,   <-- mantenere
#     # CustomOllama,        <-- mantenere
# ):
#     """Modificata: aggiunto MonitoredLLMMixin in cima alla MRO.

#     Tutto il resto del corpo della classe rimane invariato.
#     """
#     ...


# # OpenAI-like ------------------------------------------------------------

# class CustomOpenaiLikeWithLangfuse(
#     MonitoredLLMMixin,
#     # ReasoningLLMMixin,
#     # CustomOpenAI,
# ):
#     """Modificata: aggiunto MonitoredLLMMixin in cima alla MRO.

#     Il _generate e _stream definiti in questa classe vengono ora chiamati
#     da MonitoredLLMMixin via super(). Nessuna modifica al loro corpo.
#     """
#     ...


# # Vertex -----------------------------------------------------------------
# #
# # La classe Vertex eredita già da CustomOpenaiLikeWithLangfuse, quindi
# # eredita automaticamente il MonitoredLLMMixin. Aggiungiamo solo il
# # decorator al refresh token.

# class CustomVertexOpenaiLikeWithLangfuse(
#     # CustomOpenaiLikeWithLangfuse,   <-- mantenere; eredita MonitoredLLMMixin
# ):
#     """Modificata: applicato @instrument_token_refresh al metodo
#     _refresh_token_if_needed. Tutto il resto invariato."""

#     @instrument_token_refresh
#     def _refresh_token_if_needed(self) -> None:
#         # corpo originale invariato
#         ...


# # =============================================================================
# # 2. MODIFICA AL FILE EMBEDDER (es. embedder.py del plugin host)
# # =============================================================================
# #
# # Aggiungere MonitoredEmbedderMixin in cima alla MRO di
# # CachedOpenAICompatibleEmbeddings.

# from .prometheus_observability.monitored_embedder import (
#     MonitoredEmbedderMixin,
# )


# class CachedOpenAICompatibleEmbeddings(
#     MonitoredEmbedderMixin,
#     # Embeddings,   <-- mantenere
# ):
#     """Modificata: aggiunto MonitoredEmbedderMixin in cima alla MRO.

#     Il Mixin guarda self._cache (definito nell'__init__ originale)
#     per distinguere hit da miss. Nessun'altra modifica necessaria.
#     """
#     ...


# # =============================================================================
# # 3. NUOVO FILE: monitoring_hooks.py nel plugin host
# # =============================================================================
# #
# # Gli @hook e @endpoint vanno dichiarati a livello top-level del plugin
# # host (il Cat scopre i decorator solo lì), non nel sotto-package
# # prometheus_observability/. Creare un file `monitoring_hooks.py` accanto
# # a llm.py / embedder.py con questo contenuto:

# """
# # monitoring_hooks.py

# from cat.log import log
# from cat.mad_hatter.decorators import endpoint, hook
# from cat.looking_glass.stray_cat import StrayCat

# from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
# from fastapi import Response

# from .prometheus_observability.registry import (
#     EMBEDDER_CALLS_PER_REQUEST,
#     EMBEDDER_TIME_PER_REQUEST,
#     LLM_CALLS_PER_REQUEST,
#     LLM_TIME_PER_REQUEST,
#     REQUEST_DURATION,
#     get_registry,
# )
# from .prometheus_observability.request_context import end_request, start_request


# @hook(priority=100)
# def before_cat_reads_message(user_message_json, cat: StrayCat):
#     try:
#         start_request()
#     except Exception as e:
#         log.warning(f"[prometheus] start_request failed: {e}")
#     return user_message_json


# @hook(priority=-100)
# def before_cat_sends_message(message, cat: StrayCat):
#     try:
#         metrics = end_request()
#         if metrics is None:
#             return message
#         REQUEST_DURATION.observe(metrics.elapsed())
#         LLM_TIME_PER_REQUEST.observe(metrics.llm_total_time)
#         EMBEDDER_TIME_PER_REQUEST.observe(metrics.embedder_total_time)
#         LLM_CALLS_PER_REQUEST.observe(metrics.llm_call_count)
#         EMBEDDER_CALLS_PER_REQUEST.observe(metrics.embedder_call_count)
#     except Exception as e:
#         log.warning(f"[prometheus] flush metrics failed: {e}")
#     return message


# @endpoint.get("/metrics")
# def metrics_endpoint() -> Response:
#     data = generate_latest(get_registry())
#     return Response(content=data, media_type=CONTENT_TYPE_LATEST)
# """

# # NOTA: se il plugin host ha già un hook before_cat_reads_message
# # (es. quello di Langfuse nel tuo file LLM), si possono UNIRE in un
# # unico hook invece di averne due. L'ordine dei priority ne garantisce
# # comunque l'esecuzione corretta, ma per pulizia conviene un solo
# # punto di setup pre-richiesta.


# # =============================================================================
# # 4. MODIFICA AL FILE DEEP SEARCH (vectordb_deep_search)
# # =============================================================================
# #
# # Il deep search usa ThreadPoolExecutor, che NON propaga il ContextVar
# # automaticamente. Senza modifiche, le chiamate LLM dentro filter_chunk
# # continuerebbero a incrementare i counter globali ma NON ad aggiornare
# # il RequestMetrics della richiesta corrente (perché get_current() in
# # quei thread restituisce None).
# #
# # Se il deep search vive in un plugin SEPARATO da quello host del
# # monitoring, l'import deve essere ASSOLUTO al package del plugin host.
# # Esempio se il plugin host si chiama `gptim_llm_embedder`:
# #
# #   from cat.plugins.gptim_llm_embedder.prometheus_observability.request_context \
# #       import run_with_context
# #
# # Modifica nella funzione _llm_chunks_filter, sostituire la submit:
# #
# #   FROM:
# #     future_to_chunk = {
# #         executor.submit(filter_chunk, cat, recall_query, chunk): chunk
# #         for chunk in chunks_to_filter
# #     }
# #
# #   TO:
# #     from cat.plugins.gptim_llm_embedder.prometheus_observability.request_context \
# #         import run_with_context
# #     future_to_chunk = {
# #         executor.submit(run_with_context, filter_chunk, cat,
# #                         recall_query, chunk): chunk
# #         for chunk in chunks_to_filter
# #     }
# #
# # Senza questa modifica le metriche per-richiesta sottostimano il numero
# # di chiamate LLM e il loro tempo cumulativo nei turni con deep search
# # (che sono spesso quelli più interessanti da monitorare).


# # =============================================================================
# # 5. CONFIGURAZIONE GUNICORN/UVICORN PER MULTI-WORKER
# # =============================================================================
# #
# # Se il Cat gira con più worker, settare PROMETHEUS_MULTIPROC_DIR e
# # pulire i file dei worker morti. In un Dockerfile:
# #
# #   ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc
# #   RUN mkdir -p /tmp/prometheus_multiproc
# #
# # E nell'entrypoint, prima di avviare il server:
# #
# #   #!/bin/sh
# #   rm -f $PROMETHEUS_MULTIPROC_DIR/*.db
# #   exec uvicorn cat.main:cheshire_cat_api --workers 4 ...
# #
# # Per il cleanup dei worker che muoiono durante l'esecuzione, configurare
# # l'hook child_exit di gunicorn (file gunicorn.conf.py):
# #
# #   from cat.plugins.gptim_llm_embedder.prometheus_observability.registry \
# #       import cleanup_dead_worker
# #
# #   def child_exit(server, worker):
# #       cleanup_dead_worker(worker.pid)
# #
# # Con uvicorn standalone non esiste un hook equivalente: in pratica
# # accettare che i worker morti lascino file fino al prossimo restart.
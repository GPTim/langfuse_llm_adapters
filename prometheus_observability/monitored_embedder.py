"""
Mixin di monitoring Prometheus per gli embedder.

Si inserisce nella MRO della classe CachedOpenAICompatibleEmbeddings
(o simili) e intercetta `embed_query` ed `embed_documents` per misurare
durata, contare le invocazioni e distinguere cache hit da miss.

Distinzione hit/miss
--------------------
Il mixin guarda il dizionario `_cache` della classe sottostante PRIMA
di delegare al super(). Questo accoppia leggermente il mixin
all'implementazione (presuppone l'esistenza di `self._cache`), ma e'
l'unico modo robusto per etichettare correttamente le metriche senza
modificare la classe base. Se la classe non ha `_cache` il mixin
considera tutte le chiamate come "miss" (corretto: non c'e' cache).

Per `embed_documents` la situazione e' mista: in una singola chiamata
alcuni testi possono essere cache-hit e altri miss. Conteggiamo
ogni testo individualmente sul counter ma misuriamo una sola durata
wall-time per la chiamata complessiva e la sommiamo all'accumulatore
per-richiesta.

Filtro ingestion
----------------
`embed_documents` durante l'ingestion (Rabbit Hole) puo' processare
batch di centinaia o migliaia di chunk: se non filtrato, il counter
EMBEDDER_CALLS_TOTAL esploderebbe in spike che oscurerebbero il
traffico conversazionale. Il bypass via is_tracking_active() salta
completamente l'ingestion: queste chiamate non vengono contate ne'
attribuite ad alcuna richiesta.
"""

import time
from typing import List

from .registry import EMBEDDER_CALLS_TOTAL
from .request_context import get_current, is_tracking_active


class MonitoredEmbedderMixin:
    """Mixin che strumenta le chiamate embedder verso Prometheus.

    Aggiorna (solo se is_tracking_active()):
    - EMBEDDER_CALLS_TOTAL: counter globale, label method + cache
    - tempo cumulato sul RequestMetrics corrente

    Se il tracking non e' attivo il mixin e' un passthrough.
    """

    def _is_cached(self, text: str) -> bool:
        """Verifica se un testo e' gia' nella cache. Override-friendly."""
        cache = getattr(self, "_cache", None)
        if cache is None:
            return False
        return text in cache

    def embed_query(self, text: str) -> List[float]:
        # Bypass se siamo fuori dal flusso conversazionale tracciato.
        if not is_tracking_active():
            return super().embed_query(text)

        cached = self._is_cached(text)
        cache_label = "hit" if cached else "miss"

        start = time.monotonic()
        try:
            result = super().embed_query(text)
        finally:
            duration = time.monotonic() - start
            EMBEDDER_CALLS_TOTAL.labels(
                method="query", cache=cache_label
            ).inc()
            metrics = get_current()
            if metrics is not None:
                metrics.add_embedder_call(duration)
        return result

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Bypass: cruciale per ingestion, dove embed_documents puo'
        # essere chiamato con batch enormi e contaminerebbe le metriche
        # conversazionali.
        if not is_tracking_active():
            return super().embed_documents(texts)

        # Conta hit/miss per testo individuale prima di delegare:
        # dopo la chiamata super() la cache contiene anche i nuovi miss
        # e non potremmo piu' distinguerli.
        hits = sum(1 for t in texts if self._is_cached(t))
        misses = len(texts) - hits

        start = time.monotonic()
        try:
            result = super().embed_documents(texts)
        finally:
            duration = time.monotonic() - start
            if hits:
                EMBEDDER_CALLS_TOTAL.labels(
                    method="documents", cache="hit"
                ).inc(hits)
            if misses:
                EMBEDDER_CALLS_TOTAL.labels(
                    method="documents", cache="miss"
                ).inc(misses)
            metrics = get_current()
            if metrics is not None:
                # Una chiamata embed_documents = una chiamata HTTP (o
                # zero se tutto cached): la contiamo come 1 ai fini
                # del wall-time, ma incrementiamo il count per-request
                # di len(texts) per coerenza con il counter globale.
                metrics.embedder_total_time += duration
                metrics.embedder_call_count += len(texts)
        return result

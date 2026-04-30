"""
Custom embedder compatibile con API OpenAI-like, con cache LRU in-memory
sui risultati di `embed_query` e strumentazione Prometheus integrata.

Razionale
---------
Nel flusso del Cat, la stessa query puo' essere embedded piu' volte nello
stesso turno (es. recall episodic + declarative + procedural) o in turni
successivi (es. utente che riformula la stessa domanda). Cachando gli
embedding per query evitiamo chiamate HTTP ridondanti all'embedder,
riducendo latenza e costi.

Solo `embed_query` e' cachato. `embed_documents` NON lo e' per due motivi:
- viene usato in ingestion (batch di chunk), dove i testi sono tipicamente
  unici e la cache non darebbe hit
- i payload sono grossi e riempirebbero la cache velocemente

Strumentazione integrata
------------------------
La metrica e' integrata direttamente nei metodi della classe (non via
Mixin) perche' `embed_query` ed `embed_documents` non chiamano
`super().embed_*`: implementano tutta la logica internamente. Un Mixin
montato sulla MRO sarebbe scavalcato. Fare il wrap "from inside" e'
piu' diretto e meno sorprendente.

Filtro ingestion
----------------
La strumentazione e' attiva SOLO se siamo dentro una richiesta tracciata
(`is_tracking_active()`). Durante l'ingestion via Rabbit Hole il
RequestMetrics non esiste, quindi le chiamate (anche batch enormi) non
vengono contate ne' attribuite, mantenendo puliti i grafici delle
metriche conversazionali.
"""

import json
import os
import time
from collections import OrderedDict
from typing import List, Type, Optional

import httpx
from langchain_core.embeddings import Embeddings
from pydantic import ConfigDict

from cat.factory.embedder import EmbedderSettings
from cat.factory.custom_embedder import CustomOpenAIEmbeddings
from cat.log import log
from cat.mad_hatter.decorators import hook

from .prometheus_observability.registry import EMBEDDER_CALLS_TOTAL
from .prometheus_observability.request_context import (
    get_current,
    is_tracking_active,
)


MAX_CACHE_SIZE = 50


class CachedOpenAICompatibleEmbeddings(CustomOpenAIEmbeddings):
    """Embedder con cache LRU su embed_query (max 50 elementi)
    e strumentazione Prometheus integrata."""

    def __init__(self, url: str):
        self.url = os.path.join(url, "v1/embeddings")
        # OrderedDict per implementare LRU: spostiamo in fondo gli elementi
        # acceduti di recente ed evictiamo dalla testa.
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()

    # ------------------------------------------------------------------ #
    #  embed_query                                                       #
    # ------------------------------------------------------------------ #

    def embed_query(self, text: str) -> List[float]:
        # Determiniamo PRIMA se e' hit o miss, perche' subito dopo la
        # cache viene mutata (move_to_end / inserimento) e l'info viene
        # persa.
        is_hit = text in self._cache
        tracking = is_tracking_active()

        start = time.monotonic() if tracking else None

        try:
            if is_hit:
                # Hit: ritorniamo dalla cache, aggiornando l'ordine LRU.
                self._cache.move_to_end(text)
                log.debug(f"[cached-embedder] HIT (cache size: {len(self._cache)})")
                embedding = self._cache[text]
            else:
                # Miss: chiamata HTTP all'embedder remoto.
                log.debug(f"[cached-embedder] MISS (cache size: {len(self._cache)})")
                payload = json.dumps({"input": text})
                ret = httpx.post(self.url, data=payload, timeout=None)
                ret.raise_for_status()
                embedding = ret.json()["data"][0]["embedding"]

                # Salviamo in cache e applichiamo politica di eviction LRU.
                self._cache[text] = embedding
                if len(self._cache) > MAX_CACHE_SIZE:
                    self._cache.popitem(last=False)

            return embedding

        finally:
            # Strumentazione: solo se siamo in una richiesta tracciata.
            # Va in finally per coprire anche il caso di eccezione HTTP
            # (counter incrementato comunque, durata realistica).
            if tracking:
                duration = time.monotonic() - start
                cache_label = "hit" if is_hit else "miss"
                EMBEDDER_CALLS_TOTAL.labels(
                    method="query", cache=cache_label
                ).inc()
                metrics = get_current()
                if metrics is not None:
                    metrics.add_embedder_call(duration)

    # ------------------------------------------------------------------ #
    #  embed_documents                                                   #
    # ------------------------------------------------------------------ #

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        tracking = is_tracking_active()

        # Conteggio hit/miss PRIMA di mutare la cache, altrimenti dopo
        # super() i nuovi miss diventerebbero hit e i numeri sarebbero
        # falsati.
        if tracking:
            hits = sum(1 for t in texts if t in self._cache)
            misses = len(texts) - hits
            start = time.monotonic()
        else:
            hits = misses = 0
            start = None

        try:
            results: List[Optional[List[float]]] = [None] * len(texts)
            missing_texts = []
            missing_indices = []

            for i, text in enumerate(texts):
                if text in self._cache:
                    self._cache.move_to_end(text)
                    results[i] = self._cache[text]
                    log.debug(f"[cached-embedder] HIT doc '{text[:20]}...'")
                else:
                    missing_texts.append(text)
                    missing_indices.append(i)

            if missing_texts:
                log.debug(f"[cached-embedder] MISS {len(missing_texts)} docs")
                payload = json.dumps({"input": missing_texts})
                ret = httpx.post(self.url, data=payload, timeout=None)
                ret.raise_for_status()
                embeddings = [e["embedding"] for e in ret.json()["data"]]

                for idx, text, emb in zip(missing_indices, missing_texts, embeddings):
                    self._cache[text] = emb
                    results[idx] = emb
                    if len(self._cache) > MAX_CACHE_SIZE:
                        self._cache.popitem(last=False)

            return results

        finally:
            if tracking:
                duration = time.monotonic() - start
                # Counter globale: incrementato per ogni testo, distinguendo
                # hit da miss (i miss generano la chiamata HTTP, gli hit no).
                if hits:
                    EMBEDDER_CALLS_TOTAL.labels(
                        method="documents", cache="hit"
                    ).inc(hits)
                if misses:
                    EMBEDDER_CALLS_TOTAL.labels(
                        method="documents", cache="miss"
                    ).inc(misses)

                # Per-richiesta: una sola wall-time (la chiamata HTTP e'
                # una sola se ci sono miss, zero se tutto cached). Il
                # counter per-richiesta invece si incrementa di len(texts)
                # per coerenza con il counter globale.
                metrics = get_current()
                if metrics is not None:
                    metrics.embedder_total_time += duration
                    metrics.embedder_call_count += len(texts)


# ---------------------------------------------------------------------- #
#  Settings adapter                                                      #
# ---------------------------------------------------------------------- #

class EmbedderCachedOpenAICompatibleConfig(EmbedderSettings):
    url: str
    _pyclass: Type = CachedOpenAICompatibleEmbeddings

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "OpenAI-compatible API embedder (cached)",
            "description": "Embedder OpenAI-compatible con cache in-memory "
                           "sugli embed_query.",
            "link": "",
        }
    )


@hook
def factory_allowed_embedders(allowed, cat) -> List:
    allowed.append(EmbedderCachedOpenAICompatibleConfig)
    return allowed
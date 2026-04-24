"""
Custom embedder compatibile con API OpenAI-like, con cache LRU in-memory
sui risultati di `embed_query`.

Razionale
---------
Nel flusso del Cat, la stessa query può essere embedded più volte nello
stesso turno (es. recall episodic + declarative + procedural) o in turni
successivi (es. utente che riformula la stessa domanda). Cachando gli
embedding per query evitiamo chiamate HTTP ridondanti all'embedder,
riducendo latenza e costi.

Solo `embed_query` è cachato. `embed_documents` NON lo è per due motivi:
- viene usato in ingestion (batch di chunk), dove i testi sono tipicamente
  unici e la cache non darebbe hit
- i payload sono grossi e riempirebbero la cache velocemente

Thread-safety
-------------
La cache usa `functools.lru_cache` che è thread-safe sul GIL di CPython.
Per ambienti multi-worker (uvicorn --workers > 1), ogni worker ha la sua
cache indipendente: accettabile perché si riscalda rapidamente.
"""

import json
import os
from collections import OrderedDict
from typing import List, Type, Optional

import httpx
from langchain_core.embeddings import Embeddings
from pydantic import ConfigDict

from cat.factory.embedder import EmbedderSettings
from cat.log import log
from cat.mad_hatter.decorators import hook


MAX_CACHE_SIZE = 50
 
 
class CachedOpenAICompatibleEmbeddings(Embeddings):
    """Embedder con cache LRU in-memory su embed_query (max 50 elementi)."""
 
    def __init__(self, url: str):
        self.url = os.path.join(url, "v1/embeddings")
        # OrderedDict per implementare LRU: spostiamo in fondo gli elementi
        # acceduti di recente ed evictiamo dalla testa.
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()
 
    def embed_query(self, text: str) -> List[float]:
        if text in self._cache:
            # Move to end = "most recently used"
            self._cache.move_to_end(text)
            log.debug(f"[cached-embedder] HIT (cache size: {len(self._cache)})")
            return self._cache[text]
 
        log.debug(f"[cached-embedder] MISS (cache size: {len(self._cache)})")
        payload = json.dumps({"input": text})
        ret = httpx.post(self.url, data=payload, timeout=None)
        ret.raise_for_status()
        embedding = ret.json()["data"][0]["embedding"]
 
        self._cache[text] = embedding
        # Evict least recently used se sforiamo il limite
        if len(self._cache) > MAX_CACHE_SIZE:
            self._cache.popitem(last=False)
 
        return embedding
 
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
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

        if len(missing_texts):
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
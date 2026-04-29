"""
Mixin di monitoring Prometheus per i chat model LLM.

Si inserisce nella MRO delle classi Custom*WithLangfuse esistenti e
intercetta le chiamate a `_generate` (modalita' non-streaming) e
`_stream` (modalita' streaming) per misurare durata, contare le
invocazioni e tracciare il TTFT.

Posizionamento nella MRO
------------------------
Il mixin va messo PRIMA delle classi Custom* nella catena di ereditarieta':

    class CustomOpenaiLikeWithLangfuse(MonitoredLLMMixin, ReasoningLLMMixin, CustomOpenAI):
        ...

cosi' `super()._generate(...)` risolve correttamente alla logica della
classe Custom* sottostante.

Cosa NON wrappiamo
------------------
Non wrappiamo `invoke()` perche' internamente in LangChain `invoke()`
chiama `_generate()` o `_stream()`: wrappare entrambi causerebbe
double-counting. Misuriamo allo strato piu' basso possibile.

Filtro ingestion
----------------
Le chiamate effettuate fuori da una richiesta tracciata (ingestion via
Rabbit Hole, bootstrap, ecc.) vengono delegate a super() senza alcuna
strumentazione: ne' counter globali ne' osservazioni per-richiesta.
Il check e' `is_tracking_active()`: piu' veloce di un get_current()
seguito da None-check, e centralizzato in request_context.py.
"""

import time
from typing import Any, Optional

from .registry import (
    LLM_CALLS_TOTAL,
    LLM_TTFT
)
from .request_context import get_current, is_tracking_active

class MonitoredLLMMixin:
    """Strumenta le chiamate LLM verso Prometheus.

    Convenzioni:
    - invoke()  -> conta la chiamata logica (LLM_CALLS_TOTAL) e
                   accumula la durata totale sul RequestMetrics corrente.
    - _stream() -> osserva solo il TTFT (Time To First Token).
                   NON incrementa contatori, NON accumula durata:
                   quello e' gia' fatto da invoke() che sta sopra.
    """

    def invoke(
        self,
        input: Any,
        config: Optional[Any] = None,
        *,
        stop: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        if not is_tracking_active():
            return super().invoke(input, config, stop=stop, **kwargs)

        mode = "stream" if getattr(self, "streaming", False) else "invoke"
        status = "success"
        start = time.monotonic()

        try:
            return super().invoke(input, config, stop=stop, **kwargs)
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.monotonic() - start
            LLM_CALLS_TOTAL.labels(mode=mode, status=status).inc()
            metrics = get_current()
            if metrics is not None:
                metrics.add_llm_call(duration)

    def _stream(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs,
    ):
        # Fuori da una richiesta tracciata: passthrough puro.
        if not is_tracking_active():
            yield from super()._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return

        start = time.monotonic()
        first_token_seen = False

        for chunk in super()._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            if not first_token_seen:
                # text = getattr(chunk, "text", "") or ""
                LLM_TTFT.observe(time.monotonic() - start)
                first_token_seen = True

            yield chunk
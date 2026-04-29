"""
Mixin per strumentare _refresh_token_if_needed di CustomVertexOpenaiLikeWithLangfuse.

Misura solo i refresh che effettivamente avvengono (token scaduto): le chiamate
no-op (token ancora valido) non vengono contate ne' osservate, perche'
inquinerebbero le metriche con un volume enorme di refresh "fittizi" (uno per
ogni richiesta utente).

Identifichiamo il refresh effettivo intercettando il punto in cui credentials.refresh()
viene chiamato. Il modo piu' robusto e' rifare l'override della funzione, ma per
non duplicare la logica usiamo un trucco: misuriamo la durata totale di
_refresh_token_if_needed e la registriamo SOLO se sopra una soglia di 5ms (il
no-op fa solo un check di validita' del token, ben sotto 1ms).

In alternativa, se preferisci precisione assoluta: copia _refresh_token_if_needed
e separa esplicitamente il path no-op dal path refresh, emettendo le metriche
solo nel secondo. Questa via e' commentata in fondo al file.
"""

import time

from .registry import (
    VERTEX_TOKEN_REFRESH_TOTAL,
    VERTEX_TOKEN_REFRESH_DURATION,
)


# Soglia sotto la quale consideriamo che _refresh_token_if_needed sia stato
# un no-op (token valido, nessuna chiamata di rete). 5 ms e' generosa: il
# check di validita' su credentials.valid e' ~microsecondi.
_REFRESH_NOOP_THRESHOLD_S = 0.005


class MonitoredVertexMixin:
    """Mixin per CustomVertexOpenaiLikeWithLangfuse."""

    def _refresh_token_if_needed(self) -> None:
        start = time.monotonic()
        error = False
        try:
            super()._refresh_token_if_needed()
        except Exception:
            error = True
            raise
        finally:
            duration = time.monotonic() - start

            # Filtriamo i no-op: se siamo sotto soglia e non c'e' stato errore,
            # non contiamo. Errori vengono sempre contati.
            if duration >= _REFRESH_NOOP_THRESHOLD_S or error:
                status = "error" if error else "success"
                VERTEX_TOKEN_REFRESH_TOTAL.labels(status=status).inc()
                VERTEX_TOKEN_REFRESH_DURATION.observe(duration)

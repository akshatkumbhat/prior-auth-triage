"""PII-safe logging.

The Intake node is the sanitization gateway: as it parses a raw bundle it
registers every sensitive string it sees (patient name, MRN, DOB, and the raw
clinical narrative) with the process-wide :data:`redactor`. A logging filter
then masks any log record whose message contains a registered value, so even
an accidental ``logger.info(raw_text)`` elsewhere cannot leak PII to the logs.

This is defense-in-depth, not a substitute for not logging PII in the first
place: agents are written to log only de-identified summaries.
"""

from __future__ import annotations

import logging
import re
import threading

_MASK = "[REDACTED]"
_MIN_LEN = 3  # don't register trivially short strings (avoids masking noise)


class PiiRedactor:
    """Thread-safe registry of sensitive substrings to mask in log output."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._patterns: list[re.Pattern[str]] = []
        self._raw: set[str] = set()

    def register(self, *values: str | None) -> None:
        """Register one or more sensitive values to be masked from logs."""
        with self._lock:
            for value in values:
                if not value:
                    continue
                value = str(value).strip()
                if len(value) < _MIN_LEN or value in self._raw:
                    continue
                self._raw.add(value)
                self._patterns.append(re.compile(re.escape(value), re.IGNORECASE))

    def redact(self, text: str) -> str:
        """Return ``text`` with every registered value replaced by the mask."""
        with self._lock:
            patterns = list(self._patterns)
        for pattern in patterns:
            text = pattern.sub(_MASK, text)
        return text

    def clear(self) -> None:
        """Forget all registered values (used between runs / in tests)."""
        with self._lock:
            self._patterns.clear()
            self._raw.clear()


# Process-wide singleton. Intake registers into this; the filter reads from it.
redactor = PiiRedactor()


class _RedactingFilter(logging.Filter):
    """Logging filter that scrubs registered PII from every record's message."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Render args into the message now, then redact the final string.
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        redacted = redactor.redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def mask_name(name: str | None) -> str:
    """Return a PII-masked form of a person name for confirmation logs.

    e.g. ``"Jane Q. Doe" -> "J*** Q. D**"``. Used by Intake to prove it saw a
    name without writing the name itself.
    """
    if not name:
        return _MASK
    parts = name.split()
    masked = []
    for part in parts:
        if len(part) <= 1:
            masked.append(part)
        else:
            masked.append(part[0] + "*" * (len(part) - 1))
    return " ".join(masked) or _MASK


def get_logger(name: str) -> logging.Logger:
    """Return a logger that has the redacting filter attached.

    The filter is attached idempotently so repeated calls are safe.
    """
    logger = logging.getLogger(name)
    if not any(isinstance(f, _RedactingFilter) for f in logger.filters):
        logger.addFilter(_RedactingFilter())
    return logger

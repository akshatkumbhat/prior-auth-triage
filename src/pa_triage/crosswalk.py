"""ICD-10 / CPT crosswalk: loading and deterministic term matching.

The Clinical Reviewer uses an LLM only to *extract* concise clinical terms
from narrative text. Mapping those terms to codes is fully deterministic here,
so the code assignment is reproducible and unit-testable.

Matching is token-subset based: an alias matches a term when all of the
alias's tokens are present in the term's tokens. The match with the most
overlapping tokens wins, which keeps multi-word procedures from matching on a
single shared word.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from pa_triage.config import get_settings
from pa_triage.models.domain import Diagnosis, Procedure

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Tokens too generic to carry matching weight on their own.
_STOPWORDS = {"of", "the", "a", "an", "and", "or", "with", "without", "for", "to", "in"}


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


class DiagnosisEntry(BaseModel):
    icd10: str
    description: str
    terms: list[str] = Field(default_factory=list)


class ProcedureEntry(BaseModel):
    cpt: str
    description: str
    terms: list[str] = Field(default_factory=list)


class Crosswalk(BaseModel):
    diagnoses: list[DiagnosisEntry] = Field(default_factory=list)
    procedures: list[ProcedureEntry] = Field(default_factory=list)

    def _best_match(
        self, term: str, entries: list[DiagnosisEntry] | list[ProcedureEntry]
    ) -> tuple[DiagnosisEntry | ProcedureEntry | None, int]:
        term_tokens = _tokens(term)
        if not term_tokens:
            return None, 0
        best: DiagnosisEntry | ProcedureEntry | None = None
        best_score = 0
        for entry in entries:
            for alias in [*entry.terms, entry.description]:
                alias_tokens = _tokens(alias)
                if not alias_tokens:
                    continue
                # Alias fully contained in the term (typical), or vice-versa.
                if alias_tokens <= term_tokens or term_tokens <= alias_tokens:
                    score = len(alias_tokens & term_tokens)
                    if score > best_score:
                        best, best_score = entry, score
        return best, best_score

    def map_diagnosis(self, term: str) -> Diagnosis | None:
        entry, score = self._best_match(term, self.diagnoses)
        if entry is None or score == 0:
            return None
        assert isinstance(entry, DiagnosisEntry)
        return Diagnosis(icd10=entry.icd10, description=entry.description, source_text=term)

    def map_procedure(self, term: str) -> Procedure | None:
        entry, score = self._best_match(term, self.procedures)
        if entry is None or score == 0:
            return None
        assert isinstance(entry, ProcedureEntry)
        return Procedure(cpt=entry.cpt, description=entry.description, source_text=term)


def load_crosswalk(path: Path | None = None) -> Crosswalk:
    """Load the crosswalk JSON from ``path`` (defaults to configured location)."""
    path = path or get_settings().crosswalk_path
    data = json.loads(Path(path).read_text())
    return Crosswalk.model_validate(data)


@lru_cache(maxsize=1)
def get_crosswalk() -> Crosswalk:
    """Return a cached crosswalk loaded from the configured path."""
    return load_crosswalk()

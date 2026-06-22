"""Heuristic, offline stand-in for a real LLM.

This is **not** an LLM. It implements just enough of the LangChain chat-model
surface (``with_structured_output(...).invoke(...)``) for the three structured
calls the agents make, using simple deterministic heuristics. Its only purpose
is to let someone run the dashboard end-to-end with **zero setup** (no API key,
no local model) so the orchestration and UI can be demonstrated.

Outputs are illustrative, not clinical reasoning. For real reasoning, set
``LLM_PROVIDER=gemini`` (or ``ollama``) and use :func:`pa_triage.llm.get_llm`.
"""

from __future__ import annotations

from typing import Any

from pa_triage.agents.clinical import ClinicalExtraction
from pa_triage.agents.coverage import CriteriaAssessment, CriterionJudgment
from pa_triage.agents.decision import RationaleDraft


class _Runnable:
    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        return self._fn(messages)


class HeuristicLLM:
    """Deterministic, offline stand-in. See module docstring."""

    is_demo = True

    def with_structured_output(self, schema: type, **_: Any):
        if schema is ClinicalExtraction:
            return _Runnable(self._extract)
        if schema is CriteriaAssessment:
            return _Runnable(self._assess)
        if schema is RationaleDraft:
            return _Runnable(self._rationale)
        raise KeyError(f"HeuristicLLM does not support schema {schema!r}")

    # --- handlers --------------------------------------------------------
    def _extract(self, _messages: Any) -> ClinicalExtraction:
        # Rely on the deterministic crosswalk backstop over claim-stated terms.
        return ClinicalExtraction(diagnosis_terms=[], procedure_terms=[])

    def _assess(self, messages: Any) -> CriteriaAssessment:
        narrative = messages[-1][1].lower()
        met_therapy = (
            ("physical therapy" in narrative or "nsaid" in narrative)
            and "week" in narrative
            and "no physical therapy" not in narrative
        )
        met_deficit = (
            "foot drop" in narrative
            or "motor weakness" in narrative
            or ("deficit" in narrative and "without focal" not in narrative)
        )
        judgments: list[CriterionJudgment] = []
        for line in messages[-1][1].splitlines():
            line = line.strip()
            if not line.startswith("- "):
                continue
            criterion = line[2:]
            low = criterion.lower()
            if "conservative" in low or "therapy" in low:
                met = met_therapy
            elif "neurologic" in low or "deficit" in low or "radiculopathy" in low:
                met = met_deficit
            else:
                met = False
            judgments.append(
                CriterionJudgment(
                    criterion=criterion,
                    met=met,
                    evidence="Documented in narrative." if met else None,
                )
            )
        return CriteriaAssessment(
            judgments=judgments,
            notes="(Heuristic demo assessment — not real LLM reasoning.)",
        )

    def _rationale(self, messages: Any) -> RationaleDraft:
        context = messages[-1][1]
        lines = context.splitlines()
        outcome_line = lines[0] if lines else "Decision: PEND"
        fired = [ln[2:] for ln in lines if ln.startswith("- [")]
        missing = [ln[2:] for ln in lines if ln.startswith("- ") and not ln.startswith("- [")]
        body = f"{outcome_line}. "
        if fired:
            body += "This determination is based on the following policy rules: " + "; ".join(
                fired[:3]
            ) + ". "
        if missing:
            body += "Additional documentation required: " + "; ".join(missing[:3]) + "."
        return RationaleDraft(rationale=body.strip())

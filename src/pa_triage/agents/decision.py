"""Decision / Compliance node.

* **Outcome is deterministic.** :func:`decide_outcome` maps coverage facts to
  approve / deny / pend. The LLM never decides the outcome.
* **Rationale is LLM prose, strictly anchored.** The node acts as a "compliance
  copywriter": it is given the outcome, the exact rules that fired, the
  missing-info items, and a *de-identified* coded summary, and must explain only
  those, without external assumptions.
* **No PII leaves this node.** The raw narrative is never sent to the LLM here;
  only codes/descriptions are. As a final safety net the generated rationale is
  passed through the PII redactor, and the assembled decision is asserted to be
  free of any registered identifier before it is returned.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

from pa_triage.llm import get_llm
from pa_triage.logging_utils import get_logger, redactor
from pa_triage.models.domain import CodedClaim, CoverageResult, Outcome, TriageDecision
from pa_triage.models.state import AgentStep, TriageState

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models.chat_models import BaseChatModel

logger = get_logger(__name__)


class RationaleDraft(BaseModel):
    """Structured output: the human-readable rationale prose."""

    rationale: str = Field(
        description="2-4 sentence explanation that references ONLY the provided "
        "fired rules and missing-info items. No patient identifiers, no codes "
        "or facts not given."
    )


_SYSTEM_PROMPT = (
    "You are a prior-authorization compliance copywriter. Write a concise, "
    "professional rationale for the given decision. You MUST ground every claim "
    "in the provided 'rules that fired' and 'missing information' items, and you "
    "MUST explicitly reference the relevant rules. Do NOT introduce any external "
    "medical assumptions, new facts, codes, or patient identifiers. If the "
    "decision is 'pend', state what additional documentation is required."
)


def decide_outcome(coverage: CoverageResult) -> Outcome:
    """Deterministically map coverage facts to an outcome."""
    if not coverage.covered:
        return "deny"
    if coverage.missing_info:
        return "pend"
    return "approve"


def _build_context(outcome: Outcome, coverage: CoverageResult, coded: CodedClaim | None) -> str:
    parts: list[str] = [f"Decision: {outcome.upper()}"]
    if coded:
        dx = ", ".join(f"{d.description} ({d.icd10})" for d in coded.diagnoses) or "none"
        proc = ", ".join(f"{p.description} ({p.cpt})" for p in coded.procedures) or "none"
        parts.append(f"Coded diagnoses: {dx}")
        parts.append(f"Coded procedures: {proc}")
    rules = "\n".join(f"- [{r.effect}] {r.detail}" for r in coverage.fired_rules) or "- none"
    parts.append("Rules that fired:\n" + rules)
    if coverage.missing_info:
        parts.append("Missing information:\n" + "\n".join(f"- {m}" for m in coverage.missing_info))
    if coverage.ambiguity_notes:
        parts.append(f"Reviewer note on ambiguity: {coverage.ambiguity_notes}")
    return "\n".join(parts)


def generate_rationale(
    outcome: Outcome,
    coverage: CoverageResult,
    coded: CodedClaim | None,
    llm: "BaseChatModel",
) -> str:
    """Produce the anchored rationale prose, scrubbed of any registered PII."""
    structured = llm.with_structured_output(RationaleDraft)
    draft = structured.invoke(
        [("system", _SYSTEM_PROMPT), ("human", _build_context(outcome, coverage, coded))]
    )
    # Safety net: even though no PII was sent, scrub the output defensively.
    return redactor.redact(draft.rationale).strip()


def make_decision_node(
    llm: "BaseChatModel | None" = None,
) -> Callable[[TriageState], dict[str, Any]]:
    """Build the Decision/Compliance node with an injected LLM."""

    def decision_node(state: TriageState) -> dict[str, Any]:
        start = time.perf_counter()
        assert state.coverage is not None, "decision_node requires coverage state"
        active_llm = llm or get_llm()

        outcome = decide_outcome(state.coverage)
        rationale = generate_rationale(outcome, state.coverage, state.coded, active_llm)

        decision = TriageDecision(
            outcome=outcome,
            rationale=rationale,
            fired_rules=state.coverage.fired_rules,
            missing_info=state.coverage.missing_info,
        )

        # Compliance assertion: nothing in the final decision may contain a
        # registered identifier. redact() leaves clean text unchanged.
        serialized = decision.model_dump_json()
        if redactor.redact(serialized) != serialized:
            raise AssertionError("PII detected in final decision payload; refusing to emit.")

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info("Decision: outcome=%s (%d rule(s) fired).", outcome, len(decision.fired_rules))
        step = AgentStep(
            agent="decision",
            status="ok",
            summary=f"Outcome: {outcome.upper()}. {rationale}",
            output={
                "outcome": outcome,
                "rationale": rationale,
                "fired_rules": [r.model_dump() for r in decision.fired_rules],
                "missing_info": decision.missing_info,
            },
            latency_ms=latency_ms,
        )
        return {"decision": decision, "status": "complete", "trace": [step]}

    return decision_node

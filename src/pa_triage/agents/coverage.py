"""Coverage Checker node.

Deterministic where possible: the policy rules engine (:func:`evaluate_coverage`)
decides covered / not-covered / excluded / prior-auth purely from codes. The LLM
is used for exactly one ambiguous task — judging whether the free-text clinical
narrative documents each prior-authorization *criterion*. When a request has no
prior-auth criteria to weigh, the node runs fully deterministically and never
calls the LLM.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

from pa_triage.llm import get_llm
from pa_triage.logging_utils import get_logger
from pa_triage.models.domain import CodedClaim, ParsedBundle
from pa_triage.models.state import AgentStep, TriageState
from pa_triage.policy import CriteriaAssessments, Policy, evaluate_coverage, get_policy

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models.chat_models import BaseChatModel

logger = get_logger(__name__)


class CriterionJudgment(BaseModel):
    criterion: str = Field(description="The criterion text, echoed verbatim.")
    met: bool = Field(description="True only if the documentation clearly supports it.")
    evidence: str | None = Field(
        default=None, description="Brief supporting quote/paraphrase, or null if unmet."
    )


class CriteriaAssessment(BaseModel):
    """Structured output: a judgment for each prior-auth criterion."""

    judgments: list[CriterionJudgment] = Field(default_factory=list)
    notes: str | None = Field(
        default=None, description="Short overall note on ambiguity, if any."
    )


_SYSTEM_PROMPT = (
    "You are a prior-authorization criteria reviewer. You are given a clinical "
    "narrative and a list of medical-necessity criteria. For EACH criterion, "
    "decide whether the narrative explicitly documents it. Echo each criterion "
    "verbatim. Mark met=true ONLY when the documentation clearly supports it; "
    "if it is absent, vague, or merely planned, mark met=false. Do not infer "
    "beyond what is written."
)


def assess_criteria(
    narrative: str, criteria: list[str], llm: "BaseChatModel"
) -> CriteriaAssessment:
    """Ask the LLM to judge each criterion against the clinical narrative."""
    structured = llm.with_structured_output(CriteriaAssessment)
    criteria_block = "\n".join(f"- {c}" for c in criteria)
    human = (
        f"Clinical narrative:\n{narrative or '(no narrative provided)'}\n\n"
        f"Criteria to assess:\n{criteria_block}"
    )
    return structured.invoke([("system", _SYSTEM_PROMPT), ("human", human)])


def _criteria_needed(coded: CodedClaim, policy: Policy) -> dict[str, list[str]]:
    """Return {cpt: criteria} for covered, non-excluded, prior-auth procedures."""
    needed: dict[str, list[str]] = {}
    pa_map = policy.prior_auth_map
    for proc in coded.procedures:
        cpt = proc.cpt
        if cpt in policy.exclusion_map or cpt not in policy.covered_set:
            continue
        if cpt in pa_map and pa_map[cpt]:
            needed[cpt] = pa_map[cpt]
    return needed


def make_coverage_node(
    llm: "BaseChatModel | None" = None,
    policy: Policy | None = None,
) -> Callable[[TriageState], dict[str, Any]]:
    """Build the Coverage Checker node with injected dependencies."""

    def coverage_node(state: TriageState) -> dict[str, Any]:
        start = time.perf_counter()
        active_policy = policy or get_policy()
        assert state.coded is not None, "coverage_node requires coded state"
        parsed: ParsedBundle | None = state.parsed

        needed = _criteria_needed(state.coded, active_policy)
        criteria_assessments: CriteriaAssessments = {}
        ambiguity_notes: str | None = None
        used_llm = False

        if needed:
            used_llm = True
            active_llm = llm or get_llm()
            flat_criteria: list[str] = []
            for criteria in needed.values():
                for c in criteria:
                    if c not in flat_criteria:
                        flat_criteria.append(c)
            assessment = assess_criteria(
                parsed.clinical_notes if parsed else "", flat_criteria, active_llm
            )
            judged = {j.criterion: j.met for j in assessment.judgments}
            for cpt, criteria in needed.items():
                # Unmatched criteria default to False (conservative -> pend).
                criteria_assessments[cpt] = {c: judged.get(c, False) for c in criteria}
            ambiguity_notes = assessment.notes

        result = evaluate_coverage(
            state.coded, active_policy, criteria_assessments, ambiguity_notes
        )

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Coverage: covered=%s prior_auth=%s exclusions=%d missing=%d (llm_used=%s)",
            result.covered,
            result.prior_auth_required,
            len(result.exclusions_hit),
            len(result.missing_info),
            used_llm,
        )
        step = AgentStep(
            agent="coverage",
            status="ok",
            summary=(
                f"covered={result.covered}, prior_auth_required="
                f"{result.prior_auth_required}, "
                f"{len(result.exclusions_hit)} exclusion(s), "
                f"{len(result.missing_info)} missing-info item(s)."
            ),
            output={
                "covered": result.covered,
                "prior_auth_required": result.prior_auth_required,
                "exclusions_hit": result.exclusions_hit,
                "missing_info": result.missing_info,
                "fired_rules": [r.model_dump() for r in result.fired_rules],
                "llm_used": used_llm,
            },
            latency_ms=latency_ms,
        )
        return {"coverage": result, "trace": [step]}

    return coverage_node

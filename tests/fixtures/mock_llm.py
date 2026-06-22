"""A fake LangChain chat model for tests.

Tests register, per Pydantic schema, the structured value (or a callable that
receives the prompt messages and returns one) that ``.with_structured_output``
should yield. This lets a single fake serve every agent without touching a real
API.
"""

from __future__ import annotations

from typing import Any, Callable


class _FakeStructuredRunnable:
    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.calls: list[Any] = []

    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(messages)
        if callable(self._handler):
            return self._handler(messages)
        return self._handler


class FakeChatModel:
    """Minimal stand-in for a LangChain ``BaseChatModel``.

    Parameters
    ----------
    responses:
        Mapping of Pydantic schema class -> value or ``callable(messages)->value``
        returned by the runnable produced by ``with_structured_output(schema)``.
    """

    def __init__(self, responses: dict[type, Any] | None = None) -> None:
        self.responses = responses or {}
        self.structured_runnables: list[_FakeStructuredRunnable] = []

    def with_structured_output(self, schema: type, **_: Any) -> _FakeStructuredRunnable:
        if schema not in self.responses:
            raise KeyError(f"FakeChatModel has no response registered for {schema!r}")
        runnable = _FakeStructuredRunnable(self.responses[schema])
        self.structured_runnables.append(runnable)
        return runnable

    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError("FakeChatModel only supports structured output")


def scenario_llm() -> FakeChatModel:
    """A FakeChatModel that drives the full graph deterministically.

    * Clinical extraction returns empty term lists — the crosswalk backstop maps
      the claim-stated diagnoses/services, so end-to-end coding still works.
    * Criteria assessment inspects the narrative for signal phrases so that the
      'pend' and 'approve-after-prior-auth' samples diverge realistically.
    * Rationale generation echoes the decision.
    """
    # Imported here to avoid a hard dependency when the fixture is unused.
    from pa_triage.agents.clinical import ClinicalExtraction
    from pa_triage.agents.coverage import CriteriaAssessment, CriterionJudgment
    from pa_triage.agents.decision import RationaleDraft

    def assess(messages: Any) -> CriteriaAssessment:
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
                judgments.append(CriterionJudgment(criterion=criterion, met=met_therapy))
            elif "neurologic" in low or "deficit" in low or "radiculopathy" in low:
                judgments.append(CriterionJudgment(criterion=criterion, met=met_deficit))
            else:
                judgments.append(CriterionJudgment(criterion=criterion, met=False))
        return CriteriaAssessment(judgments=judgments, notes="Assessed against narrative.")

    def rationale(messages: Any) -> RationaleDraft:
        context = messages[-1][1]
        first = context.splitlines()[0]  # "Decision: APPROVE"
        return RationaleDraft(rationale=f"{first}. Determination based on the rules that fired.")

    return FakeChatModel(
        {
            ClinicalExtraction: ClinicalExtraction(diagnosis_terms=[], procedure_terms=[]),
            CriteriaAssessment: assess,
            RationaleDraft: rationale,
        }
    )

"""Tests for the Decision/Compliance node (LLM mocked)."""

from __future__ import annotations

import pytest

from pa_triage.agents.decision import (
    RationaleDraft,
    decide_outcome,
    make_decision_node,
)
from pa_triage.logging_utils import redactor
from pa_triage.models.domain import (
    CodedClaim,
    CoverageResult,
    Diagnosis,
    FiredRule,
    Procedure,
)
from pa_triage.models.state import TriageState

from tests.fixtures.mock_llm import FakeChatModel


def _coverage(covered=True, missing=None, exclusions=None, rules=None):
    return CoverageResult(
        covered=covered,
        prior_auth_required=False,
        exclusions_hit=exclusions or [],
        missing_info=missing or [],
        fired_rules=rules or [],
    )


def test_decide_outcome_mapping():
    assert decide_outcome(_coverage(covered=True, missing=[])) == "approve"
    assert decide_outcome(_coverage(covered=True, missing=["x"])) == "pend"
    assert decide_outcome(_coverage(covered=False, exclusions=["x"])) == "deny"


def test_decision_node_assembles_decision():
    rules = [FiredRule(rule_id="COV-29881", effect="covered", detail="covered service")]
    state = TriageState(
        raw_bundle={},
        coverage=_coverage(rules=rules),
        coded=CodedClaim(
            diagnoses=[Diagnosis(icd10="M17.11", description="oa", source_text="x")],
            procedures=[Procedure(cpt="29881", description="knee scope", source_text="x")],
        ),
    )
    llm = FakeChatModel(
        {RationaleDraft: RationaleDraft(rationale="Approved: the requested service is covered.")}
    )
    result = make_decision_node(llm=llm)(state)
    decision = result["decision"]
    assert decision.outcome == "approve"
    assert "covered" in decision.rationale.lower()
    assert decision.fired_rules == rules
    assert result["status"] == "complete"


def test_decision_node_scrubs_pii_from_rationale():
    # Simulate intake having registered a patient name.
    redactor.register("Marcus Reyes")
    state = TriageState(raw_bundle={}, coverage=_coverage(), coded=None)
    # A misbehaving LLM that leaks a name into the rationale.
    llm = FakeChatModel(
        {RationaleDraft: RationaleDraft(rationale="Approved for Marcus Reyes per policy.")}
    )
    result = make_decision_node(llm=llm)(state)
    rationale = result["decision"].rationale
    assert "Marcus Reyes" not in rationale
    assert "[REDACTED]" in rationale


def test_pend_outcome_when_missing_info():
    state = TriageState(
        raw_bundle={},
        coverage=_coverage(missing=["Need 6 weeks conservative therapy documentation"]),
        coded=None,
    )
    llm = FakeChatModel(
        {RationaleDraft: RationaleDraft(rationale="Pended pending additional documentation.")}
    )
    decision = make_decision_node(llm=llm)(state)["decision"]
    assert decision.outcome == "pend"
    assert decision.missing_info

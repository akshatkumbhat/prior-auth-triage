"""Tests for the Coverage Checker node (LLM mocked)."""

from __future__ import annotations

from pa_triage.agents.coverage import (
    CriteriaAssessment,
    CriterionJudgment,
    make_coverage_node,
)
from pa_triage.models.domain import CodedClaim, Diagnosis, Procedure
from pa_triage.models.state import TriageState
from pa_triage.policy import load_policy

from tests.fixtures.mock_llm import FakeChatModel


def _state(coded, notes="note"):
    from pa_triage.models.domain import ParsedBundle

    parsed = ParsedBundle(claim_id="c", subject_uuid="subj-x", clinical_notes=notes)
    return TriageState(raw_bundle={}, parsed=parsed, coded=coded)


def test_deterministic_path_does_not_call_llm():
    # Knee arthroscopy: covered, no prior auth -> LLM must not be invoked.
    coded = CodedClaim(
        diagnoses=[Diagnosis(icd10="M17.11", description="oa", source_text="oa")],
        procedures=[Procedure(cpt="29881", description="knee scope", source_text="x")],
    )
    # FakeChatModel with no registered responses raises if with_structured_output is called.
    llm = FakeChatModel({})
    node = make_coverage_node(llm=llm, policy=load_policy())
    result = node(_state(coded))
    assert result["coverage"].covered is True
    assert result["coverage"].prior_auth_required is False
    assert result["trace"][0].output["llm_used"] is False


def test_prior_auth_path_uses_llm_and_pends_when_unmet():
    policy = load_policy()
    criteria = policy.prior_auth_map["72148"]
    coded = CodedClaim(
        diagnoses=[Diagnosis(icd10="M54.16", description="radic", source_text="x")],
        procedures=[Procedure(cpt="72148", description="MRI L-spine", source_text="x")],
    )
    assessment = CriteriaAssessment(
        judgments=[CriterionJudgment(criterion=c, met=False) for c in criteria]
    )
    llm = FakeChatModel({CriteriaAssessment: assessment})
    node = make_coverage_node(llm=llm, policy=policy)
    result = node(_state(coded, notes="acute back pain, no PT tried"))
    cov = result["coverage"]
    assert cov.prior_auth_required is True
    assert cov.missing_info  # criteria unmet
    assert result["trace"][0].output["llm_used"] is True


def test_prior_auth_path_clears_when_criteria_met():
    policy = load_policy()
    criteria = policy.prior_auth_map["72148"]
    coded = CodedClaim(
        diagnoses=[Diagnosis(icd10="M54.16", description="radic", source_text="x")],
        procedures=[Procedure(cpt="72148", description="MRI L-spine", source_text="x")],
    )
    assessment = CriteriaAssessment(
        judgments=[CriterionJudgment(criterion=c, met=True) for c in criteria],
        notes="All criteria documented.",
    )
    llm = FakeChatModel({CriteriaAssessment: assessment})
    node = make_coverage_node(llm=llm, policy=policy)
    cov = node(_state(coded, notes="8 weeks PT, foot drop"))["coverage"]
    assert cov.prior_auth_required is True
    assert cov.missing_info == []
    assert cov.ambiguity_notes == "All criteria documented."

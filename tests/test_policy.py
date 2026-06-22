"""Tests for the deterministic coverage rules engine."""

from __future__ import annotations

import pytest

from pa_triage.models.domain import CodedClaim, Diagnosis, Procedure
from pa_triage.policy import evaluate_coverage, load_policy


@pytest.fixture
def policy():
    return load_policy()


def _dx(icd10="M54.16"):
    return Diagnosis(icd10=icd10, description="dx", source_text="dx")


def _proc(cpt, desc="proc"):
    return Procedure(cpt=cpt, description=desc, source_text=desc)


def test_covered_no_prior_auth_is_clean(policy):
    coded = CodedClaim(diagnoses=[_dx("M17.11")], procedures=[_proc("29881")])
    result = evaluate_coverage(coded, policy)
    assert result.covered is True
    assert result.prior_auth_required is False
    assert result.missing_info == []
    assert any(r.rule_id == "COV-29881" for r in result.fired_rules)


def test_exclusion_marks_not_covered(policy):
    coded = CodedClaim(diagnoses=[_dx("H02.835")], procedures=[_proc("15823")])
    result = evaluate_coverage(coded, policy)
    assert result.covered is False
    assert result.exclusions_hit
    assert any(r.effect == "exclusion" for r in result.fired_rules)


def test_uncovered_service_marks_not_covered(policy):
    coded = CodedClaim(diagnoses=[_dx("M54.50")], procedures=[_proc("97810")])
    result = evaluate_coverage(coded, policy)
    assert result.covered is False
    assert any(r.rule_id == "NOTCOV-97810" for r in result.fired_rules)


def test_prior_auth_criteria_met_has_no_missing_info(policy):
    coded = CodedClaim(diagnoses=[_dx()], procedures=[_proc("72148")])
    criteria = policy.prior_auth_map["72148"]
    assessments = {"72148": {c: True for c in criteria}}
    result = evaluate_coverage(coded, policy, assessments)
    assert result.covered is True
    assert result.prior_auth_required is True
    assert result.missing_info == []


def test_prior_auth_criteria_unmet_yields_missing_info(policy):
    coded = CodedClaim(diagnoses=[_dx()], procedures=[_proc("72148")])
    criteria = policy.prior_auth_map["72148"]
    assessments = {"72148": {c: False for c in criteria}}
    result = evaluate_coverage(coded, policy, assessments)
    assert result.covered is True
    assert result.prior_auth_required is True
    assert len(result.missing_info) == len(criteria)


def test_missing_diagnosis_flagged_when_required(policy):
    coded = CodedClaim(diagnoses=[], procedures=[_proc("29881")])
    result = evaluate_coverage(coded, policy)
    assert any(r.rule_id == "DX-REQUIRED" for r in result.fired_rules)
    assert result.missing_info


def test_no_procedures_flagged(policy):
    coded = CodedClaim(diagnoses=[_dx()], procedures=[])
    result = evaluate_coverage(coded, policy)
    assert any(r.rule_id == "PROC-MISSING" for r in result.fired_rules)

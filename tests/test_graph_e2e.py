"""End-to-end tests over the compiled LangGraph (LLM mocked)."""

from __future__ import annotations

import pytest

from pa_triage.crosswalk import load_crosswalk
from pa_triage.pipeline import run_triage, stream_triage
from pa_triage.policy import load_policy

from tests.fixtures.mock_llm import scenario_llm


@pytest.fixture
def deps():
    return {
        "llm": scenario_llm(),
        "crosswalk": load_crosswalk(),
        "policy": load_policy(),
    }


@pytest.mark.parametrize(
    "sample, expected_outcome",
    [
        ("01_approve_clean", "approve"),
        ("02_deny_exclusion", "deny"),
        ("03_pend_missing_info", "pend"),
        ("04_prior_auth_required", "approve"),
        ("05_deny_not_covered", "deny"),
    ],
)
def test_end_to_end_outcomes(load_sample, deps, sample, expected_outcome):
    # Fresh LLM per run so internal call logs don't bleed across cases.
    deps = {**deps, "llm": scenario_llm()}
    state = run_triage(load_sample(sample), **deps)
    assert state.status == "complete"
    assert state.decision is not None
    assert state.decision.outcome == expected_outcome
    # All four agents recorded a step.
    assert [s.agent for s in state.trace] == ["intake", "clinical", "coverage", "decision"]
    assert state.decision.fired_rules  # rationale is anchored to fired rules


def test_prior_auth_distinguishes_pend_vs_approve(load_sample, deps):
    pend = run_triage(load_sample("03_pend_missing_info"), **{**deps, "llm": scenario_llm()})
    approve = run_triage(load_sample("04_prior_auth_required"), **{**deps, "llm": scenario_llm()})
    assert pend.coverage.prior_auth_required is True
    assert pend.coverage.missing_info  # criteria not documented
    assert approve.coverage.prior_auth_required is True
    assert approve.coverage.missing_info == []  # criteria documented


def test_no_pii_in_final_state(load_sample, deps):
    # Run the sample with the most narrative; assert identifiers never surface.
    state = run_triage(load_sample("04_prior_auth_required"), **{**deps, "llm": scenario_llm()})
    blob = state.decision.model_dump_json()
    for trace in state.trace:
        blob += trace.model_dump_json()
    assert "Castellano" not in blob
    assert "MRN-9087345" not in blob
    assert "1959-07-21" not in blob


def test_malformed_bundle_is_rejected_before_llm(load_sample, deps):
    state = run_triage({"resourceType": "Observation"}, **{**deps, "llm": scenario_llm()})
    assert state.status == "rejected"
    assert state.decision is None
    assert [s.agent for s in state.trace] == ["intake"]
    assert state.trace[0].status == "rejected"


def test_stream_yields_progressive_snapshots(load_sample, deps):
    snapshots = list(stream_triage(load_sample("01_approve_clean"), **{**deps, "llm": scenario_llm()}))
    # Trace grows monotonically as nodes execute.
    trace_lengths = [len(s.trace) for s in snapshots]
    assert trace_lengths == sorted(trace_lengths)
    assert snapshots[-1].decision is not None
    assert snapshots[-1].decision.outcome == "approve"

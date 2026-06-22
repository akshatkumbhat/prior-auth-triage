"""Tests for the Intake/Parser node and PII sanitization gateway."""

from __future__ import annotations

import logging

import pytest

from pa_triage.agents.intake import (
    BundleValidationError,
    intake_node,
    parse_and_sanitize,
)
from pa_triage.logging_utils import get_logger
from pa_triage.models.state import TriageState


def test_parses_valid_bundle(load_sample):
    parsed = parse_and_sanitize(load_sample("01_approve_clean"))
    assert parsed.claim_id == "claim-001"
    assert parsed.subject_uuid.startswith("subj-")
    assert any("meniscectomy" in i.description.lower() for i in parsed.requested_items)
    assert parsed.stated_diagnoses == ["Right knee osteoarthritis"]
    assert parsed.plan_label == "Synthetic Health Plan PPO"
    assert parsed.clinical_notes and "physical therapy" in parsed.clinical_notes


def test_subject_uuid_is_stable_and_anonymized(load_sample):
    raw = load_sample("01_approve_clean")
    first = parse_and_sanitize(raw)
    second = parse_and_sanitize(raw)
    assert first.subject_uuid == second.subject_uuid  # stable across runs
    # No direct identifiers leak into the parsed model.
    blob = first.model_dump_json()
    assert "Reyes" not in blob
    assert "MRN-7781234" not in blob
    assert "1968-04-12" not in blob


@pytest.mark.parametrize(
    "sample",
    [
        "01_approve_clean",
        "02_deny_exclusion",
        "03_pend_missing_info",
        "04_prior_auth_required",
        "05_deny_not_covered",
    ],
)
def test_all_samples_parse(load_sample, sample):
    parsed = parse_and_sanitize(load_sample(sample))
    assert parsed.claim_id
    assert parsed.requested_items


def test_rejects_non_bundle():
    with pytest.raises(BundleValidationError):
        parse_and_sanitize({"resourceType": "Patient", "id": "x"})


def test_rejects_non_dict():
    with pytest.raises(BundleValidationError):
        parse_and_sanitize(["not", "a", "bundle"])  # type: ignore[arg-type]


def test_rejects_bundle_without_claim():
    raw = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": {"resourceType": "Patient", "id": "p1"}}],
    }
    with pytest.raises(BundleValidationError, match="no Claim"):
        parse_and_sanitize(raw)


def test_intake_node_emits_masked_log_and_no_pii(load_sample, caplog):
    # Logger must carry the redacting filter even under caplog capture.
    target = get_logger("pa_triage.agents.intake")
    caplog.handler.addFilter(target.filters[0])
    with caplog.at_level(logging.INFO, logger="pa_triage.agents.intake"):
        result = intake_node(TriageState(raw_bundle=load_sample("01_approve_clean")))

    assert result["parsed"].claim_id == "claim-001"
    assert result["trace"][0].status == "ok"
    log_text = caplog.text
    # Masked confirmation present, raw identifiers absent.
    assert "Reyes" not in log_text
    assert "MRN-7781234" not in log_text


def test_intake_node_rejects_gracefully():
    result = intake_node(TriageState(raw_bundle={"resourceType": "Observation"}))
    assert result["status"] == "rejected"
    assert result["errors"]
    assert result["trace"][0].status == "rejected"

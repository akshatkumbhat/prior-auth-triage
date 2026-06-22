"""Tests for the Clinical Reviewer node (LLM mocked)."""

from __future__ import annotations

from pa_triage.agents.clinical import (
    ClinicalExtraction,
    code_claim,
    make_clinical_node,
)
from pa_triage.agents.intake import parse_and_sanitize
from pa_triage.crosswalk import load_crosswalk
from pa_triage.models.state import TriageState

from tests.fixtures.mock_llm import FakeChatModel


def _state(load_sample, name):
    parsed = parse_and_sanitize(load_sample(name))
    return TriageState(raw_bundle={"resourceType": "Bundle"}, parsed=parsed)


def test_clinical_node_maps_llm_terms_to_codes(load_sample):
    parsed = parse_and_sanitize(load_sample("04_prior_auth_required"))
    # LLM extracts terms only from the narrative; codes come from the crosswalk.
    extraction = ClinicalExtraction(
        diagnosis_terms=["lumbar radiculopathy"],
        procedure_terms=["MRI lumbar spine without contrast"],
    )
    llm = FakeChatModel({ClinicalExtraction: extraction})
    node = make_clinical_node(llm=llm, crosswalk=load_crosswalk())

    result = node(TriageState(raw_bundle={}, parsed=parsed))
    coded = result["coded"]
    assert [d.icd10 for d in coded.diagnoses] == ["M54.16"]
    assert [p.cpt for p in coded.procedures] == ["72148"]
    assert result["trace"][0].agent == "clinical"


def test_code_claim_tracks_unmapped_terms(load_sample):
    parsed = parse_and_sanitize(load_sample("01_approve_clean"))
    extraction = ClinicalExtraction(
        diagnosis_terms=["right knee osteoarthritis"],
        procedure_terms=["experimental nanobot infusion"],  # not in crosswalk
    )
    coded = code_claim(extraction, parsed, load_crosswalk())
    assert "M17.11" in [d.icd10 for d in coded.diagnoses]
    assert "experimental nanobot infusion" in coded.unmapped
    # The claim-stated meniscectomy is still picked up as a backstop.
    assert "29881" in [p.cpt for p in coded.procedures]


def test_code_claim_dedupes_by_code(load_sample):
    parsed = parse_and_sanitize(load_sample("03_pend_missing_info"))
    # LLM and claim both yield the same MRI -> single 72148 entry.
    extraction = ClinicalExtraction(
        diagnosis_terms=["lumbar radiculopathy"],
        procedure_terms=["MRI lumbar spine"],
    )
    coded = code_claim(extraction, parsed, load_crosswalk())
    cpts = [p.cpt for p in coded.procedures]
    assert cpts.count("72148") == 1

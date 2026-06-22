"""Tests for the deterministic ICD-10 / CPT crosswalk matcher."""

from __future__ import annotations

from pa_triage.crosswalk import load_crosswalk


def test_maps_known_procedure_despite_filler_words():
    cw = load_crosswalk()
    # "of the" filler should not break token-subset matching.
    proc = cw.map_procedure("MRI of the lumbar spine without contrast")
    assert proc is not None
    assert proc.cpt == "72148"


def test_maps_known_diagnosis_alias():
    cw = load_crosswalk()
    dx = cw.map_diagnosis("sciatica")
    assert dx is not None
    assert dx.icd10 == "M54.16"


def test_unknown_term_returns_none():
    cw = load_crosswalk()
    assert cw.map_procedure("teleportation therapy") is None
    assert cw.map_diagnosis("dragon pox") is None


def test_match_prefers_most_specific_entry():
    cw = load_crosswalk()
    # Should map to knee arthroscopy w/ meniscectomy, not a partial single-word hit.
    proc = cw.map_procedure("arthroscopic partial meniscectomy of the right knee")
    assert proc is not None
    assert proc.cpt == "29881"


def test_source_text_is_preserved():
    cw = load_crosswalk()
    dx = cw.map_diagnosis("lumbar radiculopathy with leg pain")
    assert dx is not None
    assert dx.source_text == "lumbar radiculopathy with leg pain"

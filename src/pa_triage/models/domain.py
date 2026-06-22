"""Domain models that flow between agents.

These are deliberately decoupled from the raw FHIR shapes: the Intake node
projects FHIR into these de-identified, operational models, and every
downstream agent works against them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Outcome = Literal["approve", "deny", "pend"]


class RequestedItem(BaseModel):
    """A procedure/service requested on the claim, as it appears pre-coding."""

    model_config = ConfigDict(frozen=True)

    sequence: int | None = None
    description: str
    raw_code: str | None = None  # code already present on the claim, if any
    raw_system: str | None = None


class ParsedBundle(BaseModel):
    """Intake output: a de-identified, typed projection of the FHIR bundle.

    Note the absence of name / MRN / DOB fields. The Intake gateway reads those
    from the raw bundle, logs a masked confirmation, registers them with the
    PII redactor, and maps the patient to an anonymized ``subject_uuid`` only.
    The raw ``clinical_notes`` narrative is retained for the Clinical Reviewer.
    """

    claim_id: str
    subject_uuid: str
    encounter_class: str | None = None
    reason_texts: list[str] = Field(default_factory=list)
    requested_items: list[RequestedItem] = Field(default_factory=list)
    stated_diagnoses: list[str] = Field(default_factory=list)
    coverage_id: str | None = None
    plan_label: str | None = None
    clinical_notes: str | None = None


class Diagnosis(BaseModel):
    model_config = ConfigDict(frozen=True)

    icd10: str
    description: str
    source_text: str


class Procedure(BaseModel):
    model_config = ConfigDict(frozen=True)

    cpt: str
    description: str
    source_text: str


class CodedClaim(BaseModel):
    """Clinical Reviewer output: claim mapped to ICD-10 / CPT codes."""

    diagnoses: list[Diagnosis] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    unmapped: list[str] = Field(default_factory=list)


class FiredRule(BaseModel):
    """A single policy rule that fired during coverage evaluation."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    effect: Literal["covered", "not_covered", "prior_auth_required", "exclusion", "info"]
    detail: str


class CoverageResult(BaseModel):
    """Coverage Checker output: deterministic facts about coverage."""

    covered: bool
    prior_auth_required: bool
    exclusions_hit: list[str] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    fired_rules: list[FiredRule] = Field(default_factory=list)
    ambiguity_notes: str | None = None


class TriageDecision(BaseModel):
    """Decision/Compliance output: the final structured decision."""

    outcome: Outcome
    rationale: str
    fired_rules: list[FiredRule] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)

"""Intake / Parser node — the PII sanitization gateway.

Responsibilities:
  1. Validate and parse the raw FHIR bundle into typed resources.
  2. Reject malformed bundles with a clear, actionable error.
  3. Act as the PII gateway: read patient name / MRN / DOB, register them with
     the process-wide redactor, emit a *masked* confirmation log, and then map
     the patient to an anonymized ``subject_uuid``. Those direct identifiers
     never enter :class:`ParsedBundle`.

The raw clinical narrative is intentionally retained in ``clinical_notes`` so
the Clinical Reviewer can reason over it; the logging redactor ensures it
cannot leak into log output.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pa_triage.logging_utils import get_logger, mask_name, redactor
from pa_triage.models.domain import ParsedBundle, RequestedItem
from pa_triage.models.fhir import Bundle, Claim, Coverage, Encounter, Patient
from pa_triage.models.state import AgentStep, TriageState

logger = get_logger(__name__)

# Stable namespace so a given Patient maps to the same anonymized token across
# runs (useful for traceability) without revealing the underlying identifier.
_SUBJECT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "pa-triage.subject")


class BundleValidationError(ValueError):
    """Raised when a bundle is structurally invalid and cannot be triaged."""


def _index_resources(bundle: Bundle) -> dict[str, list[dict[str, Any]]]:
    """Group raw resource dicts in a bundle by their ``resourceType``."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in bundle.entry:
        resource = entry.resource
        if not resource:
            continue
        rtype = resource.get("resourceType")
        if rtype:
            grouped.setdefault(rtype, []).append(resource)
    return grouped


def _anonymize_subject(patient: Patient | None) -> str:
    """Return a stable, de-identified subject token for a patient."""
    seed = None
    if patient is not None:
        seed = patient.id or patient.mrn() or patient.primary_name()
    if not seed:
        # No usable identifier; fall back to a random token.
        return f"subj-{uuid.uuid4().hex[:12]}"
    return f"subj-{uuid.uuid5(_SUBJECT_NAMESPACE, str(seed)).hex[:12]}"


def parse_and_sanitize(raw_bundle: dict[str, Any]) -> ParsedBundle:
    """Parse a raw FHIR bundle into a de-identified :class:`ParsedBundle`.

    Raises
    ------
    BundleValidationError
        If the payload is not a usable FHIR Bundle containing a Claim.
    """
    if not isinstance(raw_bundle, dict):
        raise BundleValidationError("Input is not a JSON object / FHIR resource.")
    if raw_bundle.get("resourceType") != "Bundle":
        raise BundleValidationError(
            f"Expected a FHIR Bundle, got resourceType="
            f"{raw_bundle.get('resourceType')!r}."
        )

    try:
        bundle = Bundle.model_validate(raw_bundle)
    except Exception as exc:  # pydantic ValidationError and friends
        raise BundleValidationError(f"Bundle failed schema validation: {exc}") from exc

    grouped = _index_resources(bundle)
    if "Claim" not in grouped:
        raise BundleValidationError("Bundle contains no Claim resource to triage.")

    claim = Claim.model_validate(grouped["Claim"][0])
    if not claim.id:
        raise BundleValidationError("Claim resource is missing an id.")

    patient = Patient.model_validate(grouped["Patient"][0]) if "Patient" in grouped else None
    encounter = (
        Encounter.model_validate(grouped["Encounter"][0]) if "Encounter" in grouped else None
    )
    coverage = Coverage.model_validate(grouped["Coverage"][0]) if "Coverage" in grouped else None

    # --- PII gateway -----------------------------------------------------
    # Register direct identifiers and the raw narrative with the redactor so
    # they are masked everywhere in the logs, then log only a masked summary.
    clinical_notes = claim.clinical_note()
    if patient is not None:
        name = patient.primary_name()
        mrn = patient.mrn()
        redactor.register(name, mrn, patient.birthDate)
        redactor.register(clinical_notes)
        logger.info(
            "Intake: sanitized patient (name=%s, mrn=%s) -> %s",
            mask_name(name),
            f"***{mrn[-4:]}" if mrn else "[REDACTED]",
            _anonymize_subject(patient),
        )
    else:
        redactor.register(clinical_notes)
        logger.info("Intake: no Patient resource present; using random subject token.")

    requested_items = [
        RequestedItem(
            sequence=item.sequence,
            description=(item.productOrService.best_label() if item.productOrService else "")
            or "(unspecified service)",
            raw_code=(
                item.productOrService.coding[0].code
                if item.productOrService and item.productOrService.coding
                else None
            ),
            raw_system=(
                item.productOrService.coding[0].system
                if item.productOrService and item.productOrService.coding
                else None
            ),
        )
        for item in claim.item
    ]

    stated_diagnoses = [
        d.diagnosisCodeableConcept.best_label()
        for d in claim.diagnosis
        if d.diagnosisCodeableConcept and d.diagnosisCodeableConcept.best_label()
    ]

    reason_texts = [rc.best_label() for rc in encounter.reasonCode if rc.best_label()] if encounter else []

    return ParsedBundle(
        claim_id=claim.id,
        subject_uuid=_anonymize_subject(patient),
        encounter_class=(encounter.class_.display or encounter.class_.code) if encounter and encounter.class_ else None,
        reason_texts=reason_texts,
        requested_items=requested_items,
        stated_diagnoses=stated_diagnoses,
        coverage_id=coverage.id if coverage else None,
        plan_label=coverage.plan_label() if coverage else None,
        clinical_notes=clinical_notes,
    )


def intake_node(state: TriageState) -> dict[str, Any]:
    """LangGraph node wrapping :func:`parse_and_sanitize` with trace + timing."""
    start = time.perf_counter()
    try:
        parsed = parse_and_sanitize(state.raw_bundle)
    except BundleValidationError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        step = AgentStep(
            agent="intake",
            status="rejected",
            summary=f"Bundle rejected: {exc}",
            output={"error": str(exc)},
            latency_ms=latency_ms,
        )
        return {"status": "rejected", "errors": [str(exc)], "trace": [step]}

    latency_ms = (time.perf_counter() - start) * 1000
    step = AgentStep(
        agent="intake",
        status="ok",
        summary=(
            f"Parsed claim {parsed.claim_id} for {parsed.subject_uuid}: "
            f"{len(parsed.requested_items)} requested service(s), "
            f"{len(parsed.stated_diagnoses)} stated diagnosis/es."
        ),
        output={
            "claim_id": parsed.claim_id,
            "subject_uuid": parsed.subject_uuid,
            "requested_items": [i.description for i in parsed.requested_items],
            "stated_diagnoses": parsed.stated_diagnoses,
            "plan_label": parsed.plan_label,
        },
        latency_ms=latency_ms,
    )
    return {"parsed": parsed, "trace": [step]}

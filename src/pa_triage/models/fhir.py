"""Hand-rolled Pydantic subset of FHIR R4.

We model only the fields of the four resource types this PoC consumes
(Patient, Encounter, Claim, Coverage) plus the Bundle envelope. This is a
deliberate tradeoff: it is lighter than a full FHIR library and keeps the
validation legible, at the cost of not being a complete R4 validator. Unknown
fields are allowed (``extra="allow"``) so realistic bundles parse cleanly.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _FhirBase(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# --- Datatypes -----------------------------------------------------------

class Coding(_FhirBase):
    system: str | None = None
    code: str | None = None
    display: str | None = None


class CodeableConcept(_FhirBase):
    coding: list[Coding] = Field(default_factory=list)
    text: str | None = None

    def best_label(self) -> str | None:
        """Human label: prefer ``text``, else first coding display/code."""
        if self.text:
            return self.text
        for c in self.coding:
            if c.display:
                return c.display
            if c.code:
                return c.code
        return None


class Identifier(_FhirBase):
    system: str | None = None
    value: str | None = None
    # FHIR Identifier.type is a CodeableConcept (e.g. MR = medical record number)
    type: CodeableConcept | None = None


class HumanName(_FhirBase):
    family: str | None = None
    given: list[str] = Field(default_factory=list)
    text: str | None = None

    def full_name(self) -> str | None:
        if self.text:
            return self.text
        parts = [*self.given, self.family] if self.family else list(self.given)
        joined = " ".join(p for p in parts if p)
        return joined or None


class Reference(_FhirBase):
    reference: str | None = None  # e.g. "Patient/abc"
    display: str | None = None


# --- Resources -----------------------------------------------------------

class Patient(_FhirBase):
    resourceType: Literal["Patient"]
    id: str | None = None
    name: list[HumanName] = Field(default_factory=list)
    identifier: list[Identifier] = Field(default_factory=list)
    birthDate: str | None = None
    gender: str | None = None

    def primary_name(self) -> str | None:
        return self.name[0].full_name() if self.name else None

    def mrn(self) -> str | None:
        """Return the medical record number identifier value, if present."""
        for ident in self.identifier:
            type_text = (ident.type.best_label() if ident.type else None) or ""
            if "MR" in type_text.upper() or "MEDICAL RECORD" in type_text.upper():
                return ident.value
        # Fall back to the first identifier value if no typed MRN found.
        return self.identifier[0].value if self.identifier else None


class Encounter(_FhirBase):
    resourceType: Literal["Encounter"]
    id: str | None = None
    status: str | None = None
    class_: Coding | None = Field(default=None, alias="class")
    subject: Reference | None = None
    reasonCode: list[CodeableConcept] = Field(default_factory=list)


class ClaimItem(_FhirBase):
    sequence: int | None = None
    productOrService: CodeableConcept | None = None


class ClaimDiagnosis(_FhirBase):
    sequence: int | None = None
    diagnosisCodeableConcept: CodeableConcept | None = None


class ClaimSupportingInfo(_FhirBase):
    sequence: int | None = None
    category: CodeableConcept | None = None
    valueString: str | None = None


class ClaimInsurance(_FhirBase):
    sequence: int | None = None
    focal: bool | None = None
    coverage: Reference | None = None


class Claim(_FhirBase):
    resourceType: Literal["Claim"]
    id: str | None = None
    status: str | None = None
    use: str | None = None
    patient: Reference | None = None
    insurance: list[ClaimInsurance] = Field(default_factory=list)
    item: list[ClaimItem] = Field(default_factory=list)
    diagnosis: list[ClaimDiagnosis] = Field(default_factory=list)
    supportingInfo: list[ClaimSupportingInfo] = Field(default_factory=list)

    def clinical_note(self) -> str | None:
        """Return the free-text clinical narrative from supportingInfo, if any.

        Picks the supportingInfo entry categorized as a clinical note; falls
        back to the first valueString present.
        """
        fallback: str | None = None
        for info in self.supportingInfo:
            if info.valueString is None:
                continue
            fallback = fallback or info.valueString
            label = (info.category.best_label() if info.category else "") or ""
            if "note" in label.lower() or "clinical" in label.lower():
                return info.valueString
        return fallback


class Coverage(_FhirBase):
    resourceType: Literal["Coverage"]
    id: str | None = None
    status: str | None = None
    type: CodeableConcept | None = None
    subscriberId: str | None = None
    beneficiary: Reference | None = None
    payor: list[Reference] = Field(default_factory=list)

    def plan_label(self) -> str | None:
        return self.type.best_label() if self.type else None


# --- Bundle --------------------------------------------------------------

class BundleEntry(_FhirBase):
    fullUrl: str | None = None
    resource: dict[str, Any] | None = None


class Bundle(_FhirBase):
    resourceType: Literal["Bundle"]
    type: str | None = None
    entry: list[BundleEntry] = Field(default_factory=list)


# Resource registry for dispatch when reading bundle entries.
ResourceType = Annotated[Patient | Encounter | Claim | Coverage, "fhir-resource"]

_RESOURCE_MODELS: dict[str, type[_FhirBase]] = {
    "Patient": Patient,
    "Encounter": Encounter,
    "Claim": Claim,
    "Coverage": Coverage,
}


def resource_model_for(resource_type: str) -> type[_FhirBase] | None:
    """Return the model class for a FHIR ``resourceType``, or None if unmodeled."""
    return _RESOURCE_MODELS.get(resource_type)

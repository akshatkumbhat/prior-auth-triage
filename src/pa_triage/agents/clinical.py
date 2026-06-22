"""Clinical Reviewer node.

Two clearly separated responsibilities:

* **LLM (fuzzy):** read the clinical narrative + claim text and extract concise
  diagnosis and procedure *terms*. This is the only non-deterministic step.
* **Crosswalk (deterministic):** map each extracted term to an ICD-10 / CPT
  code. Explicit claim-stated diagnoses and requested services are folded in as
  a backstop so the mapping is resilient even if the LLM misses something.

The node is built via :func:`make_clinical_node` so the LLM and crosswalk are
injected (the graph passes real ones; tests pass fakes).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

from pa_triage.crosswalk import Crosswalk, get_crosswalk
from pa_triage.llm import get_llm
from pa_triage.logging_utils import get_logger
from pa_triage.models.domain import CodedClaim, ParsedBundle
from pa_triage.models.state import AgentStep, TriageState

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models.chat_models import BaseChatModel

logger = get_logger(__name__)


class ClinicalExtraction(BaseModel):
    """Structured output the LLM must return."""

    diagnosis_terms: list[str] = Field(
        default_factory=list,
        description="Concise clinical diagnosis terms found in the text "
        "(e.g. 'lumbar radiculopathy'). Do not invent codes.",
    )
    procedure_terms: list[str] = Field(
        default_factory=list,
        description="Concise requested procedure/service terms "
        "(e.g. 'MRI lumbar spine without contrast').",
    )


_SYSTEM_PROMPT = (
    "You are a clinical coding assistant for a prior-authorization workflow. "
    "Read the provided claim text and clinical narrative and extract the "
    "diagnoses and the requested procedures/services as short canonical "
    "clinical terms. Extract only what is explicitly supported by the text. "
    "Do NOT output ICD-10 or CPT codes; output plain clinical terms only. "
    "Do NOT include patient identifiers."
)


def _build_context(parsed: ParsedBundle) -> str:
    lines: list[str] = []
    if parsed.stated_diagnoses:
        lines.append("Stated diagnoses: " + "; ".join(parsed.stated_diagnoses))
    if parsed.requested_items:
        lines.append(
            "Requested services: "
            + "; ".join(i.description for i in parsed.requested_items)
        )
    if parsed.reason_texts:
        lines.append("Encounter reasons: " + "; ".join(parsed.reason_texts))
    if parsed.clinical_notes:
        lines.append("Clinical note:\n" + parsed.clinical_notes)
    return "\n".join(lines)


def extract_clinical_terms(
    parsed: ParsedBundle, llm: "BaseChatModel"
) -> ClinicalExtraction:
    """Run the LLM to extract diagnosis/procedure terms from the bundle text."""
    structured = llm.with_structured_output(ClinicalExtraction)
    messages = [
        ("system", _SYSTEM_PROMPT),
        ("human", _build_context(parsed)),
    ]
    return structured.invoke(messages)


def code_claim(extraction: ClinicalExtraction, parsed: ParsedBundle, crosswalk: Crosswalk) -> CodedClaim:
    """Map extracted + claim-stated terms to codes via the crosswalk."""
    dx_terms = [*extraction.diagnosis_terms, *parsed.stated_diagnoses]
    proc_terms = [
        *extraction.procedure_terms,
        *(i.description for i in parsed.requested_items),
    ]

    diagnoses: dict[str, Any] = {}
    procedures: dict[str, Any] = {}
    unmapped: list[str] = []

    for term in dx_terms:
        if not term.strip():
            continue
        dx = crosswalk.map_diagnosis(term)
        if dx is None:
            unmapped.append(term)
        else:
            diagnoses.setdefault(dx.icd10, dx)

    for term in proc_terms:
        if not term.strip():
            continue
        proc = crosswalk.map_procedure(term)
        if proc is None:
            unmapped.append(term)
        else:
            procedures.setdefault(proc.cpt, proc)

    # De-duplicate unmapped terms while preserving order.
    seen: set[str] = set()
    unmapped_unique = [t for t in unmapped if not (t.lower() in seen or seen.add(t.lower()))]

    return CodedClaim(
        diagnoses=list(diagnoses.values()),
        procedures=list(procedures.values()),
        unmapped=unmapped_unique,
    )


def make_clinical_node(
    llm: "BaseChatModel | None" = None,
    crosswalk: Crosswalk | None = None,
) -> Callable[[TriageState], dict[str, Any]]:
    """Build the Clinical Reviewer node with injected dependencies."""

    def clinical_node(state: TriageState) -> dict[str, Any]:
        start = time.perf_counter()
        active_llm = llm or get_llm()
        active_crosswalk = crosswalk or get_crosswalk()

        assert state.parsed is not None, "clinical_node requires parsed state"
        extraction = extract_clinical_terms(state.parsed, active_llm)
        coded = code_claim(extraction, state.parsed, active_crosswalk)

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Clinical: mapped %d diagnosis code(s), %d procedure code(s), %d unmapped term(s).",
            len(coded.diagnoses),
            len(coded.procedures),
            len(coded.unmapped),
        )
        step = AgentStep(
            agent="clinical",
            status="ok",
            summary=(
                f"Coded {len(coded.diagnoses)} diagnosis/es "
                f"({', '.join(d.icd10 for d in coded.diagnoses) or 'none'}) and "
                f"{len(coded.procedures)} procedure(s) "
                f"({', '.join(p.cpt for p in coded.procedures) or 'none'})."
            ),
            output={
                "diagnoses": [{"icd10": d.icd10, "description": d.description} for d in coded.diagnoses],
                "procedures": [{"cpt": p.cpt, "description": p.description} for p in coded.procedures],
                "unmapped": coded.unmapped,
            },
            latency_ms=latency_ms,
        )
        return {"coded": coded, "trace": [step]}

    return clinical_node

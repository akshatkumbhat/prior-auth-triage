"""Coverage policy: loading and a pure, deterministic rules engine.

The rules engine takes the coded claim plus a map of *criteria assessments*
(which prior-auth criteria the narrative documents) and returns coverage facts.
It is intentionally side-effect free and LLM-free so every rule path is unit
testable. The Coverage Checker node is responsible for producing the criteria
assessments (the only ambiguous, LLM-assisted part).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from pa_triage.config import get_settings
from pa_triage.models.domain import CodedClaim, CoverageResult, FiredRule

# criteria_assessments[cpt][criterion_text] -> was it documented?
CriteriaAssessments = dict[str, dict[str, bool]]


class Exclusion(BaseModel):
    cpt: str
    reason: str


class PriorAuthRule(BaseModel):
    cpt: str
    criteria: list[str] = Field(default_factory=list)


class Policy(BaseModel):
    policy_version: str = "unspecified"
    payer: str = "unspecified"
    requires_diagnosis: bool = True
    covered_cpt: list[str] = Field(default_factory=list)
    exclusions: list[Exclusion] = Field(default_factory=list)
    prior_auth_required: list[PriorAuthRule] = Field(default_factory=list)

    @property
    def covered_set(self) -> set[str]:
        return set(self.covered_cpt)

    @property
    def exclusion_map(self) -> dict[str, str]:
        return {e.cpt: e.reason for e in self.exclusions}

    @property
    def prior_auth_map(self) -> dict[str, list[str]]:
        return {r.cpt: r.criteria for r in self.prior_auth_required}


def load_policy(path: Path | None = None) -> Policy:
    path = path or get_settings().policy_path
    data = json.loads(Path(path).read_text())
    return Policy.model_validate(data)


@lru_cache(maxsize=1)
def get_policy() -> Policy:
    return load_policy()


def evaluate_coverage(
    coded: CodedClaim,
    policy: Policy,
    criteria_assessments: CriteriaAssessments | None = None,
    ambiguity_notes: str | None = None,
) -> CoverageResult:
    """Apply the policy ruleset to a coded claim and return coverage facts.

    Outcome is not decided here — this returns the facts (covered, prior-auth,
    exclusions, missing info, fired rules) that the Decision node maps to an
    approve/deny/pend outcome.
    """
    criteria_assessments = criteria_assessments or {}
    exclusion_map = policy.exclusion_map
    covered_set = policy.covered_set
    prior_auth_map = policy.prior_auth_map

    fired: list[FiredRule] = []
    exclusions_hit: list[str] = []
    missing_info: list[str] = []
    covered = True
    prior_auth_required = False

    if policy.requires_diagnosis and not coded.diagnoses:
        missing_info.append("A supporting diagnosis is required but none could be coded.")
        fired.append(
            FiredRule(
                rule_id="DX-REQUIRED",
                effect="info",
                detail="Policy requires a coded diagnosis to adjudicate the request.",
            )
        )

    if not coded.procedures:
        missing_info.append("No procedure/service could be coded from the request.")
        fired.append(
            FiredRule(
                rule_id="PROC-MISSING",
                effect="info",
                detail="No CPT-coded procedure was found on the claim.",
            )
        )

    for proc in coded.procedures:
        cpt = proc.cpt
        label = f"{proc.description} ({cpt})"

        if cpt in exclusion_map:
            covered = False
            reason = exclusion_map[cpt]
            exclusions_hit.append(reason)
            fired.append(
                FiredRule(rule_id=f"EXCL-{cpt}", effect="exclusion", detail=f"{label}: {reason}")
            )
            continue

        if cpt not in covered_set:
            covered = False
            fired.append(
                FiredRule(
                    rule_id=f"NOTCOV-{cpt}",
                    effect="not_covered",
                    detail=f"{label} is not a covered service under {policy.policy_version}.",
                )
            )
            continue

        fired.append(
            FiredRule(rule_id=f"COV-{cpt}", effect="covered", detail=f"{label} is a covered service.")
        )

        if cpt in prior_auth_map:
            prior_auth_required = True
            criteria = prior_auth_map[cpt]
            assessed = criteria_assessments.get(cpt, {})
            unmet = [c for c in criteria if not assessed.get(c, False)]
            if unmet:
                for c in unmet:
                    missing_info.append(f"Prior-auth criterion not documented for {label}: {c}")
                fired.append(
                    FiredRule(
                        rule_id=f"PA-{cpt}",
                        effect="prior_auth_required",
                        detail=f"Prior authorization required for {label}; criteria not fully met.",
                    )
                )
            else:
                fired.append(
                    FiredRule(
                        rule_id=f"PA-{cpt}",
                        effect="prior_auth_required",
                        detail=f"Prior authorization required for {label}; documented criteria satisfied.",
                    )
                )

    return CoverageResult(
        covered=covered,
        prior_auth_required=prior_auth_required,
        exclusions_hit=exclusions_hit,
        missing_info=missing_info,
        fired_rules=fired,
        ambiguity_notes=ambiguity_notes,
    )

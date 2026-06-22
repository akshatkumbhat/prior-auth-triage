"""LangGraph state schema and per-agent trace records.

The graph state is a Pydantic model so every node boundary is validated.
Nodes return a partial ``dict`` of updates that LangGraph merges into the
state; the ``trace`` list uses an additive reducer so each node can append its
own step without clobbering earlier ones.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pa_triage.models.domain import (
    CodedClaim,
    CoverageResult,
    ParsedBundle,
    TriageDecision,
)

AgentName = Literal["intake", "clinical", "coverage", "decision"]
StepStatus = Literal["ok", "rejected", "error"]


class AgentStep(BaseModel):
    """One entry in the execution trace, surfaced to the dashboard.

    Only de-identified data should ever reach ``summary`` / ``output``.
    """

    model_config = ConfigDict(frozen=True)

    agent: AgentName
    status: StepStatus
    summary: str
    output: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0


class TriageState(BaseModel):
    """State threaded through the LangGraph state graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Input
    raw_bundle: dict[str, Any]

    # Progressively populated by each node
    parsed: ParsedBundle | None = None
    coded: CodedClaim | None = None
    coverage: CoverageResult | None = None
    decision: TriageDecision | None = None

    # Cross-cutting. ``trace`` uses an additive reducer (operator.add) so node
    # updates append rather than replace.
    errors: list[str] = Field(default_factory=list)
    trace: Annotated[list[AgentStep], operator.add] = Field(default_factory=list)
    status: Literal["running", "rejected", "complete"] = "running"

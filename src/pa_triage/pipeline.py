"""High-level entrypoints for running the triage pipeline."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Iterator

from pa_triage.crosswalk import Crosswalk
from pa_triage.graph import build_graph
from pa_triage.models.state import TriageState
from pa_triage.policy import Policy

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models.chat_models import BaseChatModel


def _coerce_state(value: Any) -> TriageState:
    if isinstance(value, TriageState):
        return value
    return TriageState.model_validate(value)


def run_triage(
    raw_bundle: dict[str, Any],
    llm: "BaseChatModel | None" = None,
    crosswalk: Crosswalk | None = None,
    policy: Policy | None = None,
) -> TriageState:
    """Run the full pipeline on a raw FHIR bundle and return the final state."""
    graph = build_graph(llm=llm, crosswalk=crosswalk, policy=policy)
    final = graph.invoke(TriageState(raw_bundle=raw_bundle))
    return _coerce_state(final)


def stream_triage(
    raw_bundle: dict[str, Any],
    llm: "BaseChatModel | None" = None,
    crosswalk: Crosswalk | None = None,
    policy: Policy | None = None,
) -> Iterator[TriageState]:
    """Yield a state snapshot after each node executes (for live dashboards)."""
    graph = build_graph(llm=llm, crosswalk=crosswalk, policy=policy)
    for chunk in graph.stream(TriageState(raw_bundle=raw_bundle), stream_mode="values"):
        yield _coerce_state(chunk)


def measure_latency_ms(
    raw_bundle: dict[str, Any],
    llm: "BaseChatModel | None" = None,
    crosswalk: Crosswalk | None = None,
    policy: Policy | None = None,
) -> tuple[TriageState, float]:
    """Run the pipeline and return ``(final_state, wall_clock_latency_ms)``."""
    start = time.perf_counter()
    state = run_triage(raw_bundle, llm=llm, crosswalk=crosswalk, policy=policy)
    return state, (time.perf_counter() - start) * 1000

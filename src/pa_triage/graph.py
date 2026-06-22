"""LangGraph state graph wiring the four agents.

Topology::

    START -> intake -> (rejected?) --yes--> END
                         |no
                         v
                      clinical -> coverage -> decision -> END

A bad bundle is rejected at intake and routed straight to END, so the LLM
nodes never run on malformed input. Dependencies (LLM, crosswalk, policy) are
injected into :func:`build_graph` so tests can supply fakes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from pa_triage.agents.clinical import make_clinical_node
from pa_triage.agents.coverage import make_coverage_node
from pa_triage.agents.decision import make_decision_node
from pa_triage.agents.intake import intake_node
from pa_triage.crosswalk import Crosswalk
from pa_triage.models.state import TriageState
from pa_triage.policy import Policy

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models.chat_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


def _route_after_intake(state: TriageState) -> str:
    return "reject" if state.status == "rejected" else "continue"


def build_graph(
    llm: "BaseChatModel | None" = None,
    crosswalk: Crosswalk | None = None,
    policy: Policy | None = None,
) -> "CompiledStateGraph":
    """Build and compile the triage state graph."""
    graph = StateGraph(TriageState)

    graph.add_node("intake", intake_node)
    graph.add_node("clinical", make_clinical_node(llm=llm, crosswalk=crosswalk))
    graph.add_node("coverage", make_coverage_node(llm=llm, policy=policy))
    graph.add_node("decision", make_decision_node(llm=llm))

    graph.add_edge(START, "intake")
    graph.add_conditional_edges(
        "intake",
        _route_after_intake,
        {"continue": "clinical", "reject": END},
    )
    graph.add_edge("clinical", "coverage")
    graph.add_edge("coverage", "decision")
    graph.add_edge("decision", END)

    return graph.compile()

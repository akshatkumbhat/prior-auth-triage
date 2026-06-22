"""Streamlit dashboard for the prior-authorization triage PoC.

Run with:  streamlit run app/dashboard.py

Pick or upload a synthetic FHIR bundle, watch the four agents execute in
sequence, and see the final approve / deny / pend decision with its rationale
and the policy rules that fired.
"""

from __future__ import annotations

import importlib.metadata as _md
import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

# Make the src/ layout importable without installing the package — e.g. on
# Hugging Face Spaces, where the app runs straight from the repo checkout.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pa_triage.config import get_settings
from pa_triage.demo_llm import HeuristicLLM
from pa_triage.llm import LLMConfigError, get_llm
from pa_triage.models.state import AgentStep, TriageState
from pa_triage.pipeline import stream_triage

st.set_page_config(page_title="Prior-Auth Triage (PoC)", page_icon="🩺", layout="wide")

settings = get_settings()


def _mask(value: str | None) -> str:
    if not value:
        return "(empty)"
    return f"set · len={len(value)} · prefix={value[:4]}… · suffix=…{value[-4:]}"


def gemini_diagnostics() -> dict[str, object]:
    """Collect (masked) runtime info to diagnose Gemini auth. No API calls, no secrets."""
    info: dict[str, object] = {}
    for pkg in ("langchain-google-genai", "google-genai", "google-auth"):
        try:
            info[f"version:{pkg}"] = _md.version(pkg)
        except Exception:
            info[f"version:{pkg}"] = "(not installed)"

    # Google/Vertex-relevant environment that can flip the client into OAuth mode.
    watch = [
        "GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GCP_PROJECT",
        "GOOGLE_CLOUD_LOCATION", "GOOGLE_APPLICATION_CREDENTIALS",
    ]
    for k in watch:
        v = os.getenv(k)
        info[f"env:{k}"] = "(unset)" if v is None else (_mask(v) if "KEY" in k else v)
    # Catch any other ambient Google/Vertex/GCP vars we didn't anticipate.
    extra = sorted(
        k for k in os.environ
        if any(t in k.upper() for t in ("GOOGLE", "VERTEX", "GCP", "GCLOUD"))
        and k not in watch
    )
    info["env:other_google_vars"] = extra or "(none)"

    info["settings.google_api_key"] = _mask(settings.google_api_key)

    # What auth path would the client take? Build it (no network) and read the flag.
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        probe = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key or "placeholder",
            vertexai=False,
        )
        info["client._use_vertexai"] = getattr(probe, "_use_vertexai", "?")
    except Exception as exc:  # pragma: no cover - diagnostic only
        info["client_build_error"] = f"{type(exc).__name__}: {exc}"
    return info


# Emit diagnostics to stdout once at startup so they appear in the Space logs.
try:
    print("=== PA-TRIAGE GEMINI DIAGNOSTICS ===", flush=True)
    for _k, _v in gemini_diagnostics().items():
        print(f"  {_k} = {_v}", flush=True)
    print("=== END DIAGNOSTICS ===", flush=True)
except Exception as _e:  # pragma: no cover
    print("diagnostics failed:", _e, flush=True)

AGENT_LABELS = {
    "intake": "1 · Intake / Parser",
    "clinical": "2 · Clinical Reviewer",
    "coverage": "3 · Coverage Checker",
    "decision": "4 · Decision / Compliance",
}
STATUS_ICON = {"ok": "complete", "rejected": "error", "error": "error"}
# Illustrative only — NOT a measured benchmark.
ASSUMED_MANUAL_MINUTES = 15.0


# --------------------------------------------------------------------------
# Sidebar: data + backend selection
# --------------------------------------------------------------------------
st.sidebar.title("🩺 Prior-Auth Triage")
st.sidebar.caption("Multi-agent PoC over **synthetic** FHIR R4 bundles.")

st.sidebar.subheader("LLM backend")
demo_default = True
backend = st.sidebar.radio(
    "Reasoning backend",
    options=["Demo (offline heuristic, no setup)", f"Configured provider ({settings.llm_provider})"],
    index=0 if demo_default else 1,
    help="Demo mode uses a deterministic heuristic stand-in so the app runs with "
    "no API key. The configured provider uses the real LLM from your .env.",
)
use_demo = backend.startswith("Demo")

st.sidebar.subheader("Input bundle")
samples = sorted(Path(settings.samples_dir).glob("*.json")) if Path(settings.samples_dir).exists() else []
sample_names = [p.name for p in samples]
choice = st.sidebar.selectbox("Pick a sample bundle", ["— choose —", *sample_names])
uploaded = st.sidebar.file_uploader("…or upload a FHIR bundle (JSON)", type=["json"])

run = st.sidebar.button("▶ Run triage", type="primary", width="stretch")


def _load_selected() -> tuple[dict | None, str | None]:
    if uploaded is not None:
        try:
            return json.loads(uploaded.getvalue().decode("utf-8")), uploaded.name
        except json.JSONDecodeError as exc:
            st.sidebar.error(f"Uploaded file is not valid JSON: {exc}")
            return None, None
    if choice and choice != "— choose —":
        path = Path(settings.samples_dir) / choice
        return json.loads(path.read_text()), choice
    return None, None


def _resolve_llm():
    if use_demo:
        return HeuristicLLM(), None
    try:
        return get_llm(settings), None
    except LLMConfigError as exc:
        return None, str(exc)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Prior-Authorization Triage")
st.info(
    "⚠️ **All data here is fabricated.** This is a portfolio proof-of-concept and "
    "is not for use with real patient data or for clinical decision-making.",
    icon="⚠️",
)


def _render_step(step: AgentStep) -> None:
    label = AGENT_LABELS.get(step.agent, step.agent)
    state = STATUS_ICON.get(step.status, "running")
    with st.status(f"{label} — {step.summary}", state=state, expanded=False):
        st.json(step.output)
        st.caption(f"Latency: {step.latency_ms:.1f} ms")


def _render_decision(state: TriageState) -> None:
    st.subheader("Decision")
    if state.status == "rejected":
        st.error("Bundle rejected at intake. " + (state.errors[0] if state.errors else ""))
        return
    if state.decision is None:
        st.warning("No decision was produced.")
        return

    decision = state.decision
    box = {"approve": st.success, "deny": st.error, "pend": st.warning}[decision.outcome]
    box(f"**{decision.outcome.upper()}** — {decision.rationale}")

    if decision.fired_rules:
        st.markdown("**Rules that fired**")
        st.dataframe(
            [
                {"rule_id": r.rule_id, "effect": r.effect, "detail": r.detail}
                for r in decision.fired_rules
            ],
            hide_index=True,
            width="stretch",
        )
    if decision.missing_info:
        st.markdown("**Missing information (required to proceed)**")
        for item in decision.missing_info:
            st.markdown(f"- {item}")


def _render_metrics(state: TriageState, wall_ms: float) -> None:
    st.subheader("Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Automated latency", f"{wall_ms/1000:.2f} s")
    agent_ms = sum(s.latency_ms for s in state.trace)
    c2.metric("Agent compute", f"{agent_ms:.0f} ms")
    saved = ASSUMED_MANUAL_MINUTES * 60 - wall_ms / 1000
    c3.metric(
        f"vs ~{ASSUMED_MANUAL_MINUTES:.0f} min manual (illustrative)",
        f"{saved/60:.1f} min saved",
        delta="illustrative",
        delta_color="off",
    )
    st.caption(
        "The manual-review comparison is an **illustrative assumption** "
        f"(~{ASSUMED_MANUAL_MINUTES:.0f} min/claim), not a measured benchmark."
    )


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
if run:
    raw_bundle, name = _load_selected()
    if raw_bundle is None:
        st.warning("Pick a sample bundle or upload one, then click **Run triage**.")
        st.stop()

    llm, err = _resolve_llm()
    if err:
        st.error(
            f"{err}\n\nSwitch the sidebar backend to **Demo** to run without an API key."
        )
        st.stop()

    st.caption(f"Bundle: `{name}` · Backend: {'demo heuristic' if use_demo else settings.llm_provider}")
    st.subheader("Agent pipeline")

    rendered = 0
    final_state: TriageState | None = None
    start = time.perf_counter()
    try:
        for snapshot in stream_triage(raw_bundle, llm=llm):
            final_state = snapshot
            while rendered < len(snapshot.trace):
                _render_step(snapshot.trace[rendered])
                rendered += 1
    except Exception as exc:  # surface unexpected failures cleanly
        st.exception(exc)
        st.stop()
    wall_ms = (time.perf_counter() - start) * 1000

    if final_state is not None:
        _render_decision(final_state)
        _render_metrics(final_state, wall_ms)
else:
    st.markdown(
        "Use the sidebar to **pick a sample** (or upload a FHIR bundle) and click "
        "**Run triage**. The four agents — Intake, Clinical Reviewer, Coverage "
        "Checker, and Decision/Compliance — execute as a LangGraph state graph."
    )

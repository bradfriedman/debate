import asyncio
import os
from typing import cast

import streamlit as st
from dotenv import load_dotenv

from src.shared.models import ParticipantID, ModelOption
from src.backend.orchestrator import DebateOrchestrator
from src.backend.agents import AgentAPIError

# Load environment variables from .env file (local development)
load_dotenv()


def get_secret(key: str) -> str | None:
    """Get secret from Streamlit secrets (cloud) or environment variables (local)."""
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    # Fall back to environment variables (local development)
    return os.getenv(key)


# Define available models
MODELS = [
    ModelOption(label="GPT-5.2", model_id="gpt-5.2-2025-12-11", provider="openai"),
    ModelOption(label="Claude 4.5 Sonnet", model_id="claude-sonnet-4-5-20250929", provider="anthropic"),
    ModelOption(label="Gemini 3 Flash", model_id="gemini-2.0-flash", provider="gemini"),
]


def run_async(coro):
    """Helper to run async code in Streamlit's sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


st.set_page_config(page_title="Triad Debate", layout="wide")
st.title("🤖 Multi-Provider Triad Debate")

# Initialize session state
if "stage" not in st.session_state:
    st.session_state.stage = "setup"
if "orch" not in st.session_state:
    st.session_state.orch = None
if "hot_seat_round" not in st.session_state:
    st.session_state.hot_seat_round = 0
if "report" not in st.session_state:
    st.session_state.report = None
if "identities_revealed" not in st.session_state:
    st.session_state.identities_revealed = False

# Sidebar
with st.sidebar:
    st.header("Setup")
    topic = st.text_area(
        "Topic", "Is Artificial General Intelligence (AGI) an existential threat?")

    model_labels = [m.label for m in MODELS]

    m1_lbl = st.selectbox("Model 1", model_labels, index=0)
    m2_lbl = st.selectbox("Model 2", model_labels, index=1)
    m3_lbl = st.selectbox("Model 3", model_labels, index=2)

    start_disabled = st.session_state.stage != "setup"
    if st.button("Start Debate", type="primary", disabled=start_disabled):
        # Map labels back to ModelOption objects
        sel_models = [
            next(m for m in MODELS if m.label == m1_lbl),
            next(m for m in MODELS if m.label == m2_lbl),
            next(m for m in MODELS if m.label == m3_lbl),
        ]
        # Gather API keys from secrets/env for cloud + local compatibility
        api_keys = {
            "OPENAI_API_KEY": get_secret("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": get_secret("ANTHROPIC_API_KEY"),
            "GEMINI_API_KEY": get_secret("GEMINI_API_KEY"),
        }
        st.session_state.orch = DebateOrchestrator(topic, sel_models, api_keys)
        st.session_state.stage = "opening"
        st.session_state.hot_seat_round = 0
        st.session_state.report = None
        st.rerun()

    # Reset button
    if st.session_state.stage != "setup":
        if st.button("Reset Debate"):
            st.session_state.stage = "setup"
            st.session_state.orch = None
            st.session_state.hot_seat_round = 0
            st.session_state.report = None
            st.session_state.identities_revealed = False
            st.rerun()

# Main content area
if st.session_state.stage == "setup":
    st.info("👈 Configure your debate parameters in the sidebar and click **Start Debate**")

    # Show API key status
    st.subheader("API Key Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        if get_secret("OPENAI_API_KEY"):
            st.success("✅ OpenAI API Key")
        else:
            st.warning("❌ OpenAI API Key missing")
    with col2:
        if get_secret("ANTHROPIC_API_KEY"):
            st.success("✅ Anthropic API Key")
        else:
            st.warning("❌ Anthropic API Key missing")
    with col3:
        if get_secret("GEMINI_API_KEY"):
            st.success("✅ Gemini API Key")
        else:
            st.warning("❌ Gemini API Key missing")

elif st.session_state.stage == "opening":
    st.subheader("Phase 1: Opening Statements")
    orch = cast(DebateOrchestrator, st.session_state.orch)
    try:
        with st.spinner("Generating opening statements..."):
            run_async(orch.run_opening())
        st.session_state.stage = "hot_seat"
        st.rerun()
    except AgentAPIError as e:
        st.error(f"**{e.provider} Error:** {e.message}")
        st.info("Please fix the issue and click 'Reset Debate' to try again.")

elif st.session_state.stage == "hot_seat":
    orch = cast(DebateOrchestrator, st.session_state.orch)
    round_num = st.session_state.hot_seat_round

    # Display transcript so far
    st.subheader("Debate Transcript")
    for item in orch.transcript:
        target_str = f" → {item.target.value}" if item.target else ""
        with st.expander(f"**{item.speaker.value}** - {item.turn_type.value}{target_str}", expanded=True):
            st.write(item.content)

    # Run hot seat rounds
    hot_seat_order = [ParticipantID.P1, ParticipantID.P2, ParticipantID.P3]

    if round_num < 3:
        current_pid = hot_seat_order[round_num]
        st.subheader(f"Phase 2: Hot Seat Round {round_num + 1}/3")
        st.info(f"🔥 **{current_pid.value}** is in the Hot Seat!")

        try:
            with st.spinner(f"Running Hot Seat for {current_pid.value}..."):
                run_async(orch.run_hot_seat(current_pid))

            st.session_state.hot_seat_round = round_num + 1
            st.rerun()
        except AgentAPIError as e:
            st.error(f"**{e.provider} Error:** {e.message}")
            st.info("Please fix the issue and click 'Reset Debate' to try again.")
    else:
        # All hot seat rounds done, move to closing
        st.session_state.stage = "closing"
        st.rerun()

elif st.session_state.stage == "closing":
    orch = cast(DebateOrchestrator, st.session_state.orch)

    # Display full transcript
    st.subheader("Debate Transcript")
    for item in orch.transcript:
        target_str = f" → {item.target.value}" if item.target else ""
        with st.expander(f"**{item.speaker.value}** - {item.turn_type.value}{target_str}", expanded=False):
            st.write(item.content)

    st.subheader("Phase 3: Closing Statements")
    try:
        with st.spinner("Generating closing statements..."):
            run_async(orch.run_closing())

        st.session_state.stage = "synthesis"
        st.rerun()
    except AgentAPIError as e:
        st.error(f"**{e.provider} Error:** {e.message}")
        st.info("Please fix the issue and click 'Reset Debate' to try again.")

elif st.session_state.stage == "synthesis":
    orch = cast(DebateOrchestrator, st.session_state.orch)

    # Display full transcript
    st.subheader("Complete Debate Transcript")
    for item in orch.transcript:
        target_str = f" → {item.target.value}" if item.target else ""
        with st.expander(f"**{item.speaker.value}** - {item.turn_type.value}{target_str}", expanded=False):
            st.write(item.content)

    st.subheader("Phase 4: Synthesis & Reveal")
    try:
        with st.spinner("Generating final report..."):
            report = run_async(orch.generate_report())
            st.session_state.report = report

        st.session_state.stage = "complete"
        st.rerun()
    except AgentAPIError as e:
        st.error(f"**{e.provider} Error:** {e.message}")
        st.info("Please fix the issue and click 'Reset Debate' to try again.")

elif st.session_state.stage == "complete":
    orch = cast(DebateOrchestrator, st.session_state.orch)

    # Display full transcript
    st.subheader("Complete Debate Transcript")
    for item in orch.transcript:
        target_str = f" → {item.target.value}" if item.target else ""
        with st.expander(f"**{item.speaker.value}** - {item.turn_type.value}{target_str}", expanded=False):
            st.write(item.content)

    st.divider()

    # Display final report
    st.subheader("📊 Final Report")
    st.write(st.session_state.report)

    # Reveal participant identities with button
    st.divider()
    st.subheader("🎭 The Reveal")

    if not st.session_state.identities_revealed:
        if st.button("Reveal Participant Identities", type="primary"):
            st.session_state.identities_revealed = True
            st.rerun()
    else:
        for pid, model_opt in orch.assignments.items():
            st.success(f"**{pid.value}** was **{model_opt.label}** ({model_opt.model_id})")


def main():
    """Entry point for running via `uv run debate` or `streamlit run`."""
    import sys
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__, "--server.headless", "true"]
    sys.exit(stcli.main())

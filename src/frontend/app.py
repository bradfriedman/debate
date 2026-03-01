import asyncio
import os
import re
from typing import cast

import streamlit as st
from dotenv import load_dotenv

from src.shared.models import ParticipantID, ModelOption, TurnType, DialogueItem
from src.backend.orchestrator import DebateOrchestrator
from src.backend.agents import AgentAPIError

# Load environment variables from .env file (local development)
load_dotenv()


def escape_currency_dollars(text: str) -> str:
    """Escape dollar signs used as currency to prevent Streamlit from rendering them as LaTeX.

    Targets $ followed by a digit or comma (e.g. $290K, $1,000) while leaving
    LaTeX-style patterns like $\\alpha$ untouched.
    """
    return re.sub(r'\$(?=[\d,])', r'\\$', text)


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
    ModelOption(label="Claude 4.6 Sonnet", model_id="claude-sonnet-4-6", provider="anthropic"),
    ModelOption(label="Gemini 3 Pro", model_id="gemini-3-pro-preview", provider="gemini"),
]


def render_transcript(items: list[DialogueItem], closings_expanded: bool = False) -> None:
    """Render transcript with openings, grouped hot-seat rounds, and closing statements."""
    i = 0
    while i < len(items):
        item = items[i]
        if item.turn_type == TurnType.OPENING:
            with st.expander(f"**{item.speaker.value}** - Opening Statement", expanded=False):
                st.write(escape_currency_dollars(item.content))
            i += 1
        elif item.turn_type == TurnType.QUESTION:
            hot_seat_pid = item.target
            if hot_seat_pid is None:
                i += 1
                continue
            block: list[DialogueItem] = []
            while i < len(items):
                curr = items[i]
                if curr.turn_type == TurnType.QUESTION and curr.target == hot_seat_pid:
                    block.append(curr)
                    i += 1
                elif curr.turn_type == TurnType.ANSWER and curr.speaker == hot_seat_pid:
                    block.append(curr)
                    i += 1
                else:
                    break
            with st.expander(f"Hot Seat: {hot_seat_pid.value}", expanded=False):
                lines = []
                for bi in block:
                    if bi.turn_type == TurnType.QUESTION:
                        label = f"**{bi.speaker.value} \u2192 {hot_seat_pid.value}:**"
                    else:
                        label = f"**{hot_seat_pid.value} Response:**"
                    lines.append(f"{label} {escape_currency_dollars(bi.content)}")
                st.markdown("\n\n".join(lines))
        elif item.turn_type == TurnType.CLOSING:
            with st.expander(
                f"**{item.speaker.value}** - Closing Statement", expanded=closings_expanded
            ):
                st.write(escape_currency_dollars(item.content))
            i += 1
        else:
            i += 1


def run_async(coro):
    """Helper to run async code in Streamlit's sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def abort_debate(error_msg: str) -> None:
    """Reset the debate back to setup, preserving topic and model selections, and record the error."""
    st.session_state.stage = "setup"
    st.session_state.orch = None
    st.session_state.hot_seat_round = 0
    st.session_state.report = None
    st.session_state.identities_revealed = False
    st.session_state.debate_error = error_msg
    st.rerun()


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
if "topic" not in st.session_state:
    st.session_state.topic = "Is Artificial General Intelligence (AGI) an existential threat?"
if "debate_error" not in st.session_state:
    st.session_state.debate_error = None

# Sidebar
with st.sidebar:
    st.header("Setup")

    model_labels = [m.label for m in MODELS]

    m1_lbl = st.selectbox("Model 1", model_labels, index=0, key="model_select_1")
    m2_lbl = st.selectbox("Model 2", model_labels, index=1, key="model_select_2")
    m3_lbl = st.selectbox("Model 3", model_labels, index=2, key="model_select_3")

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
        st.session_state.debate_error = None
        st.session_state.orch = DebateOrchestrator(st.session_state.topic, sel_models, api_keys)
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
    if st.session_state.debate_error:
        st.error(st.session_state.debate_error)
        st.info(
            "Your topic and model selections have been preserved. "
            "Fix the issue above and click **Start Debate** to try again."
        )
    st.markdown(
        "<style>textarea { overflow-x: hidden !important; overflow-y: auto !important; }</style>",
        unsafe_allow_html=True,
    )
    st.text_area(
        "Debate Topic",
        key="topic",
        height=420,
    )
    st.info("Configure the models in the sidebar and click **Start Debate**")

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
        abort_debate(f"**{e.provider} Error** during opening statements: {e.message}")
    except Exception as e:
        abort_debate(f"Unexpected error during opening statements: {e}")

elif st.session_state.stage == "hot_seat":
    orch = cast(DebateOrchestrator, st.session_state.orch)
    round_num = st.session_state.hot_seat_round

    # Display transcript so far
    st.subheader("Debate Transcript")
    render_transcript(orch.transcript)

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
            abort_debate(f"**{e.provider} Error** during hot seat ({current_pid.value}): {e.message}")
        except Exception as e:
            abort_debate(f"Unexpected error during hot seat ({current_pid.value}): {e}")
    else:
        # All hot seat rounds done, move to closing
        st.session_state.stage = "closing"
        st.rerun()

elif st.session_state.stage == "closing":
    orch = cast(DebateOrchestrator, st.session_state.orch)

    # Display full transcript
    st.subheader("Debate Transcript")
    render_transcript(orch.transcript)

    st.subheader("Phase 3: Closing Statements")
    try:
        with st.spinner("Generating closing statements..."):
            run_async(orch.run_closing())

        st.session_state.stage = "synthesis"
        st.rerun()
    except AgentAPIError as e:
        abort_debate(f"**{e.provider} Error** during closing statements: {e.message}")
    except Exception as e:
        abort_debate(f"Unexpected error during closing statements: {e}")

elif st.session_state.stage == "synthesis":
    orch = cast(DebateOrchestrator, st.session_state.orch)

    # Display full transcript
    st.subheader("Complete Debate Transcript")
    render_transcript(orch.transcript)

    st.subheader("Phase 4: Synthesis & Reveal")
    try:
        with st.spinner("Generating final report..."):
            report = run_async(orch.generate_report())
            st.session_state.report = report

        st.session_state.stage = "complete"
        st.rerun()
    except AgentAPIError as e:
        abort_debate(f"**{e.provider} Error** during synthesis: {e.message}")
    except Exception as e:
        abort_debate(f"Unexpected error during synthesis: {e}")

elif st.session_state.stage == "complete":
    orch = cast(DebateOrchestrator, st.session_state.orch)

    # Display full transcript (closings expanded in final view)
    st.subheader("Complete Debate Transcript")
    st.markdown(f"**Topic:** {orch.topic}")
    render_transcript(orch.transcript, closings_expanded=True)

    st.divider()

    # Display final report
    st.subheader("📊 Final Report")
    st.write(escape_currency_dollars(st.session_state.report or ""))

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

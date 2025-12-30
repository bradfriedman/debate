# Triad Debate

Multi-agent debate platform where three LLMs argue a topic in a structured, blind format.

## Quick Start

```bash
uv sync
PYTHONPATH=. uv run streamlit run src/frontend/app.py
```

Requires `.env` with API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`

## Architecture

```
src/
├── frontend/app.py      # Streamlit UI, session state, debate flow control
├── backend/
│   ├── orchestrator.py  # DebateOrchestrator: coordinates phases, manages transcript
│   ├── agents.py        # BaseAgent + provider implementations (OpenAI, Anthropic, Gemini)
│   ├── tools.py         # web_search tool (DuckDuckGo)
│   └── models.py        # Re-exports from shared
└── shared/models.py     # ParticipantID, TurnType, DialogueItem, ModelOption
```

## Key Patterns

**Factory Pattern**: `create_agent(pid, model_option, topic)` in agents.py instantiates correct agent class based on provider.

**Strategy Pattern**: Each agent (OpenAIAgent, AnthropicAgent, GeminiAgent) implements `generate()` with provider-specific API handling and tool loops.

## Debate Flow

1. **Setup**: User selects 3 models. System shuffles into P1/P2/P3 (identities hidden until end).

2. **Opening Statements**: All three generate concurrently via `asyncio.gather`.

3. **Hot Seat Rounds** (3 rounds, one per participant):
   - Hot seat participant defends their position
   - Other two participants ask challenging questions (random order)
   - Questions include explicit reminder of target's opening statement
   - Questions and answers appended to shared transcript

4. **Closing Statements**: All three summarize, noting any opinion changes.

5. **Synthesis & Reveal**: P1's agent generates analysis report; model identities revealed.

## Data Flow

- `DebateOrchestrator.transcript: List[DialogueItem]` - shared conversation history
- Each `DialogueItem` has: `speaker`, `turn_type`, `target` (optional), `content`, `timestamp`
- `_format_history()` converts transcript to text for LLM prompts

## Agent Prompts (in agents.py BaseAgent)

| Method | Purpose | Key Context |
|--------|---------|-------------|
| `generate_opening()` | Initial position | Topic only |
| `generate_question()` | Challenge hot seat participant | Full transcript + target's opening statement explicitly quoted |
| `generate_answer()` | Defend position | Full transcript + the question asked |
| `generate_closing()` | Final summary | Full transcript |

## Tool Use

All agents have `web_search(query)` for fact verification. Provider-specific handling:
- **OpenAI**: `tool_calls` in response, send `tool` role message back
- **Anthropic**: `stop_reason == "tool_use"`, send `tool_result` in user message
- **Gemini**: `function_call` in parts, send `Part.from_function_response`

## Session State (Streamlit)

| Key | Values | Purpose |
|-----|--------|---------|
| `stage` | setup, opening, hot_seat, closing, synthesis, complete | Current phase |
| `orch` | DebateOrchestrator | Holds agents and transcript |
| `hot_seat_round` | 0, 1, 2 | Which participant's turn (P1, P2, P3) |
| `report` | string | Final synthesis text |

## Error Handling

`AgentAPIError` wraps provider-specific errors with user-friendly messages for auth failures, rate limits, and billing issues. Each provider has a `_handle_*_error()` helper.

## Code Style

PEP-8, 4-space indents, 120-char columns. Type hints used throughout.

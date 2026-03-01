import asyncio
import random
from collections.abc import Callable
from typing import List
from src.shared.models import ParticipantID, DialogueItem, TurnType, ModelOption
from src.backend.agents import create_agent, AgentAPIError

MINIMUM_RESPONSE_CHARS = 15
MAX_RETRIES = 3          # Total attempts per call (initial + 2 retries)
RETRY_DELAY_SECONDS = 2  # Fixed delay between attempts


class DebateOrchestrator:
    def __init__(
        self,
        topic: str,
        selected_models: List[ModelOption],
        api_keys: dict[str, str | None] | None = None
    ):
        self.topic = topic
        self.transcript: List[DialogueItem] = []
        self.api_keys = api_keys or {}

        # Randomize Models
        shuffled = list(selected_models)
        random.shuffle(shuffled)

        self.assignments = {
            ParticipantID.P1: shuffled[0],
            ParticipantID.P2: shuffled[1],
            ParticipantID.P3: shuffled[2]
        }

        # Factory Instantiation - pass appropriate API key for each provider
        self.agents = {
            pid: create_agent(pid, opt, topic, self._get_api_key(opt.provider))
            for pid, opt in self.assignments.items()
        }

    def _get_api_key(self, provider: str) -> str | None:
        """Get API key for a given provider."""
        key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        return self.api_keys.get(key_map.get(provider, ""))

    def _validate_response(self, text: str, speaker: ParticipantID, context: str) -> None:
        """Raise AgentAPIError if a response is blank or nearly blank."""
        if len(text.strip()) < MINIMUM_RESPONSE_CHARS:
            model_opt = self.assignments.get(speaker)
            provider = model_opt.provider.capitalize() if model_opt else "Unknown"
            model_name = model_opt.label if model_opt else speaker.value
            raise AgentAPIError(
                provider,
                f"{model_name} ({speaker.value}) returned a blank or near-blank response "
                f"during {context}. This may indicate an API issue or content filtering."
            )

    async def _call_with_retry(
        self,
        loop: asyncio.AbstractEventLoop,
        fn: Callable[[], str],
        speaker: ParticipantID,
        context: str,
    ) -> str:
        """Run fn() in a thread executor, retrying on blank responses before failing."""
        last_text = ""
        for attempt in range(MAX_RETRIES):
            last_text = await loop.run_in_executor(None, fn)
            if len(last_text.strip()) >= MINIMUM_RESPONSE_CHARS:
                return last_text
            if attempt < MAX_RETRIES - 1:
                print(
                    f"[Retry {attempt + 1}/{MAX_RETRIES - 1}] Blank response from "
                    f"{speaker.value} during {context}. Retrying in {RETRY_DELAY_SECONDS}s..."
                )
                await asyncio.sleep(RETRY_DELAY_SECONDS)
        self._validate_response(last_text, speaker, context)
        return last_text  # unreachable; satisfies type checker

    # ... (Run Opening/HotSeat/Closing methods remain identical to previous design) ...
    # They call agent.generate_opening() etc., which are defined in BaseAgent

    async def run_opening(self):
        # We need to wrap synchronous API calls in asyncio.to_thread to not block the loop
        # Since standard OpenAI/Anthropic clients are sync (unless using AsyncClient),
        # this is important for the UI to remain responsive.

        loop = asyncio.get_running_loop()
        tasks = [
            self._call_with_retry(loop, agent.generate_opening, pid, "opening statement")
            for pid, agent in self.agents.items()
        ]
        results = await asyncio.gather(*tasks)

        for pid, text in zip(self.agents.keys(), results):
            self.transcript.append(DialogueItem(
                speaker=pid, turn_type=TurnType.OPENING, content=text))

    async def run_hot_seat(self, hot_seat_pid: ParticipantID):
        loop = asyncio.get_running_loop()
        hot_seat_agent = self.agents[hot_seat_pid]
        others = [p for p in self.agents if p != hot_seat_pid]
        random.shuffle(others)

        for asker_pid in others:
            asker_agent = self.agents[asker_pid]

            # Asker
            q_text = await self._call_with_retry(
                loop,
                lambda a=asker_agent: a.generate_question(self.transcript, hot_seat_pid),
                asker_pid,
                f"question to {hot_seat_pid.value}",
            )
            self.transcript.append(DialogueItem(
                speaker=asker_pid, turn_type=TurnType.QUESTION, target=hot_seat_pid, content=q_text))

            # Answerer
            a_text = await self._call_with_retry(
                loop,
                lambda: hot_seat_agent.generate_answer(self.transcript, q_text, asker_pid),
                hot_seat_pid,
                f"answer to {asker_pid.value}",
            )
            self.transcript.append(DialogueItem(
                speaker=hot_seat_pid, turn_type=TurnType.ANSWER, target=asker_pid, content=a_text))

    async def run_closing(self):
        loop = asyncio.get_running_loop()
        tasks = [
            self._call_with_retry(
                loop,
                lambda agent=agent: agent.generate_closing(self.transcript),
                pid,
                "closing statement",
            )
            for pid, agent in self.agents.items()
        ]
        results = await asyncio.gather(*tasks)

        for pid, text in zip(self.agents.keys(), results):
            self.transcript.append(DialogueItem(
                speaker=pid, turn_type=TurnType.CLOSING, content=text))

    async def run_followup(self, user_message: str) -> None:
        """Append user follow-up to transcript and generate updated opening statements."""
        self.transcript.append(DialogueItem(
            speaker=ParticipantID.MODERATOR,
            turn_type=TurnType.FOLLOWUP_PROMPT,
            content=user_message
        ))

        loop = asyncio.get_running_loop()
        tasks = [
            self._call_with_retry(
                loop,
                lambda agent=agent: agent.generate_followup_opening(self.transcript),
                pid,
                "updated opening statement",
            )
            for pid, agent in self.agents.items()
        ]
        results = await asyncio.gather(*tasks)

        for pid, text in zip(self.agents.keys(), results):
            self.transcript.append(DialogueItem(
                speaker=pid, turn_type=TurnType.FOLLOWUP_OPENING, content=text))

    async def generate_report(self):
        # Use P1's agent for synthesis (reusing the instance)
        # Or instantiate a dedicated synthesis agent
        synth_agent = self.agents[ParticipantID.P1]

        lines = []
        for i in self.transcript:
            if i.turn_type == TurnType.FOLLOWUP_PROMPT:
                lines.append(f"\n[AUDIENCE FOLLOW-UP MESSAGE]: {i.content}\n")
            else:
                lines.append(f"{i.speaker.value}: {i.content}")
        full_text = "\n".join(lines)

        prompt = (
            f"Analyze this debate on '{self.topic}':\n\n{full_text}\n\n"
            "Create a report with these sections:\n"
            "1. Executive Summary\n"
            "2. Key Areas of Consensus and Divergence\n"
            "3. Notable Perspective Shifts\n\n"
            "Do NOT reveal which models were used - that will be shown separately."
        )

        loop = asyncio.get_running_loop()
        report_text = await self._call_with_retry(
            loop,
            lambda: synth_agent.generate(prompt, max_tokens=4096),
            ParticipantID.P1,
            "synthesis report",
        )
        return report_text

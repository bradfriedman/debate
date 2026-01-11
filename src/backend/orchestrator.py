import asyncio
import random
from typing import List
from src.shared.models import ParticipantID, DialogueItem, TurnType, ModelOption
from src.backend.agents import create_agent
# For synthesis, we can just use the OpenAIAgent or whichever is P1 to save logic


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

    # ... (Run Opening/HotSeat/Closing methods remain identical to previous design) ...
    # They call agent.generate_opening() etc., which are defined in BaseAgent

    async def run_opening(self):
        # We need to wrap synchronous API calls in asyncio.to_thread to not block the loop
        # Since standard OpenAI/Anthropic clients are sync (unless using AsyncClient),
        # this is important for the UI to remain responsive.

        loop = asyncio.get_running_loop()
        tasks = []
        for pid, agent in self.agents.items():
            tasks.append(loop.run_in_executor(None, agent.generate_opening))

        results = await asyncio.gather(*tasks)

        for pid, text in zip(self.agents.keys(), results):
            self.transcript.append(DialogueItem(
                speaker=pid, turn_type=TurnType.OPENING, content=text))

    async def run_hot_seat(self, hot_seat_pid: ParticipantID):
        loop = asyncio.get_running_loop()
        hot_seat_agent = self.agents[hot_seat_pid]
        others = [p for p in ParticipantID if p != hot_seat_pid]
        random.shuffle(others)

        for asker_pid in others:
            asker_agent = self.agents[asker_pid]

            # Asker
            q_text = await loop.run_in_executor(None, asker_agent.generate_question, self.transcript, hot_seat_pid)
            self.transcript.append(DialogueItem(
                speaker=asker_pid, turn_type=TurnType.QUESTION, target=hot_seat_pid, content=q_text))

            # Answerer
            a_text = await loop.run_in_executor(None, hot_seat_agent.generate_answer, self.transcript, q_text, asker_pid)
            self.transcript.append(DialogueItem(
                speaker=hot_seat_pid, turn_type=TurnType.ANSWER, target=asker_pid, content=a_text))

    async def run_closing(self):
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(
            None, agent.generate_closing, self.transcript) for agent in self.agents.values()]
        results = await asyncio.gather(*tasks)
        for pid, text in zip(self.agents.keys(), results):
            self.transcript.append(DialogueItem(
                speaker=pid, turn_type=TurnType.CLOSING, content=text))

    async def generate_report(self):
        # Use P1's agent for synthesis (reusing the instance)
        # Or instantiate a dedicated synthesis agent
        synth_agent = self.agents[ParticipantID.P1]

        full_text = "\n".join(
            [f"{i.speaker.value}: {i.content}" for i in self.transcript])

        prompt = (
            f"Analyze this debate on '{self.topic}':\n\n{full_text}\n\n"
            "Create a report with these sections:\n"
            "1. Executive Summary\n"
            "2. Key Areas of Consensus and Divergence\n"
            "3. Notable Perspective Shifts\n\n"
            "Do NOT reveal which models were used - that will be shown separately."
        )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, synth_agent.generate, prompt)

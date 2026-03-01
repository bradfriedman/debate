from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import time


class ParticipantID(str, Enum):
    P1 = "Participant 1"
    P2 = "Participant 2"
    P3 = "Participant 3"
    MODERATOR = "Moderator"


class TurnType(str, Enum):
    OPENING = "Opening Statement"
    QUESTION = "Question"
    ANSWER = "Answer"
    CLOSING = "Closing Statement"
    FOLLOWUP_PROMPT = "Follow-up"
    FOLLOWUP_OPENING = "Updated Opening"


class DialogueItem(BaseModel):
    speaker: ParticipantID
    turn_type: TurnType
    target: Optional[ParticipantID] = None
    content: str
    timestamp: float = Field(default_factory=time.time)


class ModelOption(BaseModel):
    label: str  # Display name in UI
    model_id: str  # API model string (e.g., "gpt-4o")
    provider: str  # "openai", "anthropic", "gemini"

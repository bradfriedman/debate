import abc
import json
import os

# SDKs
from openai import OpenAI, APIError as OpenAIAPIError, AuthenticationError as OpenAIAuthError
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
import anthropic
from anthropic.types import TextBlock, ToolParam, MessageParam, ToolUseBlock
from google import genai
from google.genai import types
from google.genai.errors import APIError as GeminiAPIError

from src.shared.models import ParticipantID, DialogueItem, ModelOption, TurnType
from src.backend.tools import web_search


class AgentAPIError(Exception):
    """Raised when an API call fails with a user-friendly message."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        self.message = message
        super().__init__(f"[{provider}] {message}")


def _handle_openai_error(e: Exception) -> None:
    """Convert OpenAI exceptions to AgentAPIError."""
    if isinstance(e, OpenAIAuthError):
        raise AgentAPIError("OpenAI", "Authentication failed. Check your OPENAI_API_KEY.") from e
    if isinstance(e, OpenAIAPIError):
        msg = str(e)
        if "insufficient_quota" in msg or "billing" in msg.lower():
            raise AgentAPIError("OpenAI", "Insufficient credits. Please check your OpenAI billing.") from e
        if "rate_limit" in msg.lower():
            raise AgentAPIError("OpenAI", "Rate limit exceeded. Please wait and try again.") from e
        raise AgentAPIError("OpenAI", f"API error: {msg}") from e
    raise AgentAPIError("OpenAI", f"Unexpected error: {e}") from e


def _handle_anthropic_error(e: Exception) -> None:
    """Convert Anthropic exceptions to AgentAPIError."""
    if isinstance(e, anthropic.AuthenticationError):
        raise AgentAPIError("Anthropic", "Authentication failed. Check your ANTHROPIC_API_KEY.") from e
    if isinstance(e, anthropic.BadRequestError):
        msg = str(e)
        if "credit balance" in msg.lower() or "billing" in msg.lower():
            raise AgentAPIError("Anthropic", "Insufficient credits. Please top up your Anthropic account.") from e
        raise AgentAPIError("Anthropic", f"Bad request: {msg}") from e
    if isinstance(e, anthropic.RateLimitError):
        raise AgentAPIError("Anthropic", "Rate limit exceeded. Please wait and try again.") from e
    if isinstance(e, anthropic.APIError):
        raise AgentAPIError("Anthropic", f"API error: {e}") from e
    raise AgentAPIError("Anthropic", f"Unexpected error: {e}") from e


def _handle_gemini_error(e: Exception) -> None:
    """Convert Gemini exceptions to AgentAPIError."""
    if isinstance(e, GeminiAPIError):
        msg = str(e)
        if "api_key" in msg.lower() or "authentication" in msg.lower():
            raise AgentAPIError("Gemini", "Authentication failed. Check your GEMINI_API_KEY.") from e
        if "quota" in msg.lower() or "billing" in msg.lower():
            raise AgentAPIError("Gemini", "Quota exceeded. Please check your Google Cloud billing.") from e
        if "rate" in msg.lower():
            raise AgentAPIError("Gemini", "Rate limit exceeded. Please wait and try again.") from e
        raise AgentAPIError("Gemini", f"API error: {msg}") from e
    raise AgentAPIError("Gemini", f"Unexpected error: {e}") from e

# --- Base Abstract Agent ---


class BaseAgent(abc.ABC):
    def __init__(self, pid: ParticipantID, model_id: str, topic: str):
        self.pid = pid
        self.model_id = model_id
        self.topic = topic
        self.system_prompt = (
            f"You are {self.pid.value} in a structured debate. Topic: '{self.topic}'. "
            "You have access to a 'web_search' tool. Use it if you need to verify facts. "
            "Be concise, rigorous, and direct. Do not reveal your AI identity."
        )

    def _format_history(self, transcript: list[DialogueItem]) -> str:
        text = "--- TRANSCRIPT START ---\n"
        for item in transcript:
            target = f"(to {item.target.value})" if item.target else ""
            text += f"[{item.speaker.value} - {item.turn_type.value}{target}]: {item.content}\n"
        text += "--- TRANSCRIPT END ---\n"
        return text

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Handles the LLM call, including tool execution loop."""
        pass

    # Helper methods for specific turns
    def generate_opening(self) -> str:
        return self.generate(f"Provide your Opening Statement on '{self.topic}'. Limit 200 words.")

    def generate_question(self, transcript: list[DialogueItem], target: ParticipantID) -> str:
        hist = self._format_history(transcript)

        # Extract target's opening statement to make their position explicit
        target_opening = next(
            (item.content for item in transcript
             if item.speaker == target and item.turn_type == TurnType.OPENING),
            None
        )

        target_context = ""
        if target_opening:
            target_context = (
                f"\n\nREMINDER - {target.value}'s OPENING STATEMENT was:\n"
                f'"""\n{target_opening}\n"""\n'
            )

        return self.generate(
            f"{hist}{target_context}\n"
            f"INSTRUCTION: {target.value} is in the Hot Seat. "
            f"Based on THEIR opening statement above (not another participant's), "
            f"ask them a challenging question about THEIR position. Limit 100 words."
        )

    def generate_answer(self, transcript: list[DialogueItem], question: str, asker: ParticipantID) -> str:
        hist = self._format_history(transcript)
        return self.generate(f"{hist}\nINSTRUCTION: You are in the Hot Seat. {asker.value} asked: '{question}'. Respond defensively. Limit 150 words.")

    def generate_closing(self, transcript: list[DialogueItem]) -> str:
        hist = self._format_history(transcript)
        return self.generate(f"{hist}\nINSTRUCTION: Provide Closing Statement. Summarize your view and any changes of opinion. Limit 200 words.")


# --- OpenAI Agent ---

class OpenAIAgent(BaseAgent):
    def __init__(self, pid: ParticipantID, model_id: str, topic: str, api_key: str | None = None) -> None:
        super().__init__(pid, model_id, topic)
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        # Define Tool Schema
        self.tools: list[ChatCompletionToolParam] = [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }]

    def generate(self, prompt: str) -> str:
        try:
            messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

            # First Call
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            msg = response.choices[0].message

            # Check for Tool Call
            if msg.tool_calls:
                # Append initial message to history
                messages.append({"role": "assistant", "tool_calls": msg.tool_calls})  # type: ignore[list-item]

                for tool_call in msg.tool_calls:
                    func = getattr(tool_call, "function", None)
                    if func and getattr(func, "name", None) == "web_search":
                        args = json.loads(func.arguments)
                        search_res = web_search(args["query"])

                        # Append Tool Result
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": search_res
                        })

                # Second Call (with tool results)
                final_res = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages
                )
                return final_res.choices[0].message.content or ""

            return msg.content or ""
        except AgentAPIError:
            raise
        except Exception as e:
            _handle_openai_error(e)
            return ""  # Unreachable, but satisfies type checker


# --- Anthropic Agent ---

class AnthropicAgent(BaseAgent):
    def __init__(self, pid: ParticipantID, model_id: str, topic: str, api_key: str | None = None) -> None:
        super().__init__(pid, model_id, topic)
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

        self.tools: list[ToolParam] = [{
            "name": "web_search",
            "description": "Search the web for information",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }]

    def generate(self, prompt: str) -> str:
        try:
            messages: list[MessageParam] = [{"role": "user", "content": prompt}]

            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=1024,
                system=self.system_prompt,
                messages=messages,
                tools=self.tools
            )

            # Check Stop Reason
            if response.stop_reason == "tool_use":
                # Append Assistant's tool use intent
                messages.append({"role": "assistant", "content": response.content})

                for block in response.content:
                    if isinstance(block, ToolUseBlock) and block.name == "web_search":
                        block_input = block.input
                        query = str(block_input.get("query", "")) if isinstance(block_input, dict) else ""
                        result = web_search(query)

                        # Append Tool Result
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result
                            }]
                        })

                # Final Call
                final_res = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=1024,
                    system=self.system_prompt,
                    messages=messages,
                    tools=self.tools
                )
                text_blocks = [block for block in final_res.content if isinstance(block, TextBlock)]
                return text_blocks[0].text if text_blocks else ""

            text_blocks = [block for block in response.content if isinstance(block, TextBlock)]
            return text_blocks[0].text if text_blocks else ""
        except AgentAPIError:
            raise
        except Exception as e:
            _handle_anthropic_error(e)
            return ""  # Unreachable, but satisfies type checker


# --- Gemini Agent ---

class GeminiAgent(BaseAgent):
    def __init__(self, pid: ParticipantID, model_id: str, topic: str, api_key: str | None = None) -> None:
        super().__init__(pid, model_id, topic)
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))

        # Define the web_search tool schema for Gemini
        self.web_search_declaration = types.FunctionDeclaration(
            name="web_search",
            description="Search the web for current information",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="The search query")
                },
                required=["query"]
            )
        )
        self.tools = types.Tool(function_declarations=[self.web_search_declaration])

    def generate(self, prompt: str) -> str:
        try:
            contents: list[types.Content] = [
                types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)]
                )
            ]

            # First call
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,  # type: ignore[arg-type]
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    tools=[self.tools]
                )
            )

            # Check for function call
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        func_call = part.function_call
                        if func_call.name == "web_search" and func_call.args:
                            query = func_call.args.get("query", "")
                            search_result = web_search(query)

                            # Add the assistant's function call to history
                            assistant_content = response.candidates[0].content
                            if assistant_content:
                                contents.append(assistant_content)

                            # Add function response
                            contents.append(
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_function_response(
                                        name="web_search",
                                        response={"result": search_result}
                                    )]
                                )
                            )

                            # Second call with function result
                            final_response = self.client.models.generate_content(
                                model=self.model_id,
                                contents=contents,  # type: ignore[arg-type]
                                config=types.GenerateContentConfig(
                                    system_instruction=self.system_prompt,
                                    tools=[self.tools]
                                )
                            )
                            return final_response.text or ""

            return response.text or ""
        except AgentAPIError:
            raise
        except Exception as e:
            _handle_gemini_error(e)
            return ""  # Unreachable, but satisfies type checker


# --- Factory ---


def create_agent(
    pid: ParticipantID,
    model_option: ModelOption,
    topic: str,
    api_key: str | None = None
) -> BaseAgent:
    if model_option.provider == "openai":
        return OpenAIAgent(pid, model_option.model_id, topic, api_key)
    elif model_option.provider == "anthropic":
        return AnthropicAgent(pid, model_option.model_id, topic, api_key)
    elif model_option.provider == "gemini":
        return GeminiAgent(pid, model_option.model_id, topic, api_key)
    else:
        raise ValueError(f"Unknown provider: {model_option.provider}")

"""Clients for TRACE models served behind an OpenAI-compatible endpoint (vLLM, SGLang, ...)."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from .formatting import build_messages, parse_output


@dataclass
class TraceResponse:
    answer: str                 # user-facing reply (the ANSWER block)
    state: Optional[dict]       # parsed STATE block, None if absent/malformed
    state_text: str             # raw STATE contents
    raw: str                    # full model completion

    @property
    def action(self) -> Optional[str]:
        return (self.state or {}).get("action")

    @property
    def jailbreak_score(self) -> Optional[int]:
        return (self.state or {}).get("jailbreak_score")


class TraceModel:
    """Stateless client: takes a full chat history, returns the parsed TRACE response.

    The defaults reproduce the paper's multi-turn attack evaluation: temperature 0 and a generous
    completion budget, since the STATE block precedes the answer.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:30003/v1",
        api_key: str = "EMPTY",
        variant: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        client=None,
    ):
        if client is None:
            from openai import OpenAI  # imported lazily so the package works without it
            client = OpenAI(base_url=base_url, api_key=api_key)
        self.client = client
        self.model = model
        self.variant = variant
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: List[Dict[str, str]], **overrides) -> TraceResponse:
        """`messages` is a standard chat list ending in a user message; returns the parsed reply."""
        request = build_messages(messages, self.variant)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=request,
            temperature=overrides.get("temperature", self.temperature),
            max_tokens=overrides.get("max_tokens", self.max_tokens),
        )
        raw = completion.choices[0].message.content or ""
        state, state_text, answer = parse_output(raw)
        return TraceResponse(answer=answer, state=state, state_text=state_text, raw=raw)


class Conversation:
    """Stateful multi-turn helper. Keeps the history the way the evaluations did: assistant turns
    hold only the ANSWER text, so the model never sees its own past STATE blocks."""

    def __init__(self, model: TraceModel):
        self.model = model
        self.history: List[Dict[str, str]] = []
        self.responses: List[TraceResponse] = []

    def send(self, user_message: str, **overrides) -> TraceResponse:
        resp = self.model.chat(self.history + [{"role": "user", "content": user_message}], **overrides)
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": resp.answer})
        self.responses.append(resp)
        return resp

    def reset(self) -> None:
        self.history.clear()
        self.responses.clear()

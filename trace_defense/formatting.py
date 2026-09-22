"""Input formatting and output parsing for TRACE models.

TRACE models do not consume a chat history as a list of role-tagged messages. They are trained on a
single user message that contains the whole trajectory, turn by turn:

    [Turn 1]
    USER: ...
    ASSISTANT: ...
    [Turn 2]
    USER: ...

The final turn holds only the user message. The model answers with a <STATE> block (its safety
assessment of the trajectory) followed by an <ANSWER> block (the reply). Only the ANSWER is meant
for the user, and only the ANSWER goes back into the history for the next turn -- the STATE block
never re-enters the context.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

PROMPT_VARIANTS = {
    # The prompt used for the multi-turn attack evaluations in the paper.
    "default": "state_answer_action_prompt.txt",
    # Adds a harmful-vs-harmful-looking distinction and forbids leading disclaimers;
    # used for the PHTest over-refusal measurement.
    "over_refusal": "state_answer_action_prompt_over_refusal.txt",
}

_ANSWER_RE = re.compile(r"<ANSWER>(.*?)</ANSWER>", re.DOTALL)
_STATE_RE = re.compile(r"<STATE>(.*?)</STATE>", re.DOTALL)


def load_system_prompt(variant: str = "default") -> str:
    """Return the TRACE system prompt for `variant` ("default" or "over_refusal")."""
    try:
        name = PROMPT_VARIANTS[variant]
    except KeyError:
        raise ValueError(f"unknown prompt variant {variant!r}; choose from {sorted(PROMPT_VARIANTS)}")
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


def format_trajectory(messages: List[Dict[str, str]]) -> str:
    """Collapse a chat history into the [Turn N] trajectory string.

    `messages` is a standard chat list of {"role": "user"|"assistant", "content": str}. System
    messages are ignored. The last message must be from the user; it becomes the final, unanswered
    turn. Assistant messages should be the ANSWER text the user actually saw, never a raw
    STATE+ANSWER completion.
    """
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if not turns or turns[-1]["role"] != "user":
        raise ValueError("the last message must be a user message")

    history, current = turns[:-1], turns[-1]["content"]
    lines: List[str] = []
    turn_num, i = 1, 0
    while i < len(history):
        lines.append(f"[Turn {turn_num}]")
        if history[i]["role"] == "user":
            lines.append(f"USER: {history[i]['content']}")
            i += 1
            if i < len(history) and history[i]["role"] == "assistant":
                lines.append(f"ASSISTANT: {history[i]['content']}")
                i += 1
        else:
            lines.append(f"ASSISTANT: {history[i]['content']}")
            i += 1
        turn_num += 1
    lines.append(f"[Turn {turn_num}]")
    lines.append(f"USER: {current}")
    return "\n".join(lines)


def build_messages(messages: List[Dict[str, str]], variant: str = "default") -> List[Dict[str, str]]:
    """Return the two-message request (system prompt + collapsed trajectory) to send to the model."""
    return [
        {"role": "system", "content": load_system_prompt(variant)},
        {"role": "user", "content": format_trajectory(messages)},
    ]


def parse_output(raw: str) -> Tuple[Optional[dict], str, str]:
    """Split a raw completion into (state, state_text, answer).

    `state` is the STATE block parsed as JSON, or None if absent or malformed. `state_text` is the
    raw STATE contents. `answer` is the ANSWER block; if the model produced no ANSWER tags, the
    STATE block is stripped and whatever remains is returned, mirroring the evaluation code.
    """
    if not raw:
        return None, "", ""
    # Qwen3 may emit a <think> block before the STATE; drop it.
    raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL)

    state_match = _STATE_RE.search(raw)
    state_text = state_match.group(1).strip() if state_match else ""
    state: Optional[dict] = None
    if state_text:
        try:
            state = json.loads(state_text)
        except json.JSONDecodeError:
            state = None

    answer_match = _ANSWER_RE.search(raw)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer = _STATE_RE.sub("", raw).strip()
    return state, state_text, answer

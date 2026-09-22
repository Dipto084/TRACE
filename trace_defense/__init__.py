"""TRACE: Trajectory Aware Reasoning for Multi-Turn Adversarial Conversation Evaluation.

Inference utilities for the released TRACE models. The pieces:

    format_trajectory   collapse a chat history into the [Turn N] format the models were trained on
    parse_output        split a raw completion into its STATE assessment and the user-facing ANSWER
    load_system_prompt  the system prompt that defines the STATE/ANSWER contract
    TraceModel          stateless client over any OpenAI-compatible endpoint serving a TRACE model
    Conversation        stateful multi-turn helper built on TraceModel

See README.md for the drop-in proxy server (`python -m trace_defense.server`), which makes a TRACE model
usable by any attack framework that speaks the OpenAI chat API.
"""

from .formatting import format_trajectory, parse_output, load_system_prompt, PROMPT_VARIANTS
from .client import TraceModel, Conversation, TraceResponse

__all__ = [
    "format_trajectory",
    "parse_output",
    "load_system_prompt",
    "PROMPT_VARIANTS",
    "TraceModel",
    "Conversation",
    "TraceResponse",
]

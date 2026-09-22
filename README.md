# TRACE: Trajectory Aware Reasoning for Multi-Turn Adversarial Conversation Evaluation

Official repository for the paper ([arXiv:2608.15594](https://arxiv.org/abs/2608.15594)).

TRACE is a defense against multi-turn jailbreaks. Before answering each user turn, the model
reasons over the whole conversation so far in a `<STATE>` block — manipulation cues, a benign and
an adversarial reading of the user's intent, a jailbreak score, and an action (`ALLOW`,
`CAUTION`, `DECLINE`) — and then writes its reply in an `<ANSWER>` block. Only the answer is shown
to the user.

## Models

| Model | Base | HF |
|---|---|---|
| Llama3.1-8B-TRACE | Llama-3.1-8B-Instruct | [`Dipto084/Llama3.1-8B-TRACE`](https://huggingface.co/Dipto084/Llama3.1-8B-TRACE) |
| Qwen3-8B-TRACE | Qwen3-8B | [`Dipto084/Qwen3-8B-TRACE`](https://huggingface.co/Dipto084/Qwen3-8B-TRACE) |

Both are full merged weights in bf16 and load with vanilla `transformers` / vLLM. They must be
prompted the way they were trained, which is what this repository provides.

## How the models expect to be used

Two things differ from an ordinary chat model, and both are handled by this package:

1. **The trajectory is one message.** The history is not sent as a list of role-tagged messages.
   It is collapsed into a single user message with numbered turns, under the TRACE system prompt
   (`prompts/state_answer_action_prompt.txt`):

   ```
   [Turn 1]
   USER: ...
   ASSISTANT: ...
   [Turn 2]
   USER: ...
   ```

   The last turn contains only the user message — that is the one being answered.

2. **Only the ANSWER goes back into the history.** The model's completion is
   `<STATE>{...}</STATE><ANSWER>...</ANSWER>`. The answer is what the user sees and what is
   appended to the history for the next turn. The STATE block is for logging and routing and
   never re-enters the context.

This is exactly the protocol used for every evaluation in the paper.

## Install

```bash
pip install -r requirements.txt
```

The package has no model-side dependencies; it talks to any OpenAI-compatible server (vLLM,
SGLang, ...) that hosts the weights.

## Use with any attack framework (proxy)

The proxy makes a TRACE model look like an ordinary chat model. An attack framework — or any
client — sends the conversation so far as standard `messages` and gets a reply back. The
trajectory collapse, the system prompt, and the STATE stripping happen in the proxy, so no
framework-side changes are needed.

```bash
# 1. serve the weights (any OpenAI-compatible server; this uses vLLM on port 30003)
MODEL=Dipto084/Llama3.1-8B-TRACE scripts/serve_vllm.sh

# 2. run the proxy in front of it
python -m trace_defense.server --backend http://localhost:30003/v1 \
                               --model Dipto084/Llama3.1-8B-TRACE --port 8000

# 3. point the framework's target at http://localhost:8000/v1 with model name Dipto084/Llama3.1-8B-TRACE
```

Each response also carries the turn's safety assessment in a non-standard `trace_state` field
(the parsed STATE block), for logging. Client-supplied system messages are dropped — TRACE uses its
own — and a warning is logged.

Defaults reproduce the paper's attack evaluations: `temperature 0`, and a `max_tokens` budget of
4096 (raise it with `--max-tokens` if the backend allows; the STATE block precedes the answer, so
the budget must cover both). Use `--variant over_refusal` for the prompt used in the PHTest
over-refusal measurement.

## Use from Python

```python
from trace_defense import TraceModel, Conversation

model = TraceModel(model="Dipto084/Llama3.1-8B-TRACE", base_url="http://localhost:30003/v1")
conv = Conversation(model)

resp = conv.send("What's the etiquette I should know before visiting temples in Kyoto?")
print(resp.answer)           # user-facing reply
print(resp.action)           # "ALLOW" | "CAUTION" | "DECLINE"
print(resp.jailbreak_score)  # 1-5
print(resp.state)            # full parsed STATE dict
```

`Conversation` keeps the history the way the evaluations did — assistant turns hold only the
answer text. For a stateless call with your own history, use `model.chat(messages)`.

Building blocks, if you want to wire things up yourself:

```python
from trace_defense import format_trajectory, load_system_prompt, parse_output

messages = [
    {"role": "system", "content": load_system_prompt()},          # or load_system_prompt("over_refusal")
    {"role": "user", "content": format_trajectory(history)},      # history: standard chat list ending in a user turn
]
raw = client.chat.completions.create(model=..., messages=messages, temperature=0).choices[0].message.content
state, state_text, answer = parse_output(raw)
```

See `examples/quickstart.py` for a runnable multi-turn example against either the proxy or the
backend directly.

## Baselines released with the paper

| Baseline | HF |
|---|---|
| Red-Queen-Guard (Llama-3.1-8B-Instruct, DPO) | [`Dipto084/rqg_llama31_8b`](https://huggingface.co/Dipto084/rqg_llama31_8b) |
| X-Guard (Llama-3.1-8B-Instruct, XGuard-Train SFT) | [`Dipto084/Llama-3.1-8B-XGuard-merged`](https://huggingface.co/Dipto084/Llama-3.1-8B-XGuard-merged) |

These are ordinary chat models: standard messages, standard chat template, no system prompt.

## Citation

```bibtex
@article{miah2026trace,
  title   = {TRACE: Trajectory Aware Reasoning for Multi-Turn Adversarial Conversation Evaluation},
  author  = {Miah, Md Messal Monem and Anika, Adrita and Yu, Zhiyuan and Huang, Ruihong},
  journal = {arXiv preprint arXiv:2608.15594},
  year    = {2026}
}
```

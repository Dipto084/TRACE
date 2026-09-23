# Attack evaluations from the paper

The two multi-turn attack harnesses used for the numbers in the paper, with the target-side
changes and the exact configuration for the 120-behavior test set (80 HarmBench + 40
JailbreakBench, `test_120.csv`, identical in both harnesses).

Both harnesses can also be run **unmodified** against the [proxy](../README.md#use-with-any-attack-framework-proxy):
point their target at `http://localhost:8000/v1`. What is here is the in-process variant we
actually ran, kept so the reported numbers can be reproduced exactly.

| Attack | Dir | Paper setting | Entry point |
|---|---|---|---|
| X-Teaming ([Rahman et al., 2025](https://github.com/salman-lui/x-teaming)) | `xteaming/` | 120 behaviors × 3 strategies × ≤7 turns; attacker Qwen3-32B (T=0.3), TextGrad off; plans by GPT-4o; judge GPT-4o | `setup.sh`, then `python main.py -c config/config_test120_trace.yaml` |
| Chain-of-Attack ([Yang et al., 2024](https://github.com/YancyKahn/CoA), MIT) | `coa/` | 120 behaviors; attacker Qwen3-32B, 3 streams, ≤5 rounds × 6 iterations, DP walk; judge GPT-4o on CoA's 1–10 scale | `run.sh` |

The target is always TRACE at temperature 0 with the system prompt in
[`prompts/state_answer_action_prompt.txt`](../prompts/state_answer_action_prompt.txt); only the
`<ANSWER>` block is returned to the attacker and kept in the history. Both harnesses use
`max_tokens 8192` / `max_model_len 16384` for the target, which is more headroom than the proxy's
default `--max-tokens 4096`; the STATE block precedes the answer, so the budget must cover both.

## X-Teaming (`xteaming/`)

Upstream has no license file, so we ship a patch rather than a copy. `setup.sh` clones
`salman-lui/x-teaming` at `db964e6` (2025-05-21), applies `xteaming_trace.patch`, and copies in
the prompt, `test_120.csv`, the attack plans and the config.

What the patch changes (all in the target/plumbing, nothing in the attacker's strategy set):

- `agents/target_model.py`: `SFTTargetModel` — collapses the history into the numbered
  `[Turn N]` trajectory, sends it under the TRACE system prompt, parses `<ANSWER>` back out and
  stores only that in the history (the same protocol as `trace_defense/formatting.py`). Selected
  by `sft_variant: state_answer_action` in the target config.
- `agents/base_agent.py`: for OpenAI-compatible servers, honour `max_tokens` / `max_model_len`
  from the config and strip `<think>…</think>` from reasoning attackers (Qwen3).
- `agents/attacker_agent.py`: the GPT-4o judge scores the **full** target response instead of
  X-Teaming's 512-token truncation, and the raw `<STATE>…<ANSWER>` completion is logged per turn.
- `main.py`: `sft_variant` dispatch, per-run output tag, clamping to the number of strategies a
  plan actually has, skip-and-continue on attacker refusal, and a flat per-turn attack log.
- `generate_attack_plans.py`: accepts JailbreakBench CSVs as well as HarmBench's.

`data/attack_plans_test120.json` are the GPT-4o-generated attack plans (persona / context /
approach per behavior) used for the paper's runs; plan generation is stochastic, so reuse them
rather than regenerating for a like-for-like comparison. To regenerate:
`python generate_attack_plans.py -c config/config_test120_trace.yaml`.

Run (attacker on `:30002`, target on `:30003`, `OPENAI_API_KEY` set for plans/judge):

```bash
evals/xteaming/setup.sh                      # → evals/xteaming/x-teaming
cd evals/xteaming/x-teaming
vllm serve Qwen/Qwen3-32B --port 30002 --max-model-len 16384 &
MODEL=Dipto084/Llama3.1-8B-TRACE PORT=30003 MAX_MODEL_LEN=16384 ../../../scripts/serve_vllm.sh &
python main.py -c config/config_test120_trace.yaml
```

Results land in `attacks/test120_trace_<date>/` (`all_results_*.json` plus a flat `attack_log.json`).

## Chain-of-Attack (`coa/`)

A copy of the CoA code (MIT) with our modifications; `CHANGES.md` documents every change
against upstream (the SLURM scripts it mentions in §10 are replaced by `run.sh` here). In short:
local vLLM attacker/target (`language_models.VLLMModel`), local SimCSE and moderation services in
place of the remote ones, per-stream early stopping, and the TRACE target wrapper
(`sft_wrapper.py`, applied in `main.py` when `sft_variant` is set).

The paper's run is the `coa-test120-trace` baseline in `experiment.py`
(target model key `trace-vllm`; set `TRACE_MODEL` to swap the weights, e.g. to
`Dipto084/Qwen3-8B-TRACE`). `run.sh` lists the four services to start and runs it; see
`main.py --help` for the undefended and other-target baselines.

#!/usr/bin/env bash
# Clone upstream X-Teaming at the commit we patched, apply the TRACE patch, and drop in
# the prompt, behaviors, attack plans and config used in the paper.
#   evals/xteaming/setup.sh [target_dir]      (default: ./x-teaming)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DEST="${1:-$HERE/x-teaming}"
UPSTREAM=https://github.com/salman-lui/x-teaming.git
SHA=db964e6630b6a69b78e093705c89fd34865b6c27

git clone "$UPSTREAM" "$DEST"
git -C "$DEST" checkout -q "$SHA"
git -C "$DEST" apply --whitespace=nowarn "$HERE/xteaming_trace.patch"

cp "$ROOT/prompts/state_answer_action_prompt.txt" "$DEST/agents/"   # read by SFTTargetModel
mkdir -p "$DEST/behaviors" "$DEST/strategies"
cp "$HERE/data/test_120.csv"              "$DEST/behaviors/"
cp "$HERE/data/attack_plans_test120.json" "$DEST/strategies/"
cp "$HERE/configs/config_test120_trace.yaml" "$DEST/config/"
echo "X-Teaming + TRACE patch ready in $DEST"

#!/usr/bin/env bash
# analysis/run_all_offline.sh
#
# Orchestrates the full "offline" (GPU eval only, zero training) pipeline
# while the operator is away:
#   (A) tau-sweep eval        - 27 runs x 3 conditions = 81 jobs -> eval_logs_tau_sweep/
#   (B) non-DR re-eval        - A-E matrix, 75 jobs             -> eval_logs_nondr/
#   (C) trials>1 eval-noise   - B/C/D/E x gmd_03/gmd_04, R=5     -> eval_logs_trials5/
#   (D) far-future masking    - Model C x zero/noise/shuffle     -> eval_logs_mask_{mode}/
#   + aggregation             - analysis/eval/aggregate_offline.py -> analysis/outputs/
#
# Idempotent: every step below is a run_eval_matrix.py / run_eval_tau_sweep.py
# invocation with its own --manifest file; each driver already skips any job
# whose manifest entry has status=="success" (see both scripts' main()), so
# re-running this script after a partial failure only retries what didn't
# finish - nothing gets re-launched from scratch.
#
# Gated but non-fatal per cell: after each step (unless --dry_run), this
# script checks that step's manifest.json (analysis/eval/gate_check.py) and
# logs every individual failed job - it does NOT abort just because some
# cells failed. It only aborts the whole pipeline if that step's
# failed/total exceeds --max_fail_rate (default 0.20 = 20%). Cells that
# failed stay in the manifest with status="failed" and are retried
# automatically the next time this same step's driver command runs (see each
# driver's "already recorded as success" skip check) - nothing here clears or
# rewrites the manifest, it only reads it.
#
# Usage:
#   analysis/run_all_offline.sh --dry_run              # print every command; run/write nothing
#   analysis/run_all_offline.sh                         # actually run everything (long, GPU-bound)
#   analysis/run_all_offline.sh --max_fail_rate=0.3     # raise/lower the per-step abort threshold
#
# Not run by anyone other than a human after reviewing the tau-scale-override
# and far-future-mask-range diffs (scripts/rsl_rl/play_sim_rhythm.py,
# scripts/rsl_rl/play_sim_midi.py, scripts/rsl_rl/obs_mask.py).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."  # repo root

PYTHON="${PYTHON:-python}"

DRY_RUN=""
MAX_FAIL_RATE="0.20"
for arg in "$@"; do
    case "$arg" in
        --dry_run) DRY_RUN="--dry_run" ;;
        --max_fail_rate=*) MAX_FAIL_RATE="${arg#*=}" ;;
    esac
done

gate() {
    local out_dir="$1"
    local manifest="$2"
    local label="$3"

    if [[ -n "$DRY_RUN" ]]; then
        echo "[gate:${label}] skipped (dry run)"
        return 0
    fi

    # gate_check.py exits 1 (which - combined with `set -e` - stops this
    # script) only if the failure rate exceeds $MAX_FAIL_RATE; individual
    # failures below that threshold are logged (see its stdout) and this
    # function returns 0, so the pipeline continues to the next step.
    $PYTHON analysis/eval/gate_check.py \
        --manifest "$manifest" \
        --eval_logs_root "$out_dir" \
        --max_fail_rate "$MAX_FAIL_RATE" \
        --label "$label"
}

echo "########################################################################"
echo "# (A) tau-sweep eval: 27 runs x 3 conditions (double_160/gmd_03/gmd_04)"
echo "########################################################################"
$PYTHON -u -m analysis.eval.run_eval_tau_sweep $DRY_RUN \
    --eval_logs_root eval_logs_tau_sweep \
    --manifest eval_logs_tau_sweep/eval_tau_sweep_manifest.json
gate eval_logs_tau_sweep eval_logs_tau_sweep/eval_tau_sweep_manifest.json "A-tau-sweep"

echo "########################################################################"
echo "# (B) non-DR re-eval of the A-E matrix (75 jobs)"
echo "########################################################################"
$PYTHON -u -m analysis.eval.run_eval_matrix --priority $DRY_RUN \
    --task_override Template-Porcaro-2026-ModelB-user0 \
    --eval_logs_root eval_logs_nondr \
    --manifest eval_logs_nondr/eval_matrix_manifest.json
gate eval_logs_nondr eval_logs_nondr/eval_matrix_manifest.json "B-nondr"

echo "########################################################################"
echo "# (C) trials>1 eval-noise study: B/C/D/E x gmd_03/gmd_04, R=5 trials"
echo "########################################################################"
$PYTHON -u -m analysis.eval.run_eval_matrix --priority $DRY_RUN \
    --models B,C,D,E \
    --conditions gmd_03_high_bpm138,gmd_04_extreme_bpm170 \
    --trials_per_condition 5 \
    --eval_logs_root eval_logs_trials5 \
    --manifest eval_logs_trials5/eval_matrix_manifest.json
gate eval_logs_trials5 eval_logs_trials5/eval_matrix_manifest.json "C-trials5"

echo "########################################################################"
echo "# (D) Model C far-future (0.5-1.0s) masking ablation: zero / noise / shuffle"
echo "########################################################################"
for mode in zero noise shuffle; do
    echo "--- mask_mode=${mode} ---"
    $PYTHON -u -m analysis.eval.run_eval_matrix --priority $DRY_RUN \
        --models C \
        --mask_mode "$mode" \
        --eval_logs_root "eval_logs_mask_${mode}" \
        --manifest "eval_logs_mask_${mode}/eval_matrix_manifest.json"
    gate "eval_logs_mask_${mode}" "eval_logs_mask_${mode}/eval_matrix_manifest.json" "D-mask-${mode}"
done

echo "########################################################################"
echo "# Aggregation -> analysis/outputs/"
echo "########################################################################"
if [[ -n "$DRY_RUN" ]]; then
    echo "[aggregation] skipped (dry run) - would run: $PYTHON -m analysis.eval.aggregate_offline"
else
    $PYTHON -u -m analysis.eval.aggregate_offline
fi

echo "[ALL DONE]"

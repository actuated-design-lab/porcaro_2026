"""Layer2a evaluation-matrix driver (subprocess wrapper) - DO NOT RUN YET.

*** STATUS AS OF WRITING: GPU is still busy with training (Model C seed5, or
    already moved on to the MLP sweep). This file is staged for FUTURE use
    only and has not been executed. ***

When the GPU is free, a human should first run:

    python analysis/eval/run_eval_matrix.py --dry_run

and review every command it prints, and only drop --dry_run once that output
has been checked, since dropping --dry_run launches real
scripts/rsl_rl/play_sim_rhythm.py / play_sim_midi.py subprocesses (each of
which imports isaaclab/isaacsim/torch and uses the GPU).

This driver script itself never imports isaaclab / isaacsim / omni / torch -
it only reads discover.py's (pandas) run table and shells out to the two
play_sim_*.py scripts via subprocess, which is where the actual simulator
usage lives. Building the command list and printing it (--dry_run) touches
no GPU and does not import anything beyond pandas + stdlib.

Design:
- Input: analysis.harness.discover.discover_all_runs() - only rows with
  status == "completed" are scheduled (in-progress/incomplete runs are
  skipped automatically; discover_all_runs() itself never opens the
  in-progress run's contents, see discover.py).
- Checkpoint: {run_dir}/model_1499.pt is passed via --checkpoint directly,
  bypassing experiment_name-based checkpoint discovery entirely (relevant
  because of the experiment_name bug documented in discover.py's module
  docstring - each run_dir is unambiguous on its own).
- agent entry point: A/B/C (LSTM) -> rsl_rl_lstm_cfg_entry_point,
  D/E (MLP)        -> rsl_rl_mlp_cfg_entry_point.
- task: Template-Porcaro-2026-ModelB-DR-user0 (fixed).
- BASIC = {single_4, single_8, double} x {60, 120, 160} BPM, run via
  play_sim_rhythm.py. HEAVY = {(single_8, 160), (double, 160)} get
  --heavy_trials (default R=5) trials; every other BASIC condition and every
  GMD condition gets exactly 1 trial.
- GMD = 4 fixed MIDI files under eval_assets/midi/, run via play_sim_midi.py.
- trial_seed = 1000 + trial (fixed per trial index, so re-running trial t for
  any (model, seed, condition) is reproducible).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # allow `python analysis/eval/run_eval_matrix.py` from repo root

from analysis.harness.discover import discover_all_runs  # noqa: E402

TASK_ID = "Template-Porcaro-2026-ModelB-DR-user0"
CHECKPOINT_ITER = 1499
TRIAL_SEED_BASE = 1000

AGENT_BY_MODEL: dict[str, str] = {
    "A": "rsl_rl_lstm_cfg_entry_point",
    "B": "rsl_rl_lstm_cfg_entry_point",
    "C": "rsl_rl_lstm_cfg_entry_point",
    "D": "rsl_rl_mlp_cfg_entry_point",
    "E": "rsl_rl_mlp_cfg_entry_point",
}

# Template-Porcaro-2026-ModelB-DR-user0's env_cfg defaults to
# lookahead_horizon=0.5 / use_frame_stacking=False (observation_space=35),
# which only matches models B and D. A/C/E were trained with different
# lookahead_horizon / frame stacking (see analysis/harness/identify.py's
# classify_model() and logs/rsl_rl/MODEL_MAP.md) and their checkpoints
# therefore failed to load into the unmodified task with an observation-space
# size mismatch until play_sim_rhythm.py/play_sim_midi.py gained the
# --lookahead_horizon/--use_frame_stacking/--frame_stack_k overrides (mirrors
# train.py:178-199).
MODEL_ENV_OVERRIDES: dict[str, dict] = {
    "A": {"lookahead_horizon": 0.1, "use_frame_stacking": False, "frame_stack_k": 1},
    "B": {"lookahead_horizon": 0.5, "use_frame_stacking": False, "frame_stack_k": 1},
    "C": {"lookahead_horizon": 1.0, "use_frame_stacking": False, "frame_stack_k": 1},
    "D": {"lookahead_horizon": 0.5, "use_frame_stacking": False, "frame_stack_k": 1},
    "E": {"lookahead_horizon": 0.5, "use_frame_stacking": True, "frame_stack_k": 5},
}

BASIC_PATTERNS = ["single_4", "single_8", "double"]
BASIC_BPMS = [60, 120, 160]
BASIC: list[tuple[str, int]] = [(p, b) for p in BASIC_PATTERNS for b in BASIC_BPMS]

HEAVY: set[tuple[str, int]] = {("single_8", 160), ("double", 160)}

GMD: list[str] = [
    "eval_assets/midi/gmd_01_low_bpm80.mid",
    "eval_assets/midi/gmd_02_mid_bpm105.mid",
    "eval_assets/midi/gmd_03_high_bpm138.mid",
    "eval_assets/midi/gmd_04_extreme_bpm170.mid",
]

# Monday priority queue (docs/monday_run.md section 2): condition-outer /
# (model, seed)-inner, trial=0 only, so a queue cut off at any point still
# contains the highest-priority condition across every available model/seed
# before moving to the next condition. At 5 fully-trained models x 5 seeds
# this is 3 x 25 = 75 jobs; with fewer trained models/seeds it is smaller.
MONDAY_PRIORITY_CONDITIONS: list[tuple[str, str]] = [
    ("basic", "double_160"),
    ("gmd", "gmd_03_high_bpm138"),
    ("gmd", "gmd_04_extreme_bpm170"),
]
MONDAY_MODEL_PRIORITY: list[str] = ["B", "D", "A", "C", "E"]


def trial_count_for(pattern: str, bpm: int, heavy_r: int) -> int:
    return heavy_r if (pattern, bpm) in HEAVY else 1


def build_rhythm_command(python_exe: str, checkpoint: Path, agent: str, model: str, pattern: str, bpm: int, trial: int) -> list[str]:
    trial_seed = TRIAL_SEED_BASE + trial
    overrides = MODEL_ENV_OVERRIDES[model]
    return [
        python_exe, "scripts/rsl_rl/play_sim_rhythm.py",
        "--checkpoint", str(checkpoint),
        "--task", TASK_ID,
        "--agent", agent,
        "--lookahead_horizon", str(overrides["lookahead_horizon"]),
        *(["--use_frame_stacking", "--frame_stack_k", str(overrides["frame_stack_k"])]
          if overrides["use_frame_stacking"] else []),
        "--pattern", pattern,
        "--bpm", str(bpm),
        "--trial", str(trial),
        "--seed", str(trial_seed),
        "--headless",
        "--num_envs", "1",
    ]


def build_midi_command(python_exe: str, checkpoint: Path, agent: str, model: str, midi_path: str, trial: int) -> list[str]:
    trial_seed = TRIAL_SEED_BASE + trial
    overrides = MODEL_ENV_OVERRIDES[model]
    return [
        python_exe, "scripts/rsl_rl/play_sim_midi.py",
        "--checkpoint", str(checkpoint),
        "--task", TASK_ID,
        "--agent", agent,
        "--lookahead_horizon", str(overrides["lookahead_horizon"]),
        *(["--use_frame_stacking", "--frame_stack_k", str(overrides["frame_stack_k"])]
          if overrides["use_frame_stacking"] else []),
        "--midi", midi_path,
        "--trial", str(trial),
        "--seed", str(trial_seed),
        "--headless",
        "--num_envs", "1",
    ]


def build_eval_plan(runs_df: pd.DataFrame, heavy_r: int = 5, python_exe: str | None = None) -> list[dict]:
    """Turn a discover_all_runs() DataFrame into a flat list of eval jobs.

    Only rows with status == "completed" and a model in AGENT_BY_MODEL are
    scheduled - unclassified conditions (model is None/unmapped) and the
    protected in-progress run are silently skipped.
    """
    python_exe = python_exe or sys.executable
    plan: list[dict] = []

    completed = runs_df[runs_df["status"] == "completed"]
    for row in completed.itertuples():
        model = row.model
        if model not in AGENT_BY_MODEL:
            continue
        agent = AGENT_BY_MODEL[model]
        checkpoint = Path(row.run_dir) / f"model_{CHECKPOINT_ITER}.pt"

        for pattern, bpm in BASIC:
            for trial in range(trial_count_for(pattern, bpm, heavy_r)):
                cmd = build_rhythm_command(python_exe, checkpoint, agent, model, pattern, bpm, trial)
                plan.append(
                    dict(
                        model=model,
                        seed=row.seed,
                        run_dir=row.run_dir,
                        kind="basic",
                        condition=f"{pattern}_{bpm}",
                        trial=trial,
                        cmd=cmd,
                    )
                )

        for midi_path in GMD:
            condition = Path(midi_path).stem
            for trial in range(1):  # GMD conditions are never in HEAVY -> exactly 1 trial
                cmd = build_midi_command(python_exe, checkpoint, agent, model, midi_path, trial)
                plan.append(
                    dict(
                        model=model,
                        seed=row.seed,
                        run_dir=row.run_dir,
                        kind="gmd",
                        condition=condition,
                        trial=trial,
                        cmd=cmd,
                    )
                )

    return plan


def build_priority_plan(
    runs_df: pd.DataFrame,
    condition_order: list[tuple[str, str]] = MONDAY_PRIORITY_CONDITIONS,
    model_priority: list[str] = MONDAY_MODEL_PRIORITY,
    trials_per_condition: int = 1,
    python_exe: str | None = None,
) -> list[dict]:
    """Condition-outer / (model, seed)-inner ordering, for the Monday priority queue.

    Unlike build_eval_plan() (run-outer / condition-inner, all 9 BASIC + 4
    GMD conditions per run, HEAVY conditions get heavy_r trials), this walks
    condition_order first so a queue cut off at any point already contains
    the highest-priority condition across every model/seed pair before
    moving to the next condition. Every condition here gets exactly
    trials_per_condition trials (default 1) - it does not consult HEAVY/
    trial_count_for, since the Monday batch is an explicit, separate, smaller
    design from build_eval_plan()'s full matrix.
    """
    python_exe = python_exe or sys.executable

    completed = runs_df[runs_df["status"] == "completed"].copy()
    completed = completed[completed["model"].isin(AGENT_BY_MODEL)]

    rank = {m: i for i, m in enumerate(model_priority)}
    completed["_rank"] = completed["model"].map(lambda m: rank.get(m, len(model_priority)))
    completed = completed.sort_values(["_rank", "seed"]).drop(columns="_rank")

    plan: list[dict] = []
    for kind, condition in condition_order:
        for row in completed.itertuples():
            agent = AGENT_BY_MODEL[row.model]
            checkpoint = Path(row.run_dir) / f"model_{CHECKPOINT_ITER}.pt"

            for trial in range(trials_per_condition):
                if kind == "basic":
                    pattern, bpm_str = condition.rsplit("_", 1)
                    cmd = build_rhythm_command(python_exe, checkpoint, agent, row.model, pattern, int(bpm_str), trial)
                elif kind == "gmd":
                    midi_path = next(p for p in GMD if Path(p).stem == condition)
                    cmd = build_midi_command(python_exe, checkpoint, agent, row.model, midi_path, trial)
                else:
                    raise ValueError(f"build_priority_plan: unknown kind {kind!r} for condition {condition!r}")

                plan.append(
                    dict(
                        model=row.model,
                        seed=row.seed,
                        run_dir=row.run_dir,
                        kind=kind,
                        condition=condition,
                        trial=trial,
                        cmd=cmd,
                    )
                )

    return plan


def _job_key(job: dict) -> str:
    return f"{job['model']}/seed{job['seed']}/{job['kind']}/{job['condition']}/trial{job['trial']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer2a evaluation-matrix driver (subprocess wrapper).")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Print every command that would be launched; do not execute anything.",
    )
    parser.add_argument(
        "--heavy_trials",
        type=int,
        default=5,
        help="Number of trials (R) for the HEAVY basic conditions (single_8@160, double@160).",
    )
    parser.add_argument(
        "--priority",
        action="store_true",
        default=False,
        help="Use the Monday priority queue (condition-outer, MONDAY_PRIORITY_CONDITIONS/"
             "MONDAY_MODEL_PRIORITY) instead of the full run-outer matrix from build_eval_plan().",
    )
    parser.add_argument(
        "--logs_rsl_rl_root",
        type=str,
        default=str(REPO_ROOT / "logs" / "rsl_rl"),
        help="Root directory passed to discover_all_runs().",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(REPO_ROOT / "eval_logs" / "eval_matrix_manifest.json"),
        help="JSON file used to record per-job status (only written when --dry_run is not set).",
    )
    args = parser.parse_args()

    runs_df = discover_all_runs(args.logs_rsl_rl_root)
    plan = (
        build_priority_plan(runs_df)
        if args.priority
        else build_eval_plan(runs_df, heavy_r=args.heavy_trials)
    )

    n_completed_runs = int((runs_df["status"] == "completed").sum())
    print(
        f"[run_eval_matrix] {len(plan)} evaluation subprocess invocation(s) planned "
        f"across {n_completed_runs} completed training run(s)."
    )
    for i, job in enumerate(plan):
        print(f"  [{i + 1:04d}] {_job_key(job)}")
        print(f"          {' '.join(job['cmd'])}")

    if args.dry_run:
        print("\n[Dry Run] Nothing was executed. Drop --dry_run to actually launch these jobs.")
        return

    manifest: dict = {}
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    for i, job in enumerate(plan):
        key = _job_key(job)
        if manifest.get(key, {}).get("status") == "success":
            print(f"[Skip] already recorded as success: {key}")
            continue

        print("\n" + "=" * 80)
        print(f"[{i + 1}/{len(plan)}] {key}")
        print("=" * 80)
        print("[Command]", " ".join(job["cmd"]))

        manifest[key] = {"status": "running", "started_at": datetime.now().isoformat()}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

        try:
            subprocess.run(job["cmd"], cwd=str(REPO_ROOT), check=True)
            manifest[key] = {"status": "success", "finished_at": datetime.now().isoformat()}
        except subprocess.CalledProcessError as e:
            manifest[key] = {"status": "failed", "error": str(e), "finished_at": datetime.now().isoformat()}

        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

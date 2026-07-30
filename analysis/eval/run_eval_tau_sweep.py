"""Evaluation driver for the 27-run tau x lookahead sweep (logs/rsl_rl_tau_sweep/).

Read-only discovery + subprocess wrapper, same safety contract as
analysis/harness/discover.py and analysis/eval/run_eval_matrix.py: no
isaaclab/isaacsim/torch import in this file itself, only reads small
params/*.yaml text files and lists (never opens) *.pt filenames. All GPU
usage lives in the play_sim_rhythm.py / play_sim_midi.py subprocesses this
script shells out to.

Design:
- Discovery: logs/rsl_rl_tau_sweep/porcaro_rslrl_lstm_modelB_DR/ is scanned
  directly (all 27 tau-sweep runs used --agent rsl_rl_lstm_cfg_entry_point,
  see docs/monday_run.md section 4-4) - run_name is NOT trusted for
  (tau, lookahead_horizon, seed) identity, only params/env.yaml
  (pam_tau_scale_range, lookahead_horizon) and params/agent.yaml (seed) are
  read, exactly like discover.py does for the A-E model matrix.
- classify_tau_cell() (analysis/harness/identify.py) replaces classify_model()
  - it keys off pam_tau_scale_range instead of policy/frame-stacking type.
- Output: eval_logs_tau_sweep/{run_tag}/{ckpt}/{condition}_trial{t}/ - a
  separate tree from eval_logs/, so the 81 tau-sweep eval jobs (27 runs x 3
  MONDAY_PRIORITY_CONDITIONS) never mix with the main A-E eval_logs/ output.
- Each play_sim_*.py invocation passes both --lookahead_horizon and
  --pam_tau_scale read from that specific run's own env.yaml (not a fixed
  MODEL_ENV_OVERRIDES table, since every one of the 27 runs has a distinct
  (tau, lh) pair) - this is what lets play_sim's own tau safety-check assert
  (ch_DF/F/G.current_tau_scale == --pam_tau_scale) catch a mismatch here too.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.harness.identify import classify_tau_cell, load_yaml  # noqa: E402
from analysis.eval.run_eval_matrix import (  # noqa: E402
    GMD,
    MONDAY_PRIORITY_CONDITIONS,
    CHECKPOINT_ITER,
    TASK_ID,
    TRIAL_SEED_BASE,
)

TAU_SWEEP_LOGS_ROOT = REPO_ROOT / "logs" / "rsl_rl_tau_sweep"
TAU_SWEEP_EXPERIMENT_DIR = "porcaro_rslrl_lstm_modelB_DR"  # all 27 runs used the LSTM agent cfg
TAU_SWEEP_AGENT = "rsl_rl_lstm_cfg_entry_point"
DEFAULT_EVAL_LOGS_ROOT = "eval_logs_tau_sweep"
DEFAULT_MANIFEST = REPO_ROOT / "eval_logs_tau_sweep" / "eval_tau_sweep_manifest.json"

TAU_RUN_DIR_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_tau[\d.]+_lh[\d.]+_seed(\d+)$"
)
MAX_ITERATIONS_HINT = 1500

# Same rationale as discover.py's STALE_PROTECTION_THRESHOLD_S: the
# max-mtime run dir is only "in progress" if it was touched recently -
# otherwise it's just the last tau-sweep run that ever finished.
STALE_PROTECTION_THRESHOLD_S = 30 * 60


def _list_checkpoint_iters(run_dir: Path) -> list[int]:
    iters = []
    for p in run_dir.iterdir():
        m = re.match(r"^model_(\d+)\.pt$", p.name)
        if m:
            iters.append(int(m.group(1)))
    return sorted(iters)


def discover_tau_runs(
    tau_sweep_root: str | Path = TAU_SWEEP_LOGS_ROOT,
    experiment_dir: str = TAU_SWEEP_EXPERIMENT_DIR,
    max_iterations_hint: int = MAX_ITERATIONS_HINT,
) -> pd.DataFrame:
    """Scan logs/rsl_rl_tau_sweep/{experiment_dir}/ and classify each run.

    Returns a DataFrame [tau, lh, seed, run_dir, timestamp, completed_iter,
    status, warnings], mirroring discover.py's discover_runs() shape.
    """
    root = Path(tau_sweep_root) / experiment_dir
    columns = ["tau", "lh", "seed", "run_dir", "timestamp", "completed_iter", "status", "warnings"]
    if not root.is_dir():
        return pd.DataFrame(columns=columns)

    run_dirs = sorted(p for p in root.iterdir() if p.is_dir() and TAU_RUN_DIR_RE.match(p.name))

    mtimes = {d: d.stat().st_mtime for d in run_dirs}
    protected = max(mtimes, key=mtimes.get) if mtimes else None
    if protected is not None and (time.time() - mtimes[protected]) > STALE_PROTECTION_THRESHOLD_S:
        protected = None

    rows = []
    for d in run_dirs:
        m = TAU_RUN_DIR_RE.match(d.name)
        timestamp, dirname_seed = m.group(1), int(m.group(2))

        if d == protected:
            rows.append(dict(
                tau=None, lh=None, seed=dirname_seed, run_dir=str(d), timestamp=timestamp,
                completed_iter=None, status="running (protected: params not read)",
                warnings="in_progress_run_skipped_params",
            ))
            continue

        iters = _list_checkpoint_iters(d)
        completed_iter = max(iters) if iters else -1
        status = "completed" if completed_iter >= (max_iterations_hint - 51) else "incomplete_or_crashed"

        warnings: list[str] = []
        tau = lh = None
        env_path, agent_path = d / "params" / "env.yaml", d / "params" / "agent.yaml"
        if not (env_path.exists() and agent_path.exists()):
            warnings.append("missing_params_yaml")
        else:
            env_y = load_yaml(env_path)
            agent_y = load_yaml(agent_path)
            cell = classify_tau_cell(env_y)
            if cell is None:
                warnings.append("not_degenerate_or_missing_tau_range")
            else:
                tau, lh = cell
            seed_yaml = agent_y.get("seed")
            if seed_yaml is not None and int(seed_yaml) != dirname_seed:
                warnings.append(f"seed_mismatch(dirname={dirname_seed},agent_yaml={seed_yaml})")

        rows.append(dict(
            tau=tau, lh=lh, seed=dirname_seed, run_dir=str(d), timestamp=timestamp,
            completed_iter=completed_iter, status=status, warnings=";".join(warnings),
        ))

    return pd.DataFrame(rows, columns=columns)


def build_tau_rhythm_command(
    python_exe: str, checkpoint: Path, tau: float, lh: float, pattern: str, bpm: int, trial: int,
    eval_logs_root: str,
) -> list[str]:
    trial_seed = TRIAL_SEED_BASE + trial
    return [
        python_exe, "scripts/rsl_rl/play_sim_rhythm.py",
        "--checkpoint", str(checkpoint),
        "--task", TASK_ID,
        "--agent", TAU_SWEEP_AGENT,
        "--lookahead_horizon", str(lh),
        "--pam_tau_scale", str(tau),
        "--pattern", pattern,
        "--bpm", str(bpm),
        "--trial", str(trial),
        "--seed", str(trial_seed),
        "--eval_logs_root", eval_logs_root,
        "--headless",
        "--num_envs", "1",
    ]


def build_tau_midi_command(
    python_exe: str, checkpoint: Path, tau: float, lh: float, midi_path: str, trial: int,
    eval_logs_root: str,
) -> list[str]:
    trial_seed = TRIAL_SEED_BASE + trial
    return [
        python_exe, "scripts/rsl_rl/play_sim_midi.py",
        "--checkpoint", str(checkpoint),
        "--task", TASK_ID,
        "--agent", TAU_SWEEP_AGENT,
        "--lookahead_horizon", str(lh),
        "--pam_tau_scale", str(tau),
        "--midi", midi_path,
        "--trial", str(trial),
        "--seed", str(trial_seed),
        "--eval_logs_root", eval_logs_root,
        "--headless",
        "--num_envs", "1",
    ]


def build_tau_eval_plan(
    tau_runs_df: pd.DataFrame,
    condition_order: list[tuple[str, str]] = MONDAY_PRIORITY_CONDITIONS,
    python_exe: str | None = None,
    eval_logs_root: str = DEFAULT_EVAL_LOGS_ROOT,
) -> list[dict]:
    """Condition-outer / (tau, lh)-cell / seed-inner ordering.

    Mirrors run_eval_matrix.py's build_priority_plan() ordering philosophy:
    condition outermost so a queue cut off early still covers every tau/lh
    cell for the highest-priority condition first. Within a condition, cells
    are ordered (tau, lh) then seed, matching the tau-sweep training queue's
    "seed outermost, 9 cells inner" only in the sense that all 3 seeds appear
    for a given cell together before moving to the next cell.
    """
    python_exe = python_exe or sys.executable
    completed = tau_runs_df[tau_runs_df["status"] == "completed"].copy()
    completed = completed.sort_values(["tau", "lh", "seed"])

    plan: list[dict] = []
    for kind, condition in condition_order:
        for row in completed.itertuples():
            checkpoint = Path(row.run_dir) / f"model_{CHECKPOINT_ITER}.pt"
            trial = 0  # single trial per cell, matching the Monday priority queue's trials_per_condition=1
            if kind == "basic":
                pattern, bpm_str = condition.rsplit("_", 1)
                cmd = build_tau_rhythm_command(
                    python_exe, checkpoint, row.tau, row.lh, pattern, int(bpm_str), trial, eval_logs_root
                )
            elif kind == "gmd":
                midi_path = next(p for p in GMD if Path(p).stem == condition)
                cmd = build_tau_midi_command(
                    python_exe, checkpoint, row.tau, row.lh, midi_path, trial, eval_logs_root
                )
            else:
                raise ValueError(f"build_tau_eval_plan: unknown kind {kind!r} for condition {condition!r}")

            plan.append(dict(
                tau=row.tau, lh=row.lh, seed=row.seed, run_dir=row.run_dir,
                kind=kind, condition=condition, trial=trial, cmd=cmd,
            ))

    return plan


def _job_key(job: dict) -> str:
    return f"tau{job['tau']}_lh{job['lh']}/seed{job['seed']}/{job['kind']}/{job['condition']}/trial{job['trial']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tau-sweep (27-run) evaluation driver.")
    parser.add_argument("--dry_run", action="store_true", default=False,
                         help="Print every command that would be launched; do not execute anything.")
    parser.add_argument("--tau_sweep_root", type=str, default=str(TAU_SWEEP_LOGS_ROOT),
                         help="Root directory passed to discover_tau_runs() (default logs/rsl_rl_tau_sweep/).")
    parser.add_argument("--eval_logs_root", type=str, default=DEFAULT_EVAL_LOGS_ROOT,
                         help="Output root for eval_logs_tau_sweep-style CSVs.")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST),
                         help="JSON file used to record per-job status (only written when --dry_run is not set).")
    parser.add_argument("--cells", type=str, default=None,
                         help="Comma-separated 'tau:lh' filters (e.g. '0.5:0.1,1.0:0.5') to restrict "
                              "the plan to specific cells (all seeds still included). Default: all 9 cells.")
    args = parser.parse_args()

    runs_df = discover_tau_runs(args.tau_sweep_root)
    n_completed = int((runs_df["status"] == "completed").sum())
    print(f"[run_eval_tau_sweep] discovered {len(runs_df)} tau-sweep run dir(s), {n_completed} completed.")
    if len(runs_df):
        print(runs_df.to_string(index=False))
    non_completed = runs_df[runs_df["status"] != "completed"]
    if len(non_completed):
        print(f"\n[run_eval_tau_sweep] WARNING: {len(non_completed)} run(s) not completed / not classifiable:")
        print(non_completed.to_string(index=False))

    plan = build_tau_eval_plan(runs_df, eval_logs_root=args.eval_logs_root)

    if args.cells:
        wanted = set()
        for pair in args.cells.split(","):
            tau_str, lh_str = pair.split(":")
            wanted.add((float(tau_str), float(lh_str)))
        plan = [job for job in plan if (job["tau"], job["lh"]) in wanted]

    print(f"\n[run_eval_tau_sweep] {len(plan)} evaluation subprocess invocation(s) planned "
          f"across {n_completed} completed tau-sweep run(s).")
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

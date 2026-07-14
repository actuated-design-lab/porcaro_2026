"""Read-only discovery of rsl_rl training runs.

Runs land in one of two fixed experiment_name folders, one per agent cfg
class, because scripts/rsl_rl/cli_args.py's update_rsl_rl_cfg() never applies
--experiment_name to agent_cfg.experiment_name (see the recon notes) - the
hardcoded `experiment_name` class attribute on the loaded agent cfg wins
instead:
  - rsl_rl_ppo_lstm_cfg.py -> "porcaro_rslrl_lstm_modelB_DR"      (Models A/B/C)
  - rsl_rl_ppo_mlp_cfg.py  -> "porcaro_rslrl_mlp_modelB_DR_lookahead5" (Models D/E)
Condition and seed are therefore recovered from the contents of each run's
params/env.yaml + params/agent.yaml, not from the directory name or which of
the two experiment_name folders a run happens to live in.

Safety:
- Only reads files under logs/rsl_rl/.../params/*.yaml (small text files) and
  lists (never opens) *.pt checkpoint filenames to read off their iteration
  number from the filename.
- The run directory with the most recent mtime is treated as "in progress"
  and is deliberately NOT opened (no params read, no checkpoint listing
  beyond a bare directory listing) - only its directory name is reported.
  This "most recent mtime" candidate is only actually protected if it was
  also written to within STALE_PROTECTION_THRESHOLD_S of now (see that
  constant and _protect_if_fresh() below) - otherwise nothing anywhere is
  being actively trained right now and the candidate is just the last run
  that ever finished, not one "in progress".
- No isaaclab / isaacsim / omni / torch import. No .pt files are loaded.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .identify import EXP_TAG_TO_MODEL, classify_model, load_yaml

RUN_DIR_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_seed(\d+)$")
CKPT_RE = re.compile(r"^model_(\d+)\.pt$")

# How long after its last write a run dir can still plausibly be "in
# progress". Existing runs write a checkpoint every save_interval=50
# iterations, observed at ~9-10 minutes/checkpoint - this is set to several
# times that cadence so a live run is never mistaken for stale between
# checkpoint writes, while a run whose training process has actually exited
# (or moved on to a different experiment_name folder for good) stops being
# protected shortly after, instead of forever.
STALE_PROTECTION_THRESHOLD_S = 30 * 60


def _protect_if_fresh(candidate: Path | None) -> Path | None:
    """Only treat `candidate` as the in-progress run if it was touched recently.

    The "most-recently-modified directory" a caller computes below is, on
    its own, just "latest across whatever folders exist" - it can't tell
    "still being written to right now" from "was the last thing ever
    written, and training has since stopped completely". Gating on mtime
    recency (metadata-only, consistent with this module's read-only
    contract) fixes that: a run stops being protected once nothing has
    touched it in a while, regardless of whether it's still the max-mtime
    dir across every known folder.
    """
    if candidate is None:
        return None
    age_s = time.time() - candidate.stat().st_mtime
    return candidate if age_s <= STALE_PROTECTION_THRESHOLD_S else None


def _list_checkpoint_iters(run_dir: Path) -> list[int]:
    """List (never open) model_*.pt filenames and parse their iteration number."""
    iters = []
    for p in run_dir.iterdir():
        m = CKPT_RE.match(p.name)
        if m:
            iters.append(int(m.group(1)))
    return sorted(iters)


def _expected_obs_dim(lookahead_horizon: float, dt_ctrl: float, use_frame_stacking: bool, frame_stack_k: int) -> int:
    lookahead_steps = int(lookahead_horizon / dt_ctrl)
    base_obs_dim = 10 + lookahead_steps
    return base_obs_dim * frame_stack_k if use_frame_stacking else base_obs_dim


def discover_runs(
    experiment_root: str | Path,
    max_iterations_hint: int = 1500,
    protected_dir: Path | None = None,
) -> pd.DataFrame:
    """Scan an experiment_name directory and classify each run.

    Args:
        protected_dir: if given, this exact directory is treated as
            "in progress" and skipped without opening its contents, instead
            of recomputing a per-folder max-mtime locally. Pass this
            explicitly when scanning multiple experiment_name folders
            together (see discover_all_runs()) so a folder that simply
            stopped receiving writes once training moved to a *different*
            folder isn't permanently misclassified as still running. If
            None, falls back to the old per-folder "max mtime within this
            folder" heuristic (kept for standalone/backward-compatible
            calls, but see the caveat below).

    Returns a DataFrame with columns:
      [model, seed, run_dir, timestamp, completed_iter, status, warnings]
    """
    experiment_root = Path(experiment_root)
    if not experiment_root.is_dir():
        raise FileNotFoundError(f"experiment_root does not exist: {experiment_root}")

    run_dirs = sorted(
        p for p in experiment_root.iterdir() if p.is_dir() and RUN_DIR_RE.match(p.name)
    )

    if protected_dir is None:
        # Standalone-call fallback: protect the most-recently-modified dir
        # WITHIN this folder only (stat() is metadata only, not a content
        # read). CAVEAT: this cannot distinguish "still being written right
        # now" from "was the last thing ever written to this folder" - once
        # training moves permanently to a different experiment_name folder,
        # the last run here will look protected forever. discover_all_runs()
        # avoids this by computing one global protected_dir across every
        # known folder and passing it in here explicitly.
        mtimes = {d: d.stat().st_mtime for d in run_dirs}
        protected_dir = _protect_if_fresh(max(mtimes, key=mtimes.get) if mtimes else None)

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        m = RUN_DIR_RE.match(run_dir.name)
        timestamp, seed_from_name = m.group(1), int(m.group(2))
        warnings: list[str] = []

        if run_dir == protected_dir:
            rows.append(
                dict(
                    model=None,
                    seed=seed_from_name,
                    run_dir=str(run_dir),
                    timestamp=timestamp,
                    completed_iter=None,
                    status="running (protected: params not read)",
                    warnings="in_progress_run_skipped_params",
                )
            )
            continue

        iters = _list_checkpoint_iters(run_dir)
        completed_iter = max(iters) if iters else -1
        near_final = completed_iter >= (max_iterations_hint - 51)
        status = "completed" if near_final else "incomplete_or_crashed"

        params_dir = run_dir / "params"
        env_path, agent_path = params_dir / "env.yaml", params_dir / "agent.yaml"
        model = None
        if not (env_path.exists() and agent_path.exists()):
            warnings.append("missing_params_yaml")
        else:
            env_y = load_yaml(env_path)
            agent_y = load_yaml(agent_path)

            model = classify_model(env_y, agent_y)
            if model is None:
                warnings.append("unclassified_condition")

            agent_seed = agent_y.get("seed")
            if agent_seed is not None and int(agent_seed) != seed_from_name:
                warnings.append(f"seed_mismatch(dirname={seed_from_name},agent_yaml={agent_seed})")

            try:
                dt_ctrl = float(env_y["sim"]["dt"]) * float(env_y["decimation"])
                lh = float(env_y["lookahead_horizon"])
                fs = bool(env_y.get("use_frame_stacking", False))
                fsk = int(env_y.get("frame_stack_k", 1))
                expected = _expected_obs_dim(lh, dt_ctrl, fs, fsk)
                actual = int(env_y["observation_space"])
                if expected != actual:
                    warnings.append(f"obs_space_mismatch(expected={expected},actual={actual})")
            except Exception as e:  # noqa: BLE001 - report, don't crash discovery
                warnings.append(f"obs_space_check_failed:{e!r}")

        rows.append(
            dict(
                model=model,
                seed=seed_from_name,
                run_dir=str(run_dir),
                timestamp=timestamp,
                completed_iter=completed_iter,
                status=status,
                warnings=";".join(warnings),
            )
        )

    return pd.DataFrame(
        rows,
        columns=["model", "seed", "run_dir", "timestamp", "completed_iter", "status", "warnings"],
    )


# The two experiment_name folders runs can currently land in - see module
# docstring. Kept as a list (not a set) so discover_all_runs()'s output order
# is stable (LSTM sweep first, then MLP sweep).
KNOWN_EXPERIMENT_DIRS: list[str] = [
    "porcaro_rslrl_lstm_modelB_DR",
    "porcaro_rslrl_mlp_modelB_DR_lookahead5",
]

_DISCOVERY_COLUMNS = ["model", "seed", "run_dir", "timestamp", "completed_iter", "status", "warnings"]


def discover_all_runs(
    logs_rsl_rl_root: str | Path,
    experiment_dirs: list[str] | None = None,
    max_iterations_hint: int = 1500,
) -> pd.DataFrame:
    """Scan every known experiment_name folder under logs/rsl_rl/ and concatenate.

    A missing folder (e.g. the MLP sweep hasn't produced any runs yet) is
    silently skipped rather than raising, so this is safe to call before
    Models D/E have ever been trained.

    The "in progress" run is protected GLOBALLY (single most-recently
    modified run dir across every existing experiment_name folder combined),
    not per-folder - see discover_runs()'s protected_dir docstring for why a
    per-folder-only heuristic permanently misclassifies the last run in a
    folder that training has since moved away from.
    """
    logs_rsl_rl_root = Path(logs_rsl_rl_root)
    experiment_dirs = experiment_dirs if experiment_dirs is not None else KNOWN_EXPERIMENT_DIRS

    existing_roots = [
        logs_rsl_rl_root / exp_dir_name
        for exp_dir_name in experiment_dirs
        if (logs_rsl_rl_root / exp_dir_name).is_dir()
    ]
    if not existing_roots:
        return pd.DataFrame(columns=_DISCOVERY_COLUMNS)

    # stat() is metadata only, not a content read - safe even if one of these
    # run dirs is the one actively being written to right now.
    all_run_dirs = [
        d
        for root in existing_roots
        for d in root.iterdir()
        if d.is_dir() and RUN_DIR_RE.match(d.name)
    ]
    global_protected_dir = _protect_if_fresh(
        max(all_run_dirs, key=lambda d: d.stat().st_mtime) if all_run_dirs else None
    )

    frames = [
        discover_runs(root, max_iterations_hint=max_iterations_hint, protected_dir=global_protected_dir)
        for root in existing_roots
    ]
    return pd.concat(frames, ignore_index=True)


def _manifest_key_to_exp_tag(key: str) -> tuple[str, int]:
    """Split a manifest key into (exp_tag, seed).

    Manifest keys are "{experiment_name}/seed{N}" where experiment_name is
    make_experiment_name(cond) == f"porcaro_{cond['exp_tag']}" (see
    scripts/rsl_rl/run_experiment_matrix.py). EXP_TAG_TO_MODEL is keyed by
    the bare exp_tag (e.g. "lstm_lookahead0.1"), so the leading "porcaro_"
    prefix must be stripped before doing the lookup.
    """
    experiment_name, seed_str = key.split("/seed")
    exp_tag = experiment_name.removeprefix("porcaro_")
    return exp_tag, int(seed_str)


def cross_check_manifest(discovered: pd.DataFrame, manifest_path: str | Path) -> list[str]:
    """Compare discovered (model, seed) pairs against experiment_matrix_manifest.json.

    Returns a flat list of human-readable warning strings (missing seeds,
    manifest-vs-log status mismatches, conditions the manifest knows about
    that classify_model() cannot label, etc). Does not raise on mismatch -
    this is a reporting pass only.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())

    warnings: list[str] = []

    discovered_pairs = {
        (row.model, row.seed) for row in discovered.itertuples() if row.model is not None
    }
    protected_pairs = {
        (row.model, row.seed) for row in discovered.itertuples() if row.status.startswith("running")
    }

    for key, info in manifest.items():
        exp_tag, seed = _manifest_key_to_exp_tag(key)
        model = EXP_TAG_TO_MODEL.get(exp_tag)
        if model is None:
            warnings.append(f"manifest_key_unmapped: '{key}' has no entry in EXP_TAG_TO_MODEL")
            continue

        status = info.get("status")
        if status == "running":
            if not protected_dir_seed_matches(discovered, seed):
                warnings.append(
                    f"manifest_says_running_but_no_protected_run_dir_matches_seed: {key}"
                )
            continue

        if (model, seed) not in discovered_pairs:
            warnings.append(
                f"missing_or_unclassified_in_logs: manifest has '{key}' (status={status}) "
                f"but no completed run classified as model={model} seed={seed} was found"
            )

    manifest_pairs = set()
    for key in manifest:
        exp_tag, seed = _manifest_key_to_exp_tag(key)
        model = EXP_TAG_TO_MODEL.get(exp_tag)
        if model is not None:
            manifest_pairs.add((model, seed))
    extra = discovered_pairs - manifest_pairs
    for model, seed in sorted(extra):
        warnings.append(f"log_run_not_in_manifest: model={model} seed={seed}")

    return warnings


def protected_dir_seed_matches(discovered: pd.DataFrame, seed: int) -> bool:
    protected_rows = discovered[discovered["status"].str.startswith("running")]
    return bool((protected_rows["seed"] == seed).any())


if __name__ == "__main__":
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    logs_rsl_rl_root = repo_root / "logs" / "rsl_rl"
    manifest_path = repo_root / "logs" / "experiment_matrix_manifest.json"

    df = discover_all_runs(logs_rsl_rl_root)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    if manifest_path.exists():
        print("\n--- manifest cross-check warnings ---")
        for w in cross_check_manifest(df, manifest_path):
            print(f"WARNING: {w}")
    else:
        print(f"\n[discover.py] manifest not found at {manifest_path}, skipping cross-check", file=sys.stderr)

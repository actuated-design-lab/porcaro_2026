"""Layer2a aggregation for the offline (A)-(D) eval studies.

Reads simulation_log.csv trees produced by:
  (A) analysis/eval/run_eval_tau_sweep.py       -> eval_logs_tau_sweep/
  (B) run_eval_matrix.py --task_override ...    -> eval_logs_nondr/
  (C) run_eval_matrix.py --trials_per_condition -> eval_logs_trials5/
  (D) run_eval_matrix.py --mask_mode ...        -> eval_logs_mask_{zero,noise,shuffle}/

and the original A-E matrix's analysis/outputs/eval_summary.csv /
eval_strikes_raw.csv (already produced by the Monday eval pass), and writes:
  - analysis/outputs/tau_optimal_lookahead.csv + fig_tau_vs_optimal_lookahead.png
  - analysis/outputs/memory_group_comparison.csv  (B/C/E vs A/D, CI'd)
  - analysis/outputs/trials5_eval_noise.csv
  - analysis/outputs/mask_comparison.csv

No isaaclab/torch import - pandas/numpy/scipy/matplotlib + this repo's
analysis.harness modules only. Every load_and_extract() call is best-effort:
a missing/empty input tree is reported and skipped rather than raising, so
this can be re-run at any point while (A)-(D) are still in flight.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.harness.strike_extract import extract_strikes  # noqa: E402
from analysis.harness.stats import seed_level, across_seed  # noqa: E402

OUTPUTS_DIR = REPO_ROOT / "analysis" / "outputs"

CONDITION_TRIAL_RE = re.compile(r"^(.+)_trial(\d+)$")
TAU_RUN_TAG_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_tau([\d.]+)_lh([\d.]+)_seed(\d+)$")


def _parse_condition_dir(condition_dir: str) -> tuple[str, str, int] | None:
    """'{pattern}_{bpm}bpm_trial{t}' | 'gmd_..._trial{t}' -> (kind, task, trial)."""
    m = CONDITION_TRIAL_RE.match(condition_dir)
    if not m:
        return None
    condition_part, trial_str = m.group(1), int(m.group(2))
    if condition_part.startswith("gmd_"):
        return "gmd", condition_part, trial_str
    if condition_part.endswith("bpm"):
        return "basic", condition_part[: -len("bpm")], trial_str
    return None


def extract_strikes_tree(eval_logs_root: Path) -> pd.DataFrame:
    """Walk {root}/*/*/*/simulation_log.csv and return a strike-level table
    enriched with [run_tag, task, eval_trial] (no model/seed - callers attach
    whatever identity columns make sense for their eval_logs_root)."""
    rows = []
    for csv_path in sorted(Path(eval_logs_root).glob("*/*/*/simulation_log.csv")):
        condition_dir = csv_path.parent.name
        run_tag = csv_path.parent.parent.parent.name
        parsed = _parse_condition_dir(condition_dir)
        if parsed is None:
            print(f"[aggregate_offline] WARNING: could not parse condition dir {condition_dir!r}, skipping")
            continue
        kind, task, trial = parsed
        df = pd.read_csv(csv_path)
        try:
            strikes = extract_strikes(df, bpm=None, target_ref=20.0, tol_ms=30.0, min_strike_frac=0.25)
        except Exception as e:
            print(f"[aggregate_offline] WARNING: extract_strikes failed for {csv_path}: {e!r}")
            continue
        strikes = strikes.assign(run_tag=run_tag, kind=kind, task=task, eval_trial=trial)
        rows.append(strikes)

    if not rows:
        return pd.DataFrame(columns=["strike_idx", "target_time", "peak_time", "timing_err_ms",
                                      "peak_force", "success", "run_tag", "kind", "task", "eval_trial"])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# (A) tau-sweep: optimal lookahead per tau
# ---------------------------------------------------------------------------

def tau_optimal_lookahead(eval_logs_root: Path = REPO_ROOT / "eval_logs_tau_sweep") -> pd.DataFrame | None:
    if not eval_logs_root.is_dir():
        print(f"[aggregate_offline] (A) {eval_logs_root} does not exist yet - skipping tau-eval aggregation.")
        return None

    strikes = extract_strikes_tree(eval_logs_root)
    if strikes.empty:
        print(f"[aggregate_offline] (A) no simulation_log.csv under {eval_logs_root} yet - skipping.")
        return None

    tau_lh_seed = strikes["run_tag"].str.extract(TAU_RUN_TAG_RE)
    tau_lh_seed.columns = ["tau", "lh", "seed"]
    strikes = pd.concat([strikes, tau_lh_seed], axis=1)
    unparsed = strikes["tau"].isna().sum()
    if unparsed:
        print(f"[aggregate_offline] (A) WARNING: {unparsed} strike rows had an unparseable run_tag "
              f"(not matching {{timestamp}}_tau{{X}}_lh{{Y}}_seed{{Z}}) - dropped.")
    strikes = strikes.dropna(subset=["tau", "lh", "seed"]).copy()
    strikes["tau"] = strikes["tau"].astype(float)
    strikes["lh"] = strikes["lh"].astype(float)
    strikes["seed"] = strikes["seed"].astype(int)

    # seed_level()/across_seed() expect a "model" column - use the (tau,lh)
    # cell label as the "model" so every existing stats.py helper applies unchanged.
    strikes["model"] = strikes.apply(lambda r: f"tau{r['tau']}_lh{r['lh']}", axis=1)

    sl = seed_level(strikes)
    across = across_seed(sl, value_col="success_rate")
    across[["tau", "lh"]] = across["model"].str.extract(r"^tau([\d.]+)_lh([\d.]+)$").astype(float)

    # Average across the 3 conditions (task) per cell to get one "overall"
    # success rate per (tau, lh), then argmax lh within each tau.
    per_cell = across.groupby(["tau", "lh"], as_index=False)["mean"].mean().rename(columns={"mean": "success_rate"})
    optimal = per_cell.loc[per_cell.groupby("tau")["success_rate"].idxmax()].sort_values("tau")
    optimal = optimal.rename(columns={"lh": "optimal_lookahead_horizon"})

    per_cell.to_csv(OUTPUTS_DIR / "tau_lookahead_success_rate.csv", index=False)
    optimal.to_csv(OUTPUTS_DIR / "tau_optimal_lookahead.csv", index=False)
    print(f"[aggregate_offline] (A) wrote tau_lookahead_success_rate.csv ({len(per_cell)} cells) "
          f"and tau_optimal_lookahead.csv ({len(optimal)} tau values)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(optimal["tau"], optimal["optimal_lookahead_horizon"], marker="o")
        ax.set_xlabel("PAM time-constant scale (tau)")
        ax.set_ylabel("Optimal lookahead_horizon (s)")
        ax.set_title("Optimal lookahead vs. tau (averaged over 3 eval conditions)")
        ax.set_xscale("log", base=2)
        fig.tight_layout()
        fig_path = OUTPUTS_DIR / "fig_tau_vs_optimal_lookahead.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"[aggregate_offline] (A) wrote {fig_path}")
    except Exception as e:
        print(f"[aggregate_offline] (A) WARNING: figure generation failed: {e!r}")

    return optimal


# ---------------------------------------------------------------------------
# Memory-group (B/C/E) vs no-memory-group (A/D) comparison
# ---------------------------------------------------------------------------

MEMORY_MODELS = ["B", "C", "E"]
NO_MEMORY_MODELS = ["A", "D"]


def memory_group_comparison(
    eval_summary_csv: Path = OUTPUTS_DIR / "eval_summary.csv",
) -> pd.DataFrame | None:
    if not eval_summary_csv.exists():
        print(f"[aggregate_offline] {eval_summary_csv} not found - run the main A-E Layer2a "
              "aggregation first (or point --eval_summary_csv elsewhere). Skipping memory-group comparison.")
        return None

    summary = pd.read_csv(eval_summary_csv)
    summary["group"] = summary["model"].map(
        lambda m: "memory(B/C/E)" if m in MEMORY_MODELS else ("no_memory(A/D)" if m in NO_MEMORY_MODELS else "other")
    )
    summary = summary[summary["group"] != "other"]

    out = summary[["task", "model", "group", "n_seeds", "mean", "ci_lo", "ci_hi", "seed_std"]].sort_values(
        ["task", "group", "model"]
    )
    out_path = OUTPUTS_DIR / "memory_group_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"[aggregate_offline] wrote {out_path} ({len(out)} rows)")
    return out


# ---------------------------------------------------------------------------
# (C) trials>1 eval-noise variance
# ---------------------------------------------------------------------------

def trials_eval_noise_summary(eval_logs_root: Path = REPO_ROOT / "eval_logs_trials5") -> pd.DataFrame | None:
    if not eval_logs_root.is_dir():
        print(f"[aggregate_offline] (C) {eval_logs_root} does not exist yet - skipping eval-noise aggregation.")
        return None

    strikes = extract_strikes_tree(eval_logs_root)
    if strikes.empty:
        print(f"[aggregate_offline] (C) no simulation_log.csv under {eval_logs_root} yet - skipping.")
        return None

    # run_tag -> (model, seed) via the main discover table (these cells reuse
    # the exact same A-E run_dirs as the main matrix).
    from analysis.harness.discover import discover_all_runs

    runs_df = discover_all_runs(REPO_ROOT / "logs" / "rsl_rl")
    runs_df = runs_df[runs_df["status"] == "completed"].copy()
    run_tag_to_model_seed = {Path(r.run_dir).name: (r.model, r.seed) for r in runs_df.itertuples()}

    strikes["model"] = strikes["run_tag"].map(lambda t: run_tag_to_model_seed.get(t, (None, None))[0])
    strikes["seed"] = strikes["run_tag"].map(lambda t: run_tag_to_model_seed.get(t, (None, None))[1])
    unmatched = strikes["model"].isna().sum()
    if unmatched:
        print(f"[aggregate_offline] (C) WARNING: {unmatched} strike rows had a run_tag not found in "
              "discover_all_runs() - dropped.")
    strikes = strikes.dropna(subset=["model", "seed"])

    sl = seed_level(strikes)
    out_path = OUTPUTS_DIR / "trials5_eval_noise.csv"
    sl.sort_values(["task", "model", "seed"]).to_csv(out_path, index=False)
    print(f"[aggregate_offline] (C) wrote {out_path} ({len(sl)} rows); "
          f"n_trials range: {sl['n_trials'].min()}-{sl['n_trials'].max()} "
          f"(eval_noise_success_std should now be non-zero, unlike the trials=1 main matrix)")
    return sl


# ---------------------------------------------------------------------------
# (D) far-future masking comparison
# ---------------------------------------------------------------------------

def mask_comparison(
    mask_roots: dict[str, Path] | None = None,
    baseline_summary_csv: Path = OUTPUTS_DIR / "eval_summary.csv",
) -> pd.DataFrame | None:
    if mask_roots is None:
        mask_roots = {
            mode: REPO_ROOT / f"eval_logs_mask_{mode}" for mode in ("zero", "noise", "shuffle")
        }

    rows = []
    for mode, root in mask_roots.items():
        if not root.is_dir():
            print(f"[aggregate_offline] (D) {root} does not exist yet - skipping mode={mode}.")
            continue
        strikes = extract_strikes_tree(root)
        if strikes.empty:
            print(f"[aggregate_offline] (D) no simulation_log.csv under {root} yet - skipping mode={mode}.")
            continue
        strikes["model"] = "C"
        from analysis.harness.discover import discover_all_runs
        runs_df = discover_all_runs(REPO_ROOT / "logs" / "rsl_rl")
        runs_df = runs_df[(runs_df["status"] == "completed") & (runs_df["model"] == "C")]
        run_tag_to_seed = {Path(r.run_dir).name: r.seed for r in runs_df.itertuples()}
        strikes["seed"] = strikes["run_tag"].map(run_tag_to_seed)
        strikes = strikes.dropna(subset=["seed"])

        sl = seed_level(strikes)
        across = across_seed(sl, value_col="success_rate")
        across["mask_mode"] = mode
        rows.append(across)

    if baseline_summary_csv.exists():
        baseline = pd.read_csv(baseline_summary_csv)
        baseline = baseline[baseline["model"] == "C"].copy()
        baseline["mask_mode"] = "none (baseline)"
        rows.append(baseline)
    else:
        print(f"[aggregate_offline] (D) WARNING: baseline {baseline_summary_csv} not found - "
              "comparison table will have no unmasked reference row.")

    if not rows:
        print("[aggregate_offline] (D) nothing to compare yet - skipping.")
        return None

    out = pd.concat(rows, ignore_index=True)
    out = out[["mask_mode", "task", "model", "n_seeds", "mean", "ci_lo", "ci_hi", "seed_std"]].sort_values(
        ["task", "mask_mode"]
    )
    out_path = OUTPUTS_DIR / "mask_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"[aggregate_offline] (D) wrote {out_path} ({len(out)} rows)")
    return out


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== (A) tau-sweep: optimal lookahead per tau ===")
    tau_optimal_lookahead()
    print("\n=== memory(B/C/E) vs no-memory(A/D) comparison ===")
    memory_group_comparison()
    print("\n=== (C) trials>1 eval-noise variance ===")
    trials_eval_noise_summary()
    print("\n=== (D) far-future masking comparison ===")
    mask_comparison()


if __name__ == "__main__":
    main()

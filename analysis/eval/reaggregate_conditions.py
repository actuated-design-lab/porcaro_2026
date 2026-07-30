"""Read-only re-slicing of already-computed eval data, condition-by-condition.

Two independent re-aggregations, both reusing analysis/eval/aggregate_offline.py's
extract_strikes_tree() / TAU_RUN_TAG_RE and analysis.harness.stats' seed_level()/
across_seed() - no new GPU eval is launched here, this only re-reads
eval_logs_tau_sweep/ and eval_logs_nondr/ (already fully populated - see
their manifest.json, both 100% status=="success") plus the existing DR
baseline in analysis/outputs/eval_strikes_raw.csv.

(1) Tau-sweep, condition-by-condition:
    - per (tau, lh, condition): success_rate mean/seed_std/n_seeds across the
      3 seeds (tau_lookahead_success_rate_by_condition.csv)
    - GMD-only (gmd_03_high_bpm138 + gmd_04_extreme_bpm170) optimal lookahead
      per tau (tau_optimal_lookahead_gmd.csv) - a deliberately different
      slice from aggregate_offline.py's tau_optimal_lookahead() (which
      averages over all 3 conditions including double_160).

(2) DR vs non-DR, model x condition:
    - eval_logs_nondr/ (Model A-E re-evaluated under the non-DR task) vs the
      original DR eval (analysis/outputs/eval_strikes_raw.csv from the
      Monday Layer2a pass), joined on (model, condition)
      (dr_vs_nondr_comparison.csv).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.eval.aggregate_offline import extract_strikes_tree, TAU_RUN_TAG_RE, OUTPUTS_DIR  # noqa: E402
from analysis.harness.stats import seed_level, across_seed  # noqa: E402
from analysis.harness.discover import discover_all_runs  # noqa: E402

GMD_CONDITIONS = ["gmd_03_high_bpm138", "gmd_04_extreme_bpm170"]


# ---------------------------------------------------------------------------
# (1) Tau-sweep, condition-by-condition
# ---------------------------------------------------------------------------

def tau_by_condition(eval_logs_root: Path = REPO_ROOT / "eval_logs_tau_sweep") -> tuple[pd.DataFrame, pd.DataFrame]:
    strikes = extract_strikes_tree(eval_logs_root)
    if strikes.empty:
        raise RuntimeError(f"no simulation_log.csv under {eval_logs_root} - nothing to re-aggregate")

    tau_lh_seed = strikes["run_tag"].str.extract(TAU_RUN_TAG_RE)
    tau_lh_seed.columns = ["tau", "lh", "seed"]
    strikes = pd.concat([strikes, tau_lh_seed], axis=1)
    unparsed = strikes["tau"].isna().sum()
    if unparsed:
        print(f"[reaggregate] WARNING: {unparsed} strike rows had an unparseable run_tag - dropped.")
    strikes = strikes.dropna(subset=["tau", "lh", "seed"]).copy()
    strikes["tau"] = strikes["tau"].astype(float)
    strikes["lh"] = strikes["lh"].astype(float)
    strikes["seed"] = strikes["seed"].astype(int)
    strikes["model"] = strikes.apply(lambda r: f"tau{r['tau']}_lh{r['lh']}", axis=1)

    sl = seed_level(strikes)
    across = across_seed(sl, value_col="success_rate")
    across[["tau", "lh"]] = across["model"].str.extract(r"^tau([\d.]+)_lh([\d.]+)$").astype(float)

    by_condition = across.rename(columns={"task": "condition", "mean": "success_rate"})[
        ["tau", "lh", "condition", "success_rate", "seed_std", "n_seeds"]
    ].sort_values(["condition", "tau", "lh"])

    gmd_only = by_condition[by_condition["condition"].isin(GMD_CONDITIONS)]
    per_cell_gmd = (
        gmd_only.groupby(["tau", "lh"], as_index=False)["success_rate"].mean()
    )
    optimal_gmd = (
        per_cell_gmd.loc[per_cell_gmd.groupby("tau")["success_rate"].idxmax()]
        .sort_values("tau")
        .rename(columns={"lh": "optimal_lh_gmd", "success_rate": "success"})
    )

    return by_condition, optimal_gmd


# ---------------------------------------------------------------------------
# (2) DR vs non-DR, model x condition
# ---------------------------------------------------------------------------

def dr_vs_nondr(
    nondr_eval_logs_root: Path = REPO_ROOT / "eval_logs_nondr",
    dr_strikes_csv: Path = OUTPUTS_DIR / "eval_strikes_raw.csv",
) -> pd.DataFrame:
    if not dr_strikes_csv.exists():
        raise RuntimeError(
            f"{dr_strikes_csv} not found - the original DR eval's strike-level table "
            "(from the Monday Layer2a pass) is required as the DR baseline."
        )

    # run_tag -> (model, seed) via the authoritative A-E discovery table -
    # eval_logs_nondr/ re-evaluated the exact same run_dirs as the main
    # matrix, just under the non-DR task, so run_tag identity is unchanged.
    runs_df = discover_all_runs(REPO_ROOT / "logs" / "rsl_rl")
    runs_df = runs_df[runs_df["status"] == "completed"]
    run_tag_to_model_seed = {Path(r.run_dir).name: (r.model, r.seed) for r in runs_df.itertuples()}

    nondr_strikes = extract_strikes_tree(nondr_eval_logs_root)
    if nondr_strikes.empty:
        raise RuntimeError(f"no simulation_log.csv under {nondr_eval_logs_root} - nothing to re-aggregate")
    nondr_strikes["model"] = nondr_strikes["run_tag"].map(lambda t: run_tag_to_model_seed.get(t, (None, None))[0])
    nondr_strikes["seed"] = nondr_strikes["run_tag"].map(lambda t: run_tag_to_model_seed.get(t, (None, None))[1])
    unmatched = nondr_strikes["model"].isna().sum()
    if unmatched:
        print(f"[reaggregate] WARNING: {unmatched} non-DR strike rows had a run_tag not found in "
              "discover_all_runs() - dropped.")
    nondr_strikes = nondr_strikes.dropna(subset=["model", "seed"])

    nondr_sl = seed_level(nondr_strikes)
    nondr_across = across_seed(nondr_sl, value_col="success_rate").rename(
        columns={"task": "condition", "mean": "success_nonDR", "n_seeds": "n_seeds_nonDR"}
    )[["model", "condition", "success_nonDR", "n_seeds_nonDR"]]

    dr_strikes = pd.read_csv(dr_strikes_csv)
    dr_sl = seed_level(dr_strikes)
    dr_across = across_seed(dr_sl, value_col="success_rate").rename(
        columns={"task": "condition", "mean": "success_DR", "n_seeds": "n_seeds_DR"}
    )[["model", "condition", "success_DR", "n_seeds_DR"]]

    merged = dr_across.merge(nondr_across, on=["model", "condition"], how="outer")
    merged["delta"] = merged["success_nonDR"] - merged["success_DR"]
    merged["n_seeds"] = merged[["n_seeds_DR", "n_seeds_nonDR"]].max(axis=1)
    merged = merged[["model", "condition", "success_DR", "success_nonDR", "delta", "n_seeds"]]

    model_order = {m: i for i, m in enumerate("ABCDE")}
    condition_order = {"double_160": 0, "gmd_03_high_bpm138": 1, "gmd_04_extreme_bpm170": 2}
    merged = merged.sort_values(
        by=["model", "condition"],
        key=lambda col: col.map(model_order) if col.name == "model" else col.map(condition_order),
    )
    return merged


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("(1) Tau-sweep success rate by condition")
    print("=" * 80)
    by_condition, optimal_gmd = tau_by_condition()

    by_condition_path = OUTPUTS_DIR / "tau_lookahead_success_rate_by_condition.csv"
    by_condition.to_csv(by_condition_path, index=False)
    print(f"\nwrote {by_condition_path} ({len(by_condition)} rows)")
    print(by_condition.to_string(index=False))

    optimal_gmd_path = OUTPUTS_DIR / "tau_optimal_lookahead_gmd.csv"
    optimal_gmd.to_csv(optimal_gmd_path, index=False)
    print(f"\nwrote {optimal_gmd_path} ({len(optimal_gmd)} rows)")
    print(optimal_gmd.to_string(index=False))

    print("\n" + "=" * 80)
    print("(2) DR vs non-DR, model x condition")
    print("=" * 80)
    comparison = dr_vs_nondr()
    comparison_path = OUTPUTS_DIR / "dr_vs_nondr_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\nwrote {comparison_path} ({len(comparison)} rows)")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

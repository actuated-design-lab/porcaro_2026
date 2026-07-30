"""Read-only figure generation for the offline experiment-inventory report.

Reads only already-existing aggregation CSVs under analysis/outputs/ (all
produced by prior offline/CPU-only passes - aggregate_offline.py /
reaggregate_conditions.py) plus TensorBoard scalar event files under
logs/rsl_rl/ and logs/rsl_rl_tau_sweep/ (via analysis.harness.tb_curves /
analysis.eval.run_eval_tau_sweep's discover_tau_runs - no isaaclab/torch
import, CPU-only). Launches NO new training/eval jobs. Writes only PNGs
under analysis/summary/figs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.harness.tb_curves import (  # noqa: E402
    MODEL_COLORS,
    MODEL_LABELS,
    SCALAR_TAG,
    load_scalar_series,
)
from analysis.eval.run_eval_tau_sweep import discover_tau_runs  # noqa: E402

OUTPUTS_DIR = REPO_ROOT / "analysis" / "outputs"
FIGS_DIR = REPO_ROOT / "analysis" / "summary" / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# dataviz-skill categorical slots (validated ordering), used for series that
# are NOT one of the A-E models (which keep tb_curves.MODEL_COLORS for
# cross-figure consistency with the pre-existing fig4a/fig_perseed figures).
PALETTE = {
    "blue": "#2a78d6",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
    "magenta": "#e87ba4",
    "orange": "#eb6834",
}

TASK_LABELS = {
    "double_160": "double_160 (simple, in-distribution)",
    "gmd_03_high_bpm138": "GMD-03 (high, 138bpm)",
    "gmd_04_extreme_bpm170": "GMD-04 (extreme, 170bpm)",
}
TASK_ORDER = ["double_160", "gmd_03_high_bpm138", "gmd_04_extreme_bpm170"]
GMD_CONDITIONS = ["gmd_03_high_bpm138", "gmd_04_extreme_bpm170"]


def savefig(fig, name: str) -> Path:
    out = FIGS_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[gen_figures] wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Section B: learning curves
# ---------------------------------------------------------------------------

def copy_fig4a() -> Path | None:
    """Re-embed the existing A-E learning-curve figure (analysis/harness/tb_curves.py
    output) under analysis/summary/figs/ - not regenerated, just copied, since
    it already exists and reflects the current logs/rsl_rl/ state."""
    src = OUTPUTS_DIR / "fig4a_learning_curves.png"
    if not src.exists():
        print(f"[gen_figures] WARNING: {src} not found, skipping copy.")
        return None
    dst = FIGS_DIR / "fig4a_learning_curves.png"
    dst.write_bytes(src.read_bytes())
    print(f"[gen_figures] copied {src} -> {dst}")
    return dst


TAU_COLORS = {0.5: PALETTE["blue"], 1.0: PALETTE["yellow"], 2.0: PALETTE["red"]}


def tau_sweep_learning_curves() -> Path | None:
    """3-panel (one per tau) learning-curve figure for the 27-run tau x lookahead
    x seed grid, mean +/- std across the 3 seeds per (tau, lh) cell."""
    runs_df = discover_tau_runs()
    completed = runs_df[runs_df["status"] == "completed"]
    if completed.empty:
        print("[gen_figures] WARNING: no completed tau-sweep runs found - skipping.")
        return None

    frames = []
    for row in completed.itertuples():
        series = load_scalar_series(row.run_dir, tag=SCALAR_TAG)
        if series.empty:
            continue
        series = series.assign(tau=row.tau, lh=row.lh, seed=row.seed)
        frames.append(series[["tau", "lh", "seed", "step", "value"]])
    if not frames:
        print("[gen_figures] WARNING: no scalar data in any tau-sweep run - skipping.")
        return None
    curves = pd.concat(frames, ignore_index=True)

    taus = sorted(curves["tau"].unique())
    fig, axes = plt.subplots(1, len(taus), figsize=(5 * len(taus), 4.2), dpi=150, sharey=True)
    if len(taus) == 1:
        axes = [axes]

    lh_linestyles = {}
    for ax, tau in zip(axes, taus):
        sub_tau = curves[curves["tau"] == tau]
        lhs = sorted(sub_tau["lh"].unique())
        cmap = plt.get_cmap("viridis")
        for i, lh in enumerate(lhs):
            sub = sub_tau[sub_tau["lh"] == lh]
            seeds = sorted(sub["seed"].unique())
            if not seeds:
                continue
            step_min, step_max = sub["step"].min(), sub["step"].max()
            grid = np.linspace(step_min, step_max, 150)
            interp = []
            for s in seeds:
                g = sub[sub["seed"] == s].sort_values("step")
                interp.append(np.interp(grid, g["step"], g["value"]))
            stacked = np.vstack(interp)
            mean, std = stacked.mean(axis=0), stacked.std(axis=0)
            color = cmap(i / max(1, len(lhs) - 1))
            ax.plot(grid, mean, color=color, linewidth=1.5, label=f"lh={lh:g}s (n={len(seeds)})")
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.set_title(f"tau x{tau:g}")
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(SCALAR_TAG)
    fig.suptitle("Tau-sweep learning curves (27 runs = 3 tau x 3 lookahead x 3 seed), mean +/- std")
    return savefig(fig, "fig_tau_sweep_learning_curves.png")


# ---------------------------------------------------------------------------
# Section C: play/eval bar charts
# ---------------------------------------------------------------------------

def _bar_panel(ax, summary: pd.DataFrame, models: list[str], title: str):
    tasks = [t for t in TASK_ORDER if t in summary["task"].unique()]
    x = np.arange(len(tasks))
    n = len(models)
    width = 0.8 / n
    for i, model in enumerate(models):
        means, lo_err, hi_err = [], [], []
        for t in tasks:
            row = summary[(summary["model"] == model) & (summary["task"] == t)]
            if row.empty:
                means.append(np.nan)
                lo_err.append(0)
                hi_err.append(0)
                continue
            r = row.iloc[0]
            means.append(r["mean"])
            lo_err.append(max(0.0, r["mean"] - r["ci_lo"]))
            hi_err.append(max(0.0, r["ci_hi"] - r["mean"]))
        offset = (i - (n - 1) / 2) * width
        ax.bar(
            x + offset, means, width=width * 0.9,
            yerr=[lo_err, hi_err], capsize=3,
            color=MODEL_COLORS.get(model), label=f"{model}: {MODEL_LABELS.get(model, model).split(': ', 1)[-1]}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=8, rotation=10)
    ax.set_ylabel("Success rate (+/-30ms), 95% bootstrap CI")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")


def eval_success_bars() -> Path | None:
    csv = OUTPUTS_DIR / "eval_summary.csv"
    if not csv.exists():
        print(f"[gen_figures] WARNING: {csv} not found - skipping eval bar charts.")
        return None
    summary = pd.read_csv(csv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    _bar_panel(axes[0], summary, ["A", "B", "C"], "(i) Lookahead axis: A(0.1s) / B(0.5s) / C(1.0s), LSTM")
    _bar_panel(axes[1], summary, ["D", "B", "E"], "(ii) Memory axis: D(MLP) / B(LSTM) / E(framestack MLP), lh=0.5s")
    fig.suptitle("Zero-shot eval success rate (+/-30ms), DR task, trial=1, n=5 seeds/model")
    return savefig(fig, "fig_eval_success_lookahead_memory.png")


def perseed_model_d() -> Path | None:
    csv = OUTPUTS_DIR / "eval_seed_level.csv"
    if not csv.exists():
        print(f"[gen_figures] WARNING: {csv} not found - skipping per-seed D plot.")
        return None
    seed_level = pd.read_csv(csv)
    sub = seed_level[seed_level["model"] == "D"]
    tasks = [t for t in TASK_ORDER if t in sub["task"].unique()]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    seeds = sorted(sub["seed"].unique())
    x = np.arange(len(seeds))
    width = 0.8 / len(tasks)
    cmap = [PALETTE["blue"], PALETTE["yellow"], PALETTE["red"]]
    for i, t in enumerate(tasks):
        vals = [
            sub[(sub["seed"] == s) & (sub["task"] == t)]["success_rate"].iloc[0]
            if not sub[(sub["seed"] == s) & (sub["task"] == t)].empty else np.nan
            for s in seeds
        ]
        offset = (i - (len(tasks) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.9, color=cmap[i % len(cmap)], label=TASK_LABELS.get(t, t))
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed{s}" for s in seeds])
    ax.set_ylabel("Success rate (+/-30ms)")
    ax.set_title("Model D (memoryless MLP, lh=0.5s): per-seed eval success rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.axhline(0.0, color="black", linewidth=0.8)
    return savefig(fig, "fig_perseed_eval_modelD.png")


# ---------------------------------------------------------------------------
# Section D: tau vs optimal lookahead (GMD-only)
# ---------------------------------------------------------------------------

def tau_vs_lookahead_gmd() -> Path | None:
    by_cond_csv = OUTPUTS_DIR / "tau_lookahead_success_rate_by_condition.csv"
    optimal_csv = OUTPUTS_DIR / "tau_optimal_lookahead_gmd.csv"
    if not (by_cond_csv.exists() and optimal_csv.exists()):
        print("[gen_figures] WARNING: tau-sweep GMD CSVs not found - skipping.")
        return None

    by_cond = pd.read_csv(by_cond_csv)
    optimal = pd.read_csv(optimal_csv)

    gmd_only = by_cond[by_cond["condition"].isin(GMD_CONDITIONS)]
    per_cell = gmd_only.groupby(["tau", "lh"], as_index=False)["success_rate"].mean()

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for tau in sorted(per_cell["tau"].unique()):
        sub = per_cell[per_cell["tau"] == tau].sort_values("lh")
        color = TAU_COLORS.get(tau, PALETTE["violet"])
        ax.plot(sub["lh"], sub["success_rate"], marker="o", color=color, linewidth=1.8, label=f"tau x{tau:g}")
        opt_row = optimal[optimal["tau"] == tau]
        if not opt_row.empty:
            ax.scatter(
                opt_row["optimal_lh_gmd"], opt_row["success"],
                s=140, facecolors="none", edgecolors=color, linewidths=2.2, zorder=5,
            )
    ax.set_xlabel("Lookahead horizon (s)")
    ax.set_ylabel("Success rate (+/-30ms), GMD-03 + GMD-04 mean")
    ax.set_title("Optimal lookahead vs. PAM time-constant scale (tau), GMD-only")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return savefig(fig, "fig_tau_vs_optimal_lookahead_gmd.png")


# ---------------------------------------------------------------------------
# Section E: robustness / auxiliary
# ---------------------------------------------------------------------------

def dr_vs_nondr_bars() -> Path | None:
    csv = OUTPUTS_DIR / "dr_vs_nondr_comparison.csv"
    if not csv.exists():
        print(f"[gen_figures] WARNING: {csv} not found - skipping DR vs non-DR figure.")
        return None
    df = pd.read_csv(csv)
    tasks = [t for t in TASK_ORDER if t in df["condition"].unique()]
    models = sorted(df["model"].unique())

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    x = np.arange(len(tasks))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        vals = [
            df[(df["model"] == model) & (df["condition"] == t)]["delta"].iloc[0]
            if not df[(df["model"] == model) & (df["condition"] == t)].empty else np.nan
            for t in tasks
        ]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.9, color=MODEL_COLORS.get(model), label=model)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=8, rotation=10)
    ax.set_ylabel("Delta success rate (non-DR minus DR)")
    ax.set_title("DR vs. non-DR: per-model, per-condition success-rate delta")
    ax.legend(fontsize=8, ncol=5, loc="upper center")
    ax.grid(alpha=0.3, axis="y")
    return savefig(fig, "fig_dr_vs_nondr.png")


def eval_noise_bars() -> Path | None:
    csv = OUTPUTS_DIR / "trials5_eval_noise.csv"
    if not csv.exists():
        print(f"[gen_figures] WARNING: {csv} not found - skipping eval-noise figure.")
        return None
    df = pd.read_csv(csv)

    across = (
        df.groupby(["model", "task"])["success_rate"]
        .agg(seed_std=lambda v: v.std(ddof=1))
        .reset_index()
    )
    within = (
        df.groupby(["model", "task"])["eval_noise_success_std"]
        .mean()
        .reset_index()
        .rename(columns={"eval_noise_success_std": "eval_noise_mean"})
    )
    merged = across.merge(within, on=["model", "task"])

    models = sorted(merged["model"].unique())
    tasks = [t for t in TASK_ORDER if t in merged["task"].unique()]

    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 4.5), dpi=150, sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    x = np.arange(len(models))
    for ax, t in zip(axes, tasks):
        sub = merged[merged["task"] == t].set_index("model").reindex(models)
        ax.bar(x - 0.2, sub["seed_std"], width=0.38, color=PALETTE["blue"], label="across-seed std")
        ax.bar(x + 0.2, sub["eval_noise_mean"], width=0.38, color=PALETTE["orange"], label="within-seed eval-noise std (mean)")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(TASK_LABELS.get(t, t), fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("Std-dev of success rate")
    axes[0].legend(fontsize=8)
    fig.suptitle("Eval noise (R=5 trials/seed) vs. across-seed variance, B/C/D/E on GMD-03/04")
    return savefig(fig, "fig_eval_noise.png")


def mask_comparison_bars() -> Path | None:
    csv = OUTPUTS_DIR / "mask_comparison.csv"
    if not csv.exists():
        print(f"[gen_figures] WARNING: {csv} not found - skipping mask comparison figure.")
        return None
    df = pd.read_csv(csv)
    tasks = [t for t in TASK_ORDER if t in df["task"].unique()]
    mode_order = ["none (baseline)", "zero", "noise", "shuffle"]
    mode_colors = {
        "none (baseline)": "#7a7a76",
        "zero": PALETTE["blue"],
        "noise": PALETTE["red"],
        "shuffle": PALETTE["violet"],
    }

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    x = np.arange(len(tasks))
    width = 0.8 / len(mode_order)
    for i, mode in enumerate(mode_order):
        means, lo_err, hi_err = [], [], []
        for t in tasks:
            row = df[(df["mask_mode"] == mode) & (df["task"] == t)]
            if row.empty:
                means.append(np.nan)
                lo_err.append(0)
                hi_err.append(0)
                continue
            r = row.iloc[0]
            means.append(r["mean"])
            lo_err.append(max(0.0, r["mean"] - r["ci_lo"]))
            hi_err.append(max(0.0, r["ci_hi"] - r["mean"]))
        offset = (i - (len(mode_order) - 1) / 2) * width
        ax.bar(x + offset, means, width=width * 0.9, yerr=[lo_err, hi_err], capsize=3,
               color=mode_colors[mode], label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=8, rotation=10)
    ax.set_ylabel("Success rate (+/-30ms), 95% CI")
    ax.set_title("Far-future (0.5-1.0s) observation masking, Model C, n=5 seeds")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)
    return savefig(fig, "fig_mask_comparison.png")


def main() -> None:
    copy_fig4a()
    tau_sweep_learning_curves()
    eval_success_bars()
    perseed_model_d()
    tau_vs_lookahead_gmd()
    dr_vs_nondr_bars()
    eval_noise_bars()
    mask_comparison_bars()


if __name__ == "__main__":
    main()

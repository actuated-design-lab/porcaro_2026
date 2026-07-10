"""Layer1 learning-curve plotting from TensorBoard event files.

Reads only the "Train/mean_reward" scalar tag (recon note: rsl_rl's
on_policy_runner.py calls `self.writer.add_scalar("Train/mean_reward",
statistics.mean(locs["rewbuffer"]), locs["it"])`) via tensorboard's own
EventAccumulator - NOT tbparse (not installed in this environment) and NOT
torch/isaaclab/isaacsim/omni (never imported here; only the small
protobuf-based tensorboard event reader is used, on CPU).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, no display / GPU needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

from .discover import discover_all_runs  # noqa: E402

SCALAR_TAG = "Train/mean_reward"

# Kept in sync with analysis/harness/identify.py's classify_model() mapping.
MODEL_LABELS: dict[str, str] = {
    "A": "A: LSTM, lookahead=0.1s",
    "B": "B: LSTM, lookahead=0.5s",
    "C": "C: LSTM, lookahead=1.0s",
    "D": "D: MLP, lookahead=0.5s",
    "E": "E: MLP+framestack(k=5), lookahead=0.5s",
}
MODEL_COLORS: dict[str, str] = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
    "D": "#d62728",
    "E": "#9467bd",
}


def _find_event_file(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("events.out.tfevents.*"))
    return candidates[0] if candidates else None


def load_scalar_series(run_dir: str | Path, tag: str = SCALAR_TAG) -> pd.DataFrame:
    """Read one scalar tag from one run's TB event file.

    Returns a DataFrame [step, value, wall_time]; empty (not raising) if the
    event file or the tag is missing, so one bad/incomplete run doesn't abort
    aggregation across the rest.
    """
    run_dir = Path(run_dir)
    event_file = _find_event_file(run_dir)
    if event_file is None:
        return pd.DataFrame(columns=["step", "value", "wall_time"])

    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return pd.DataFrame(columns=["step", "value", "wall_time"])

    events = ea.Scalars(tag)
    return pd.DataFrame(
        {
            "step": [e.step for e in events],
            "value": [e.value for e in events],
            "wall_time": [e.wall_time for e in events],
        }
    )


def collect_curves(runs_df: pd.DataFrame, tag: str = SCALAR_TAG) -> pd.DataFrame:
    """Load `tag` for every completed, classified run in runs_df.

    Returns a long DataFrame [model, seed, step, value].
    """
    frames = []
    completed = runs_df[runs_df["status"] == "completed"]
    for row in completed.itertuples():
        if row.model is None:
            continue
        series = load_scalar_series(row.run_dir, tag=tag)
        if series.empty:
            continue
        series = series.assign(model=row.model, seed=row.seed)
        frames.append(series[["model", "seed", "step", "value"]])

    if not frames:
        return pd.DataFrame(columns=["model", "seed", "step", "value"])
    return pd.concat(frames, ignore_index=True)


def _mean_std_band(curves: pd.DataFrame, model: str, n_grid: int = 200):
    """Interpolate every seed's curve for `model` onto a common step grid.

    Returns (grid, mean, std, n_seeds). Interpolation (rather than requiring
    identical step values across seeds) is used because TB logging cadence
    can differ slightly run-to-run.
    """
    sub = curves[curves["model"] == model]
    seeds = sorted(sub["seed"].unique())
    if not seeds:
        return np.array([]), np.array([]), np.array([]), 0

    step_min, step_max = sub["step"].min(), sub["step"].max()
    grid = np.linspace(step_min, step_max, n_grid)

    interpolated = []
    for s in seeds:
        seed_series = sub[sub["seed"] == s].sort_values("step")
        interpolated.append(np.interp(grid, seed_series["step"], seed_series["value"]))
    stacked = np.vstack(interpolated)  # (n_seeds, n_grid)
    return grid, stacked.mean(axis=0), stacked.std(axis=0), len(seeds)


def plot_learning_curves(curves: pd.DataFrame, out_path: str | Path, tag: str = SCALAR_TAG) -> Path:
    """Plot mean +/- std learning curves, one line+band per model, to out_path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for model in sorted(curves["model"].unique()):
        grid, mean, std, n_seeds = _mean_std_band(curves, model)
        if len(grid) == 0:
            continue
        color = MODEL_COLORS.get(model)
        label = f"{MODEL_LABELS.get(model, model)} (n={n_seeds})"
        ax.plot(grid, mean, color=color, label=label, linewidth=1.5)
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(tag)
    ax.set_title(f"Layer1 learning curves ({tag}, mean +/- std across seeds)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def summarize_final_reward(curves: pd.DataFrame) -> pd.DataFrame:
    """Per-model mean/std/count of each seed's LAST logged Train/mean_reward value."""
    if curves.empty:
        return pd.DataFrame(columns=["model", "mean", "std", "count"])
    last_per_seed = curves.sort_values("step").groupby(["model", "seed"]).tail(1)
    return (
        last_per_seed.groupby("model")["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def seed_final_window_table(curves: pd.DataFrame, model: str, last_n_iters: int = 100) -> pd.DataFrame:
    """Per-seed mean of `value` over the last `last_n_iters` logged iterations for `model`.

    Uses each seed's own max logged step as the window anchor (rather than a
    hardcoded max_iterations), so a seed that logged slightly fewer/more
    points than another still gets a well-defined "last N iterations" window.
    """
    sub = curves[curves["model"] == model]
    rows = []
    for seed, g in sub.groupby("seed"):
        g = g.sort_values("step")
        if g.empty:
            continue
        max_step = g["step"].max()
        window = g[g["step"] >= max_step - last_n_iters + 1]
        rows.append(
            dict(
                model=model,
                seed=seed,
                n_points=len(window),
                step_min=int(window["step"].min()),
                step_max=int(window["step"].max()),
                mean_reward_last_n=float(window["value"].mean()),
                std_reward_last_n=float(window["value"].std()),
            )
        )
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def classify_bimodal(table: pd.DataFrame, threshold: float = 10.0, value_col: str = "mean_reward_last_n") -> dict:
    """Split a seed_final_window_table() result into >= / < threshold groups."""
    high = table[table[value_col] >= threshold]
    low = table[table[value_col] < threshold]
    return {
        "threshold": threshold,
        "n_high": len(high),
        "n_low": len(low),
        "high_seeds": high["seed"].tolist(),
        "low_seeds": low["seed"].tolist(),
    }


def plot_perseed(
    curves: pd.DataFrame,
    model: str,
    tag: str = SCALAR_TAG,
    ax: "plt.Axes | None" = None,
    title: str | None = None,
) -> "plt.Axes":
    """Plot one raw (un-averaged) line per seed for `model` onto `ax` (no mean/std band).

    If `ax` is None, creates its own standalone figure/axes and returns the
    Axes (caller is then responsible for saving/closing that figure - see
    plot_perseed_standalone() for the save-to-file convenience wrapper).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4), dpi=150)

    sub = curves[curves["model"] == model]
    seeds = sorted(sub["seed"].unique())
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(seeds):
        g = sub[sub["seed"] == s].sort_values("step")
        ax.plot(g["step"], g["value"], label=f"seed{s}", color=cmap(i % 10), linewidth=1.2, alpha=0.9)

    ax.set_ylabel(tag)
    ax.set_title(title or f"Model {model}: per-seed {tag} (n={len(seeds)})")
    ax.legend(loc="best", fontsize=8, ncol=min(len(seeds), 5))
    ax.grid(alpha=0.3)
    return ax


def plot_perseed_standalone(curves: pd.DataFrame, model: str, out_path: str | Path, tag: str = SCALAR_TAG) -> Path:
    """Single-model per-seed figure (fig_perseed_A.png style)."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    plot_perseed(curves, model, tag=tag, ax=ax)
    ax.set_xlabel("Iteration")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_perseed_grid(curves: pd.DataFrame, models: list[str], out_path: str | Path, tag: str = SCALAR_TAG) -> Path:
    """Stacked per-seed subplots, one row per model (fig_perseed_ABC.png style)."""
    n = len(models)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.5 * n), dpi=150, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        plot_perseed(curves, model, tag=tag, ax=ax, title=f"Model {model}: {MODEL_LABELS.get(model, model)}")
    axes[-1].set_xlabel("Iteration")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    logs_rsl_rl_root = repo_root / "logs" / "rsl_rl"
    out_path = repo_root / "analysis" / "outputs" / "fig4a_learning_curves.png"

    runs_df = discover_all_runs(logs_rsl_rl_root)
    curves = collect_curves(runs_df)

    if curves.empty:
        print("[tb_curves] No scalar data found across completed runs - nothing to plot.")
    else:
        saved = plot_learning_curves(curves, out_path)
        print(f"[tb_curves] Saved: {saved}")
        print("\nFinal-iteration Train/mean_reward summary (per model, across seeds):")
        print(summarize_final_reward(curves).to_string(index=False))

        # --- per-seed bimodality check (Model A vs. B/C as a stability contrast) ---
        outputs_dir = repo_root / "analysis" / "outputs"

        available_models = set(curves["model"].unique())
        if "A" in available_models:
            saved_a = plot_perseed_standalone(curves, "A", outputs_dir / "fig_perseed_A.png")
            print(f"\n[tb_curves] Saved: {saved_a}")

        abc_models = [m for m in ["A", "B", "C"] if m in available_models]
        if abc_models:
            saved_abc = plot_perseed_grid(curves, abc_models, outputs_dir / "fig_perseed_ABC.png")
            print(f"[tb_curves] Saved: {saved_abc}")

        print("\nPer-seed mean(Train/mean_reward) over each seed's last 100 logged iterations:")
        for model in abc_models:
            table = seed_final_window_table(curves, model, last_n_iters=100)
            print(f"\n-- Model {model} --")
            print(table.to_string(index=False))
            if model == "A":
                bimodal = classify_bimodal(table, threshold=10.0)
                print(
                    f"\n[Model A bimodality check @ threshold=10.0] "
                    f"high(>=10): {bimodal['n_high']} seed(s) {bimodal['high_seeds']}, "
                    f"low(<10): {bimodal['n_low']} seed(s) {bimodal['low_seeds']}"
                )

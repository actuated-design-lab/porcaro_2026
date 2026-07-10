"""Seed-level aggregation and seed-level statistical tests.

Design principle (per task spec): the unit of statistical inference is the
SEED, not the individual strike. Strikes within a (model, seed, task,
eval_trial) are pooled only to compute that cell's summary statistics;
across-seed comparisons (bootstrap CIs, permutation tests) always resample /
permute at the seed level so that within-seed correlation and per-seed
training-run variance are respected.

No isaaclab / torch imports - pandas / numpy / scipy-free (uses numpy RNG only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_STRIKE_COLS = {"model", "seed", "task", "eval_trial", "success", "timing_err_ms"}


def seed_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a strike-level table down to one row per (model, seed, task).

    Input `df` is expected to be the concatenation of extract_strikes() output
    across many runs, each enriched with identifying columns:
    [model, seed, task, eval_trial, strike_idx, target_time, peak_time,
     timing_err_ms, peak_force, success].

    Returns a DataFrame with columns:
      [model, seed, task, success_rate, abs_err_ms_mean,
       eval_noise_success_std, eval_noise_err_std, n_trials]
    where success_rate / abs_err_ms_mean are first averaged over strikes
    within each eval_trial, then averaged over eval_trials (the seed's point
    estimate); eval_noise_*_std is the std-dev across eval_trials (i.e. the
    within-seed evaluation noise, as opposed to across-seed variance which
    across_seed() computes separately).
    """
    missing = REQUIRED_STRIKE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"seed_level: missing columns {missing}")

    working = df.assign(abs_err_ms=df["timing_err_ms"].abs())

    per_trial = (
        working.groupby(["model", "seed", "task", "eval_trial"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            abs_err_ms_mean=("abs_err_ms", "mean"),
            n_strikes=("success", "size"),
        )
    )

    per_seed = (
        per_trial.groupby(["model", "seed", "task"], as_index=False)
        .agg(
            success_rate=("success_rate", "mean"),
            abs_err_ms_mean=("abs_err_ms_mean", "mean"),
            eval_noise_success_std=("success_rate", "std"),
            eval_noise_err_std=("abs_err_ms_mean", "std"),
            n_trials=("eval_trial", "nunique"),
        )
    )
    per_seed[["eval_noise_success_std", "eval_noise_err_std"]] = per_seed[
        ["eval_noise_success_std", "eval_noise_err_std"]
    ].fillna(0.0)

    return per_seed


def across_seed(
    seed_level_df: pd.DataFrame,
    n_boot: int = 10000,
    seed: int = 0,
    value_col: str = "success_rate",
) -> pd.DataFrame:
    """Bootstrap CI over seeds, for each (model, task) group.

    Returns [model, task, n_seeds, mean, ci_lo, ci_hi, seed_std, eval_noise_mean].
    ci_lo == ci_hi (CI width 0) when all seed values in a group are identical,
    including the degenerate n_seeds == 1 case.
    """
    if value_col not in seed_level_df.columns:
        raise ValueError(f"across_seed: value_col '{value_col}' not in columns")

    eval_noise_col = "eval_noise_success_std" if value_col == "success_rate" else "eval_noise_err_std"

    rng = np.random.default_rng(seed)
    rows = []
    for (model, task), g in seed_level_df.groupby(["model", "task"]):
        vals = g[value_col].to_numpy(dtype=float)
        n = len(vals)
        mean_val = float(np.mean(vals)) if n > 0 else float("nan")
        seed_std = float(np.std(vals, ddof=1)) if n > 1 else 0.0

        if n == 0:
            ci_lo = ci_hi = float("nan")
        else:
            boot_idx = rng.integers(0, n, size=(n_boot, n))
            boot_means = vals[boot_idx].mean(axis=1)
            ci_lo, ci_hi = (float(x) for x in np.percentile(boot_means, [2.5, 97.5]))

        eval_noise_mean = float(g[eval_noise_col].mean()) if eval_noise_col in g.columns and n > 0 else float("nan")

        rows.append(
            dict(
                model=model,
                task=task,
                n_seeds=n,
                mean=mean_val,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                seed_std=seed_std,
                eval_noise_mean=eval_noise_mean,
            )
        )

    return pd.DataFrame(rows, columns=["model", "task", "n_seeds", "mean", "ci_lo", "ci_hi", "seed_std", "eval_noise_mean"])


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size: (P(x>y) - P(x<y)), in [-1, 1]."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    diff = x[:, None] - y[None, :]
    gt = int(np.sum(diff > 0))
    lt = int(np.sum(diff < 0))
    return float((gt - lt) / (len(x) * len(y)))


def perm_test(
    seed_level_df: pd.DataFrame,
    task_id: str,
    model_a: str,
    model_b: str,
    n_perm: int = 20000,
    seed: int = 0,
    value_col: str = "success_rate",
) -> dict:
    """Two-sided permutation test on SEED-level means, plus Cliff's delta.

    Strikes are never pooled directly - the test statistic (difference of
    means) and every permutation are computed on one value per seed.
    """
    task_rows = seed_level_df[seed_level_df["task"] == task_id]
    a = task_rows[task_rows["model"] == model_a][value_col].to_numpy(dtype=float)
    b = task_rows[task_rows["model"] == model_b][value_col].to_numpy(dtype=float)

    if len(a) == 0 or len(b) == 0:
        raise ValueError(
            f"perm_test: no seed-level data for task={task_id!r} "
            f"model_a={model_a!r} (n={len(a)}) model_b={model_b!r} (n={len(b)})"
        )

    obs_diff = float(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a, b])
    na = len(a)
    n = len(pooled)

    rng = np.random.default_rng(seed)
    # Vectorized permutation: argsort of random keys gives a random permutation per row.
    random_keys = rng.random((n_perm, n))
    perm_order = np.argsort(random_keys, axis=1)
    perm_pooled = pooled[perm_order]  # (n_perm, n)
    perm_diffs = perm_pooled[:, :na].mean(axis=1) - perm_pooled[:, na:].mean(axis=1)

    p_value = float(np.mean(np.abs(perm_diffs) >= abs(obs_diff) - 1e-12))
    delta = cliffs_delta(a, b)

    return dict(
        task=task_id,
        model_a=model_a,
        model_b=model_b,
        n_a=na,
        n_b=len(b),
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        obs_diff=obs_diff,
        p_value=p_value,
        cliffs_delta=delta,
        n_perm=n_perm,
    )

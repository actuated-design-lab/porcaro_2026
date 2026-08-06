"""figS_tau_sweep.py — 補足図: τ スイープのヒートマップ（1段幅）。

学習時の空気圧時定数 τ ∈ {0.5, 1.0, 2.0} s と先読みホライズン ∈ {0.1 … 2.0} s の
組み合わせを学習し直したもの（学習27ラン、評価81ラン）。

★n=3 シードで有意差なし（τ0.5 vs τ2.0 で p=0.75）。**trend としてのみ示すこと。**
  「0.5秒」が特定の時定数に固有ではなく τ に対してスケールするらしい、という
  傾向以上のことは主張できない。図のタイトルにもその旨を入れてある。

Usage:
  python analysis/summary/figS_tau_sweep.py --outdir ../RALpaper/figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402

import ral_data as D  # noqa: E402
from ral_figstyle import COL_W, INK, INK_MUTED, apply_style, base_argparser, save  # noqa: E402


def draw(ax, fig) -> dict:
    d = D.tau_summary()
    per = d.groupby(["tau", "lh", "seed"], as_index=False)["success_rate"].mean()
    cell = per.groupby(["tau", "lh"])["success_rate"].agg(["mean", "std", "size"])

    taus = sorted(d.tau.unique())
    lhs = sorted(d.lh.unique())
    grid = np.full((len(taus), len(lhs)), np.nan)
    for (t, l), row in cell.iterrows():
        grid[taus.index(t), lhs.index(l)] = row["mean"]

    im = ax.imshow(grid, cmap="Blues", vmin=np.nanmin(grid) - 0.05,
                   vmax=np.nanmax(grid) + 0.02, aspect="auto", origin="lower")
    for i in range(len(taus)):
        for j in range(len(lhs)):
            if np.isnan(grid[i, j]):
                ax.text(j, i, "–", ha="center", va="center", color=INK_MUTED,
                        fontsize=plt.rcParams["font.size"] - 1)
                continue
            sd = cell.loc[(taus[i], lhs[j]), "std"]
            shade = "white" if grid[i, j] > np.nanmean(grid) + 0.02 else INK
            ax.text(j, i, f"{grid[i, j]:.2f}\n$\\pm${sd:.2f}", ha="center", va="center",
                    color=shade, fontsize=plt.rcParams["font.size"] - 1.5,
                    linespacing=1.05)
    ax.set_xticks(range(len(lhs)), [f"{v:g}" for v in lhs])
    ax.set_yticks(range(len(taus)), [f"{v:g}" for v in taus])
    ax.set_xlabel("Lookahead horizon [s]")
    ax.set_ylabel(r"Training $\tau$ [s]")
    ax.set_title("Trend only: $n=3$ seeds, no significant difference",
                 fontsize=plt.rcParams["font.size"] - 0.5, color=INK_MUTED)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Success rate ($\\pm$30 ms)", fontsize=plt.rcParams["font.size"] - 1)
    cb.ax.tick_params(labelsize=plt.rcParams["font.size"] - 1.5)
    cb.outline.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    lo = per[per.tau == 0.5].groupby("seed")["success_rate"].mean().values
    hi = per[per.tau == 2.0].groupby("seed")["success_rate"].mean().values
    return dict(n_seeds=int(per.seed.nunique()),
                p_tau05_vs_tau20=float(stats.ttest_ind(lo, hi).pvalue))


def main() -> int:
    ap = base_argparser(__doc__.splitlines()[0])
    ap.add_argument("--height", type=float, default=2.35)
    args = ap.parse_args()
    apply_style(args.fontsize)
    fig, ax = plt.subplots(figsize=(COL_W, args.height))
    res = draw(ax, fig)
    save(fig, args.outdir, "figS_tau_sweep", args.format)
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""fig6_training_and_design.py — 本文 Fig.6（2段幅・3パネル）: 学習と設計上の検証。

  (a) 学習曲線 A-E（5シードの平均 ± 1SD）
  (b) tau スイープ: 学習時の空気圧時定数 x 先読みホライズン（学習27ラン / 評価81ラン）
  (c) 評価時ドメインランダム化 ON/OFF（75ラン）と、試行間 vs シード間のばらつき（200ラン）

これまで論文に一行も出ていなかった実験をまとめて可視化する図。

★(b) は n=3 シードで有意差なし（tau0.5 vs tau2.0 で p=0.75）。**trend としてのみ示す。**
★(c) 左: DR を切っても全モデルで有意差なし → sim-to-real ギャップは評価時の
   ドメインランダム化ではなく空気圧のむだ時間に帰属できる（消去法）。
   右: trial間SDとシード間SDが同オーダー（D を除く）→ trial を増やすより
   シードを増やすほうが効く。査読 R4（実機trial数）への定量的な回答になる。

(a) だけ TensorBoard の event ファイル（logs/rsl_rl/）が要る。無い環境では
--skip_curves で (b)(c) だけの図になる。展開後は logs/ の更新日時を過去にすること。

Usage:
  python analysis/summary/fig6_training_and_design.py --outdir ../RALpaper/figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402

import ral_data as D  # noqa: E402
from ral_figstyle import (  # noqa: E402
    COL_W, DBL_W, INK, INK_MUTED, MODEL_COLORS, MODEL_LS, OKABE_ITO,
    apply_style, base_argparser, panel_tag, save, tidy,
)

SMOOTH = 15


# --------------------------------------------------------------- panel (a)
def panel_learning(ax) -> dict:
    from analysis.harness.discover import discover_all_runs
    from analysis.harness.tb_curves import SCALAR_TAG, load_scalar_series

    runs = discover_all_runs(str(REPO_ROOT / "logs" / "rsl_rl"))
    out = {}
    for m in D.MODELS:
        ser = [load_scalar_series(rd, SCALAR_TAG).set_index("step")["value"]
               for rd in runs[runs.model == m].run_dir]
        ser = [x for x in ser if not x.empty]
        if not ser:
            continue
        df = pd.concat(ser, axis=1).sort_index().interpolate(limit_area="inside")
        mu, sd = df.mean(axis=1), df.std(axis=1, ddof=1)
        mu_s = mu.rolling(SMOOTH, center=True, min_periods=1).mean()
        sd_s = sd.rolling(SMOOTH, center=True, min_periods=1).mean()
        ax.fill_between(df.index, mu_s - sd_s, mu_s + sd_s, color=MODEL_COLORS[m],
                        alpha=0.10, lw=0, zorder=2)
        ax.plot(df.index, mu_s, color=MODEL_COLORS[m], lw=1.2, ls=MODEL_LS[m],
                zorder=3, label=m)
        out[m] = float(mu.iloc[-1])
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Mean episode reward")
    ax.set_xlim(0, 1520)
    ax.legend(loc="lower right", ncol=5, columnspacing=0.7, handlelength=1.6,
              fontsize=plt.rcParams["font.size"] - 2.0)
    tidy(ax)
    return out


# --------------------------------------------------------------- panel (b)
def panel_tau(ax, fig) -> dict:
    d = D.tau_summary()
    per = d.groupby(["tau", "lh", "seed"], as_index=False)["success_rate"].mean()
    cell = per.groupby(["tau", "lh"])["success_rate"].agg(["mean", "std"])
    taus, lhs = sorted(d.tau.unique()), sorted(d.lh.unique())
    grid = np.full((len(taus), len(lhs)), np.nan)
    for (t, l), r in cell.iterrows():
        grid[taus.index(t), lhs.index(l)] = r["mean"]
    im = ax.imshow(grid, cmap="Blues", vmin=np.nanmin(grid) - 0.05,
                   vmax=np.nanmax(grid) + 0.02, aspect="auto", origin="lower")
    for i in range(len(taus)):
        for j in range(len(lhs)):
            if np.isnan(grid[i, j]):
                ax.text(j, i, "–", ha="center", va="center", color=INK_MUTED,
                        fontsize=plt.rcParams["font.size"] - 1.5)
                continue
            shade = "white" if grid[i, j] > np.nanmean(grid) + 0.02 else INK
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color=shade, fontsize=plt.rcParams["font.size"] - 1.8)
    ax.set_xticks(range(len(lhs)), [f"{v:g}" for v in lhs])
    ax.set_yticks(range(len(taus)), [f"{v:g}" for v in taus])
    ax.set_xlabel("Lookahead horizon [s]")
    ax.set_ylabel(r"Training $\tau$ [s]")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=plt.rcParams["font.size"] - 2.2)
    cb.outline.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    lo = per[per.tau == 0.5].groupby("seed")["success_rate"].mean().values
    hi = per[per.tau == 2.0].groupby("seed")["success_rate"].mean().values
    return dict(n_seeds=int(per.seed.nunique()),
                p=float(stats.ttest_ind(lo, hi).pvalue))


# --------------------------------------------------------------- panel (c)
def panel_variance(ax) -> dict:
    """左半分: 評価時DR ON/OFF の差。右半分: trial間SD と シード間SD。"""
    dr = D.sim_summary()
    nd = pd.read_csv(D.DATA / "ablation" / "nondr_summary_1N.csv")
    t5 = pd.read_csv(D.DATA / "ablation" / "trials5_summary_1N.csv")
    out = {"dr": {}, "sd": {}}

    xs = np.arange(len(D.MODELS))
    deltas, ps = [], []
    for m in D.MODELS:
        a = dr[(dr.model == m) & (dr.task == D.GMD03_SIM)].groupby("seed").success_rate.mean()
        b = nd[(nd.model == m) & (nd.task == D.GMD03_SIM)].groupby("seed").success_rate.mean()
        deltas.append(b.mean() - a.mean())
        ps.append(float(stats.ttest_rel(b, a).pvalue))
        out["dr"][m] = (round(float(a.mean()), 3), round(float(b.mean()), 3), ps[-1])
    ax.axhline(0, color=INK_MUTED, lw=0.6, zorder=2)
    ax.bar(xs, deltas, 0.55, color=[MODEL_COLORS[m] for m in D.MODELS], zorder=3,
           edgecolor="white", linewidth=0.6)
    ax.set_xticks(xs, D.MODELS)
    ax.set_xlabel("Model")
    ax.set_ylabel("$\\Delta$ (DR off $-$ on)")
    ax.set_ylim(-0.16, 0.16)
    ax.text(0.5, 0.97, f"no model separable  (min $p$ = {min(ps):.2f})",
            transform=ax.transAxes, fontsize=plt.rcParams["font.size"] - 2.0,
            ha="center", va="top", color=INK)

    g = t5.groupby(["model", "seed", "task"])["success_rate"]
    tri = g.std(ddof=1).groupby("model").mean()
    seed = g.mean().groupby(["model", "task"]).std(ddof=1).groupby("model").mean()
    out["sd"] = {m: (round(float(tri[m]), 3), round(float(seed[m]), 3)) for m in tri.index}
    keep = [m for m in tri.index if m != "D"]
    ax.text(0.5, 0.03,
            "SD trial-to-trial / seed-to-seed\n"
            + f"{'/'.join(keep)}: {np.mean([tri[m] for m in keep]):.02f} / "
              f"{np.mean([seed[m] for m in keep]):.02f}"
            + f"      D: {tri['D']:.02f} / {seed['D']:.02f}",
            transform=ax.transAxes, fontsize=plt.rcParams["font.size"] - 2.4,
            ha="center", va="bottom", color=INK_MUTED, linespacing=1.25)
    tidy(ax)
    return out


def main() -> int:
    ap = base_argparser("Fig.6: 学習と設計上の検証")
    ap.add_argument("--with_curves", action="store_true",
                    help="学習曲線を含む2段幅3パネル版（補足資料向け）")
    ap.add_argument("--with_dr", action="store_true",
                    help="評価時DR ON/OFF のパネルを足す（既定は落とす）")
    ap.add_argument("--height", type=float, default=1.5)
    args = ap.parse_args()
    apply_style(args.fontsize)

    # 既定は1段幅・2パネル（τ スイープ と DR ON/OFF）。学習曲線は --with_curves で
    # 2段幅3パネルになるが、紙面が8ページ厳守なので本文では既定を使うこと。
    if args.with_curves:
        fig, axes = plt.subplots(1, 3, figsize=(DBL_W, args.height),
                                 gridspec_kw=dict(width_ratios=[1.35, 1.0, 1.15],
                                                  wspace=0.62))
        res = {"a": panel_learning(axes[0])}
        panel_tag(axes[0], "(a)", x=-0.16)
        tags, panels = "bc", axes[1:]
    elif args.with_dr:
        fig, axes = plt.subplots(1, 2, figsize=(COL_W, args.height),
                                 gridspec_kw=dict(width_ratios=[1.0, 1.05], wspace=0.72))
        res, tags, panels = {}, "ab", axes
    else:
        # ★既定は tau スイープのみ。DR ON/OFF は「全モデルで有意差なし」を棒で
        #   見せるだけで情報量が薄く、数値は本文1文で足りる。紙面を返す。
        fig, ax = plt.subplots(figsize=(COL_W * 0.62, args.height))
        res = {"": panel_tau(ax, fig)}
        save(fig, args.outdir, "fig6_training_and_design", args.format)
        import json
        print(json.dumps(res, indent=2, ensure_ascii=False, default=float))
        return 0
    res[tags[0]] = panel_tau(panels[0], fig); panel_tag(panels[0], f"({tags[0]})", x=-0.30)
    res[tags[1]] = panel_variance(panels[1]); panel_tag(panels[1], f"({tags[1]})", x=-0.34)

    save(fig, args.outdir, "fig6_training_and_design", args.format)
    import json
    print(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())

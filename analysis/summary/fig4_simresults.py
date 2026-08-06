"""fig4_simresults.py — 本文 Fig.4（2段幅・4パネル）: リズム別の sim / 実機。

  (a) single 8th, 120 BPM   実機のみ（sim では未評価）
  (b) double, 160 BPM       両ドメイン
  (c) GMD, 138 BPM          両ドメイン
  (d) GMD, 170 BPM          sim のみ（実機では未評価）

★(a)(b) は学習パターンそのもの。rhythm_generator.py の pattern_keys は
  single_4 / single_8 / double / rest で、bpm_options=[60,80,100,120,140,160]。
  カリキュラム Level2 では double が確率0.5でサンプルされる。つまり
  test_single8_bpm120 と test_double_bpm160 は「学習分布内」であり、
  真に未知なのは GMD 抜粋だけ。図でもその区別を帯で示す。

  この差はデータに出ている。sim の「打てない率」は A/D が double で 0.11-0.18 に
  対し GMD では 0.45-0.48。周期パターンは観測の拍位相 (sin, cos) だけで解けるので
  先読みも記憶も要らず、モデル間の差が出ない。

Usage:
  python analysis/summary/fig4_simresults.py --outdir ../RALpaper/figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402

import ral_data as D  # noqa: E402
from ral_figstyle import (  # noqa: E402
    COL_W, DBL_W, DOMAIN_COLORS, DOMAIN_LABEL, INK, INK_MUTED, MODEL_COLORS,
    MODEL_LS, apply_style, base_argparser, panel_tag, save, tidy,
)

SMOOTH_WINDOW = 15   # 学習曲線の移動平均。生のままだと5本重ねて読めない

# sim と実機で共通する条件だけ。(sim側のtask名, 実機側のmidi名, 表示名)
# (sim側task, 実機側midi, 表示名, 学習分布内か)  None = そのドメインでは未評価
CONDITIONS = [
    (None, "test_single8_bpm120.mid", "single 8th, 120 BPM", True),
    ("double_160bpm", "test_double_bpm160.mid", "double, 160 BPM", True),
    ("gmd_03_high_bpm138", "gmd_03_high_bpm138.mid", "GMD, 138 BPM", False),
    ("gmd_04_extreme_bpm170", None, "GMD, 170 BPM", False),
]


# --------------------------------------------------------------- panel (a)
def panel_learning_curves(ax) -> dict:
    from analysis.harness.discover import discover_all_runs
    from analysis.harness.tb_curves import SCALAR_TAG, load_scalar_series

    runs = discover_all_runs(str(REPO_ROOT / "logs" / "rsl_rl"))
    out = {}
    for m in D.MODELS:
        series = []
        for rd in runs[runs.model == m].run_dir:
            s = load_scalar_series(rd, SCALAR_TAG)
            if not s.empty:
                series.append(s.set_index("step")["value"])
        if not series:
            continue
        df = pd.concat(series, axis=1).sort_index().interpolate(limit_area="inside")
        mu, sd = df.mean(axis=1), df.std(axis=1, ddof=1)
        mu_s = mu.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
        sd_s = sd.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
        ax.fill_between(df.index, mu_s - sd_s, mu_s + sd_s,
                        color=MODEL_COLORS[m], alpha=0.10, lw=0, zorder=2)
        # 色だけで5本を分けない。B/C/E はほぼ重なるので線種を二次符号化に使う
        ax.plot(df.index, mu_s, color=MODEL_COLORS[m], lw=1.3, ls=MODEL_LS[m],
                zorder=3, label=m)
        out[m] = dict(n_seeds=int(df.shape[1]), final_mean=float(mu.iloc[-1]),
                      final_sd=float(sd.iloc[-1]))
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Mean episode reward")
    ax.set_xlim(0, 1520)
    ax.legend(loc="lower right", ncol=5, columnspacing=0.9, handlelength=1.9,
              fontsize=plt.rcParams["font.size"] - 1.2)
    tidy(ax)
    return out


# ----------------------------------------------------------- panel (b)/(c)
def panel_condition(ax, sim_task, hw_task, title, in_dist) -> dict:
    """1つのリズムについて、5モデルを sim と実機で並べる。

    片方のドメインでしか評価していない条件は、その側の棒を描かない。
    並べられないものを並べたように見せない。
    """
    sim, hw = D.sim_summary(), D.hw_summary()
    xs = np.arange(len(D.MODELS))
    w = 0.36
    out = {}
    for k, dom in enumerate(["sim", "real"]):
        task = sim_task if dom == "sim" else hw_task
        if task is None:
            continue
        mu, sd, vals = [], [], []
        for m in D.MODELS:
            v = (sim[(sim.model == m) & (sim.task == task)] if dom == "sim"
                 else hw[(hw.model == m) & (hw.midi == task)])
            v = v.groupby("seed")["success_rate"].mean().values
            mu.append(v.mean()); sd.append(v.std(ddof=1)); vals.append(v)
        pos = xs + (k - 0.5) * w
        ax.bar(pos, mu, w * 0.9, color=DOMAIN_COLORS[dom], zorder=3,
               edgecolor="white", linewidth=0.7, label=DOMAIN_LABEL[dom])
        ax.errorbar(pos, mu, yerr=sd, color=INK_MUTED, capsize=1.2, elinewidth=0.6,
                    ls="none", zorder=5)
        out[dom] = {m: (round(a, 3), round(b, 3)) for m, a, b in zip(D.MODELS, mu, sd)}
    if sim_task is None or hw_task is None:
        miss = "simulation" if sim_task is None else "hardware"
        ax.text(0.5, 0.965, f"not run in {miss}", transform=ax.transAxes,
                fontsize=plt.rcParams["font.size"] - 2.4, color=INK_MUTED,
                ha="center", va="top")
    ax.set_xticks(xs, D.MODELS)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_title(title, fontsize=plt.rcParams["font.size"] - 0.5, color=INK, pad=9)
    # 学習分布内かどうかを見出しの下に帯で示す
    ax.text(0.5, 1.015, "training pattern" if in_dist else "unseen (GMD)",
            transform=ax.transAxes, fontsize=plt.rcParams["font.size"] - 2.6,
            color=INK_MUTED, ha="center", va="bottom")
    tidy(ax)
    return out


# ------------------------------------------------------------------- main
def main() -> int:
    ap = base_argparser("Fig.4: リズム別の sim / 実機")
    ap.add_argument("--height", type=float, default=1.85, help="図の高さ [in]")
    args = ap.parse_args()
    apply_style(args.fontsize)

    fig, axes = plt.subplots(1, 4, figsize=(DBL_W, args.height),
                             gridspec_kw=dict(wspace=0.30))
    res = {}
    for tag, ax, (st, ht, title, indist) in zip("abcd", axes, CONDITIONS):
        res[tag] = panel_condition(ax, st, ht, title, indist)
        panel_tag(ax, f"({tag})", x=-0.30, y=1.30)
    axes[0].set_ylabel("Success rate  ($\\pm$30 ms)")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.supxlabel("Model", fontsize=plt.rcParams["font.size"], y=-0.04)
    axes[2].legend(loc="upper right", fontsize=plt.rcParams["font.size"] - 2.0,
                   handlelength=1.2, ncol=1, labelspacing=0.25)

    save(fig, args.outdir, "fig4_simresults", args.format)
    import json
    print(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())

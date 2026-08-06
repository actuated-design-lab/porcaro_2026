"""fig5_quanteval.py — 本文 Fig.5（§IV-C2、1段幅・4パネル縦積み）。

  (a) 実機の主結果: 5モデル × sim/実機 の打撃成功率
  (b) 転移に成功した3方策（B/C/E）のタイミング誤差分布
  (c) Model D のシード別 sim/実機（転移の「信頼性」）
  (d) Model C の 0.5-1.0 s プレビューを壊すマスクアブレーション

全パネルで **青 = シミュレーション / 橙 = 実機** に統一してある。
(a) と (c) は棒、(b) はヒストグラム、(d) は棒。

★1段（3.5in幅）に4段積みにしてあるのは紙面の都合。2段に横並びにすると
  誌面が1段ぶん増えて8ページに収まらない。高さは 4.42 in が上限
  （4.7 にすると9ページになる）。

Usage:
  python analysis/summary/fig5_quanteval.py --outdir ../RALpaper/figures
  python analysis/summary/fig5_quanteval.py --outdir out --panel a --format both
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402

import ral_data as D  # noqa: E402
from ral_figstyle import (  # noqa: E402
    BAND, COL_W, DOMAIN_COLORS, DOMAIN_LABEL, INK, INK_MUTED, MASK_COLORS,
    MODEL_COLORS, MODEL_SHORT, apply_style, base_argparser, panel_tag, save, tidy,
)

WORKING = ["B", "C", "E"]   # 転移に成功した3方策


# --------------------------------------------------------------- panel (a)
def panel_overview(ax) -> dict:
    """5モデルを sim と実機で並べる。この図の一枚看板。

    A は実機でほぼ消滅し、D はシード間のばらつきが極端で、B/C/E が残る。
    A の青と橙の落差だけが極端に大きいことが「先読み不足の影響は実機で増幅される」
    という主張そのものになっている。
    """
    sim = D.seed_level(D.sim_summary(), "task", D.MAIN_SIM)
    hw = D.seed_level(D.hw_summary(), "midi", D.MAIN_HW)
    xs = np.arange(len(D.MODELS))
    w = 0.36
    out = {}
    for k, (dom, d) in enumerate([("sim", sim), ("real", hw)]):
        mu = [D.mean_sd(d, m)[0] for m in D.MODELS]
        sd = [D.mean_sd(d, m)[1] for m in D.MODELS]
        pos = xs + (k - 0.5) * w
        ax.bar(pos, mu, w * 0.9, color=DOMAIN_COLORS[dom], zorder=3,
               edgecolor="white", linewidth=0.7, label=DOMAIN_LABEL[dom])
        ax.errorbar(pos, mu, yerr=sd, color=INK_MUTED, capsize=1.5, elinewidth=0.7,
                    ls="none", zorder=5)
        for i, m in enumerate(D.MODELS):
            v = d[d.model == m]["success_rate"].values
            ax.scatter(np.full(len(v), pos[i]) + D.jitter(len(v), 0.07), v, s=4,
                       color=INK, alpha=0.35, linewidths=0, zorder=6)
        out[dom] = {m: (round(a, 3), round(b, 3)) for m, a, b in zip(D.MODELS, mu, sd)}
    ax.set_xticks(xs, [MODEL_SHORT[m] for m in D.MODELS])
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.5, 1.0])
    ax.legend(loc="upper left", fontsize=plt.rcParams["font.size"] - 1.8,
              handlelength=1.3, ncol=2, columnspacing=1.0,
              bbox_to_anchor=(0.0, 1.06))
    tidy(ax)
    return out


# --------------------------------------------------------------- panel (b)
def panel_timing(ax) -> dict:
    """転移に成功した3方策の、打撃が成立した打点のタイミング誤差。

    成功率の絶対値が低い理由がここにある。系統的な遅れ（空圧のむだ時間）が
    +20 ms 前後あり、その上にジッタが乗るので、±30 ms の窓を半分ほど外れる。
    """
    hs = D.struck_only(D.hw_strikes())
    hs = hs[hs.midi == D.GMD03_HW]
    bins = np.arange(-150, 151, 10)
    ax.axvspan(-30, 30, color=BAND, zorder=0)
    ax.axvline(0, color=INK_MUTED, lw=0.6, ls=":", zorder=2)
    out, lines = {}, []
    for m in WORKING:
        e = hs[hs.model == m]["timing_err_ms"].dropna().values
        ax.hist(e[np.abs(e) <= 150], bins=bins, density=True, histtype="step",
                lw=1.3, color=MODEL_COLORS[m], zorder=3, label=m)
        out[m] = dict(n=int(len(e)), mean=float(e.mean()), sd=float(e.std(ddof=1)),
                      within30=float(np.mean(np.abs(e) <= 30)))
        lines.append(f"{m}  {e.mean():+.0f} $\\pm$ {e.std(ddof=1):.0f} ms")
    ax.text(0.985, 0.97, "\n".join(lines), transform=ax.transAxes,
            fontsize=plt.rcParams["font.size"] - 2.2, ha="right", va="top",
            linespacing=1.3)
    ax.text(0.015, 0.03, "shaded: $\\pm$30 ms", transform=ax.transAxes,
            fontsize=plt.rcParams["font.size"] - 2.2, color=INK_MUTED,
            ha="left", va="bottom")
    ax.set_xlim(-150, 150)
    ax.set_xticks([-150, -75, 0, 75, 150])
    ax.set_xlabel("Timing error [ms]   (late $\\rightarrow$)")
    ax.set_ylabel("Density")
    ax.set_yticks([0, 0.01, 0.02])
    ax.legend(loc="upper left", fontsize=plt.rcParams["font.size"] - 1.8,
              handlelength=1.3, ncol=3, columnspacing=0.9)
    tidy(ax, ygrid=False)
    return out


# --------------------------------------------------------------- panel (c)
def panel_memory_seeds(ax) -> dict:
    """Model D のシード別。破綻する2シードが sim と実機で一致する。

    平均±SDで語ると D は「やや悪い」に見えるが、実体は二峰性で、
    機能する3シードは B/E と同等、残り2シードは完全に打てない。
    したがってこれは「性能」ではなく「転移の信頼性」の問題である。
    """
    s = D.sim_summary()
    h = D.hw_summary()
    s = s[(s.model == "D") & (s.task == D.GMD03_SIM)].groupby("seed")["success_rate"].mean()
    h = h[(h.model == "D") & (h.midi == D.GMD03_HW)].groupby("seed")["success_rate"].mean()
    seeds = sorted(set(s.index) & set(h.index))
    xs = np.arange(len(seeds))
    w = 0.36
    ax.bar(xs - w / 2, [s[k] for k in seeds], w, color=DOMAIN_COLORS["sim"],
           zorder=3, edgecolor="white", linewidth=0.7)
    ax.bar(xs + w / 2, [h[k] for k in seeds], w, color=DOMAIN_COLORS["real"],
           zorder=3, edgecolor="white", linewidth=0.7)
    for i, k in enumerate(seeds):
        if s[k] < 0.005 and h[k] < 0.005:
            ax.annotate("both 0.000", xy=(i, 0.02),
                        fontsize=plt.rcParams["font.size"] - 2.4, ha="center",
                        va="bottom", color=INK, rotation=90)
    bref = D.seed_level(D.hw_summary(), "midi", D.MAIN_HW)
    bref = bref[bref.model == "B"]["success_rate"].mean()
    ax.axhline(bref, color=INK_MUTED, lw=0.8, ls=":", zorder=2)
    ax.annotate("Model B, hardware", xy=(3.55, bref), xytext=(3.3, 0.80),
                fontsize=plt.rcParams["font.size"] - 2.4, color=INK_MUTED,
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.5))
    ax.set_xticks(xs, [str(int(k)) for k in seeds])
    ax.set_xlabel("Training seed  (Model D)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.5, 1.0])
    tidy(ax)
    return dict(sim={int(k): float(s[k]) for k in seeds},
                real={int(k): float(h[k]) for k in seeds})


# --------------------------------------------------------------- panel (d)
def panel_mask(ax) -> dict:
    """Model C の 0.5-1.0 s プレビューを推論時に壊す。

    zero と shuffle が落ちない = 遠い側は観測にはあるが使われていない。
    noise だけ落ちるが、真のプレビューは83%が厳密にゼロの疎な信号なので、
    密な一様乱数は情報を消すのではなく入力分布そのものを壊している。
    したがって noise はプレビューの利用について何も語らない。
    """
    base = D.sim_summary()
    base = base[base.model == "C"].copy()
    base["cond"] = "none"
    frames = [base]
    for tag in ["zero", "noise", "shuffle"]:
        d = D.mask_summary(tag)
        d["cond"] = tag
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)

    conds = ["none", "zero", "shuffle", "noise"]
    labels = ["none", "zero", "shuffle", "noise $\\dagger$"]
    b0 = d[d.cond == "none"].groupby("seed")["success_rate"].mean()
    out = {}
    for i, c in enumerate(conds):
        v = d[d.cond == c].groupby("seed")["success_rate"].mean()
        ax.bar(i, v.mean(), 0.62, color=MASK_COLORS[c], zorder=3,
               edgecolor="white", linewidth=0.7)
        ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), color=INK, capsize=2,
                    elinewidth=0.8, zorder=5, ls="none")
        ax.scatter(np.full(len(v), i) + D.jitter(len(v), 0.07), v.values, s=5,
                   color=INK, alpha=0.5, linewidths=0, zorder=6)
        if c == "none":
            out[c] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)))
            continue
        aligned = v.reindex(b0.index)
        pv = float(stats.ttest_rel(aligned, b0).pvalue)
        out[c] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                      delta=float((aligned - b0).mean()), p=pv)
        ax.text(i, 0.93, f"{(aligned - b0).mean():+.3f}\n{D.ptext(pv)}", ha="center",
                va="bottom", fontsize=plt.rcParams["font.size"] - 2.6, color=INK,
                linespacing=1.1, zorder=8)
    ax.axhline(b0.mean(), color=MASK_COLORS["none"], lw=0.8, ls=":", zorder=2)
    ax.set_xticks(range(len(conds)), labels)
    ax.set_xlabel("Corrupted 0.5\u20131.0 s preview  (Model C, simulation)", labelpad=1)
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.5, 1.0])
    tidy(ax)
    return out


PANELS = {"a": panel_overview, "b": panel_timing,
          "c": panel_memory_seeds, "d": panel_mask}


def main() -> int:
    ap = base_argparser(__doc__.splitlines()[0])
    ap.add_argument("--panel", default="all", choices=["all", "a", "b", "c", "d"],
                    help="単一パネルだけを別ファイルに書き出す（スライド用）")
    ap.add_argument("--height", type=float, default=4.42,
                    help="4パネル版の高さ [in]。★4.7 にすると本文が9ページになる")
    args = ap.parse_args()
    apply_style(args.fontsize)

    if args.panel != "all":
        fig, ax = plt.subplots(figsize=(COL_W, 1.5))
        res = PANELS[args.panel](ax)
        save(fig, args.outdir, f"fig5_panel_{args.panel}", args.format)
        print(res)
        return 0

    fig, axes = plt.subplots(4, 1, figsize=(COL_W, args.height), layout="constrained")
    fig.get_layout_engine().set(hspace=0.10, h_pad=0.02, w_pad=0.02)
    res = {}
    for tag, ax in zip("abcd", axes):
        res[tag] = PANELS[tag](ax)
        panel_tag(ax, f"({tag})", x=-0.185, y=1.22)
    save(fig, args.outdir, "fig5_quanteval", args.format)

    import json
    print(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())

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


# --------------------------------------------------------------- panel (d)
OUTCOME = [("success", "#009E73"), ("late/early", "#E69F00"), ("no strike", "#BFBFBF")]


def panel_failure(ax) -> dict:
    """打点1つ1つを3つの排他的な結末に分解する（sim / 実機を並べる）。

    成功率という1つの数字だと「なぜ 0.45 なのか」が見えない。分解すると
    A と D の失敗が「遅い」ではなく「打てていない」であること、しかも
    その割合が実機で倍増することが1枚で言える。B/C/E の残余はタイミング。
    ★no-strike rate は §IV-C の3指標のうちの1つで、これまで本文の数字
      （D 0.47 / E 0.14）だけだった。ここで図になる。
    """
    def decomp(df, task_col, tasks, force_col="peak_force"):
        d = df[df[task_col].isin(tasks)].copy()
        struck = d[force_col] >= 1.0
        ok = struck & (d.timing_err_ms.abs() <= 30)
        d["cat"] = np.where(~struck, "no strike",
                            np.where(ok, "success", "late/early"))
        t = d.groupby(["model", "cat"]).size().unstack(1).reindex(columns=[c for c, _ in OUTCOME]).fillna(0)
        return t.div(t.sum(axis=1), axis=0)

    sim = decomp(D.sim_strikes(), "task", D.MAIN_SIM)
    hw = decomp(D.hw_strikes(), "midi", D.MAIN_HW)
    xs = np.arange(len(D.MODELS))
    w = 0.36
    out = {}
    for k, (lab, t) in enumerate([("S", sim), ("H", hw)]):
        pos = xs + (k - 0.5) * w
        bottom = np.zeros(len(D.MODELS))
        for cat, col in OUTCOME:
            v = t.reindex(D.MODELS)[cat].values.astype(float)
            ax.bar(pos, v, w * 0.9, bottom=bottom, color=col, zorder=3,
                   edgecolor="white", linewidth=0.5,
                   label=cat if k == 0 else None)
            bottom += v
        out[lab] = {m: {c: round(float(t.loc[m, c]), 3) for c, _ in OUTCOME}
                    for m in D.MODELS if m in t.index}
        for p_ in pos:
            ax.text(p_, -0.06, lab, ha="center", va="top", color=INK_MUTED,
                    fontsize=plt.rcParams["font.size"] - 3.0)
    ax.set_xticks(xs, D.MODELS)
    ax.tick_params(axis="x", pad=9)
    ax.set_xlabel("Model      (S: simulation,  H: hardware)", labelpad=1)
    ax.set_ylabel("Fraction of onsets")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
              fontsize=plt.rcParams["font.size"] - 2.0, handlelength=1.1,
              columnspacing=1.0, borderpad=0.2)
    tidy(ax, ygrid=False)
    return out


# 本文に載せる並び。★Model D のシード別(panel_memory_seeds)とマスク(panel_mask)は
#   本文の記述だけで足り、図にすると紙面を食うだけなので既定から外してある。
#   --panel seeds / --panel mask で単体書き出しは可能。
MAIN_ORDER = [panel_overview, panel_timing, panel_failure]
PANELS = {"a": panel_overview, "b": panel_timing, "c": panel_failure,
          "seeds": panel_memory_seeds, "mask": panel_mask}


def main() -> int:
    ap = base_argparser(__doc__.splitlines()[0])
    ap.add_argument("--panel", default="all",
                    choices=["all", "a", "b", "c", "seeds", "mask"],
                    help="単一パネルだけを別ファイルに書き出す（スライド用）")
    ap.add_argument("--height", type=float, default=3.55,
                    help="本文版(3パネル)の高さ [in]")
    args = ap.parse_args()
    apply_style(args.fontsize)

    if args.panel != "all":
        fig, ax = plt.subplots(figsize=(COL_W, 1.5))
        res = PANELS[args.panel](ax)
        save(fig, args.outdir, f"fig5_panel_{args.panel}", args.format)
        print(res)
        return 0

    fig, axes = plt.subplots(len(MAIN_ORDER), 1, figsize=(COL_W, args.height),
                             layout="constrained")
    fig.get_layout_engine().set(hspace=0.10, h_pad=0.02, w_pad=0.02)
    res = {}
    for tag, fn, ax in zip("abc", MAIN_ORDER, axes):
        res[tag] = fn(ax)
        panel_tag(ax, f"({tag})", x=-0.185, y=1.22)
    save(fig, args.outdir, "fig5_quanteval", args.format)

    import json
    print(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())

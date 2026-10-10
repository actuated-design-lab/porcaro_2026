"""fig_jfps_replay.py — JFPS 2026秋季 前刷の図：開ループ再生での手首角の誤差（圧力モデル別）

入力: data/user0/jfps2026/replay/compare/summary.md（analysis/eval/compare_replay.py の出力。git 管理下）の
      「wrist：RMSE [deg]」と「圧力 DF：NRMSE [%]」の表
並べるもの: (a) DF 圧力の NRMSE、(b) 手首角の RMSE。
           従来（table）／流量上限（orificeV2）／流量上限＋しきい値（shaped）／実測圧力（measured。角度の上限）
動作: gmd138 の2本は元圧が下がっていたので除く（前刷の方針）

使い方: python analysis/summary/fig_jfps_replay.py
出力:   data/user0/jfps2026/replay/compare/fig_replay_wrist.{pdf,png}（git 管理外。前刷のリポジトリに figs/ としてコピーする）
"""
from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CMP = os.path.join(ROOT, "data", "user0", "jfps2026", "replay", "compare")
MOTIONS = [("tm_A_quasistatic", "Quasi-static"), ("tm_B_steps", "Steps"), ("tm_C_sine", "Sine"),
           ("tm_D_antagonist", "Antagonist"), ("tm_E_dbl160_seed2", "Striking 1"), ("tm_E_dbl160_seed3", "Striking 2")]
MODELS = [("table", "Table model (T)", "#b9b7b0"), ("orificeV2", "Lag + flow limit (O)", "#4a5568"),
          ("shaped", "O + threshold", "#c2410c"), ("measured", "Measured pressure", "#2563eb")]


def read_table(md: str, heading: str) -> dict[str, dict[str, float]]:
    lines = open(md, encoding="utf-8").read().splitlines()
    i = next(k for k, s in enumerate(lines) if s.startswith("## ") and heading in s)
    rows = []
    for s in lines[i + 1:]:
        if s.startswith("## "):
            break
        if s.startswith("|"):
            rows.append(s)
    head = [c.strip() for c in rows[0].strip("|").split("|")]
    out = {}
    for s in rows[2:]:
        c = [x.strip() for x in s.strip("|").split("|")]
        if len(c) != len(head):
            break
        out[c[0]] = {h: float(v) if re.match(r"^[+-]?[\d.]+$", v) else np.nan for h, v in zip(head[1:], c[1:])}
    return out


W = read_table(os.path.join(CMP, "summary.md"), "wrist：RMSE")
P = read_table(os.path.join(CMP, "summary.md"), "圧力 DF：NRMSE")
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, (axp, axw) = plt.subplots(1, 2, figsize=(3.4, 2.3), sharey=True, gridspec_kw=dict(width_ratios=[1, 1.25]))
y = np.arange(len(MOTIONS))[::-1]
for ax, Tab, mods, xl, xmax in ((axp, P, MODELS[:3], "DF pressure NRMSE [%]", 8), (axw, W, MODELS, "Wrist RMSE [deg]", 13)):
    h = 0.8 / len(mods)
    for j, (key, lab, col) in enumerate(mods):
        ax.barh(y + ((len(mods) - 1) / 2 - j) * h, [Tab[m][key] for m, _ in MOTIONS], height=h, color=col, label=lab)
    ax.set_xlabel(xl); ax.set_xlim(0, xmax); ax.grid(axis="x", alpha=0.25, lw=0.5)
axp.set_yticks(y); axp.set_yticklabels([n for _, n in MOTIONS])
axp.set_title("(a) Pressure", loc="left", fontsize=8); axw.set_title("(b) Joint angle", loc="left", fontsize=8)
h_, l_ = axw.get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.55, 1.0), handlelength=1.2, columnspacing=0.8)
fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=0.6)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(CMP, f"fig_replay_wrist.{ext}"), dpi=200)
for m, n in MOTIONS:
    print(f"{n:12s}", "wrist", "  ".join(f"{k}={W[m][k]:.1f}" for k, _, _ in MODELS), "| DF p", "  ".join(f"{k}={P[m][k]:.1f}" for k, _, _ in MODELS[:3]))

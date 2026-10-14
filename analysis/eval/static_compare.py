"""static_compare.py — 機構側の静的なつり合いを実機と sim で比べる（id_static_cocontraction）

jetson_project の signals/id_static_cocontraction.csv（10/14 追加）は、0.05 Hz の三角波（0.05→0.55 MPa、2往復）を
拮抗筋の圧力 0.10 / 0.20 / 0.30 MPa で DF と F に流す。拮抗筋の圧力を変えたときに、手首角−圧力の曲線が
  ・横（角度方向）にどれだけずれるか   → 力の釣り合いの位置（たるみの境目 θ_t、DF と F の力の比）
  ・傾きがどう変わるか                 → PAM の剛性（力マップの長さ方向の傾き）
を、実機と sim で並べる。sim は実測圧力をそのまま入れた再生（圧力モデルの誤差を除く）を想定。

指標（区間 S_<筋>_ant<拮抗筋の圧力>_tri_<k> ごと。掃引する筋の圧力 P に対して）
  center(P) : 上げの枝と下げの枝の角度の平均（ヒステリシスを除いた「つり合いの曲線」）
  width(P)  : 上げの枝 − 下げの枝（ループの縦幅）
  offset    : center の sim − 実機 の平均 [deg]
  slope     : center を P で1次近似した傾き [deg/MPa]（実機と sim、比も出す）
  shift     : 拮抗筋 0.10 → 0.30 MPa で、P = 0.30 MPa での center がどれだけ動くか [deg]

使い方:
  python analysis/eval/static_compare.py --sim data/user0/jfps2026/replay/static/sim_id_static__measured_raw__play_w40_cp15.csv
  （--jetson で jetson_project の場所。既定は porcaro_2026 の隣）
出力（<sim のフォルダ>/summary/）: static_curves.csv, static_metrics.csv, static_summary.md, static_compare.{pdf,png}
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "tools", "user0", "jfps2026"))
import real_log as RL  # noqa: E402

DT = 0.005
MOTION = "id_static_cocontraction"
SEG_RE = re.compile(r"S_(?P<ch>DF|F)_ant(?P<ant>[0-9.]+)_tri_(?P<k>\d)$")
GRID = np.linspace(0.10, 0.50, 41)          # 掃引する筋の圧力 [MPa]（端は折り返しの過渡を避けて外す）


def smooth(x, n=21):
    return np.convolve(x, np.ones(n) / n, "same")


def load_real(jetson, hint=None, motion=MOTION):
    cmd = pd.read_csv(RL.jetson_signal_path(jetson, motion))
    c3 = cmd[["cmd_pressure_DF", "cmd_pressure_F", "cmd_pressure_G"]].values
    A, _ = RL.align_real_log(pd.read_csv(RL.find_real_log(jetson, motion, hint)), c3, DT,
                             columns=RL.PRES_COLS + ("wrist_angle_deg",))
    r = pd.DataFrame(A, columns=["P_DF", "P_F", "P_G", "wrist"])
    seg = pd.read_csv(RL.jetson_signal_path(jetson, motion).replace(".csv", "_annotated.csv")).segment.values
    r["segment"] = seg[np.clip((np.arange(len(r)) * DT / RL.CMD_DT).astype(int), 0, len(seg) - 1)]
    return r


def branches(p, a):
    """上げ・下げの枝を GRID 上に補間して、center と width を返す（範囲外は NaN）"""
    p, a = smooth(p), smooth(a)
    dp = np.gradient(p)
    out = {}
    for name, m in (("up", dp > 0), ("dn", dp < 0)):
        if m.sum() < 10:
            out[name] = np.full(len(GRID), np.nan); continue
        o = np.argsort(p[m]); x, y = p[m][o], a[m][o]
        v = np.interp(GRID, x, y); v[(GRID < x.min()) | (GRID > x.max())] = np.nan
        out[name] = v
    return (out["up"] + out["dn"]) / 2, out["up"] - out["dn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="replay_open_loop.py の出力 csv（id_static_cocontraction）")
    ap.add_argument("--jetson", default=os.environ.get("JETSON_PROJECT", os.path.join(os.path.dirname(ROOT), "jetson_project")))
    ap.add_argument("--real_log", default=None, help="実測ログが複数あるときに選ぶファイル名")
    ap.add_argument("--motion", default=MOTION, help="試験の名前（id_static_cocontraction / id_static_F_taut）")
    a = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(a.sim)), "summary"); os.makedirs(out, exist_ok=True)

    r = load_real(a.jetson, a.real_log, a.motion)
    s = pd.read_csv(a.sim); n = min(len(s), len(r)); r, s = r.iloc[:n], s.iloc[:n]
    rows, curves = [], []
    for seg in pd.unique(r.segment):
        mm = SEG_RE.match(str(seg))
        if not mm:
            continue
        ch, ant, k = mm["ch"], float(mm["ant"]), int(mm["k"])
        ii = np.flatnonzero(r.segment.values == seg)
        for src, p, ang in (("real", r[f"P_{ch}"].values[ii], r.wrist.values[ii]),
                            ("sim", s[f"P_out_{ch}"].values[ii], s.wrist_angle_deg.values[ii])):
            c, w = branches(p, ang)
            curves.append(pd.DataFrame(dict(ch=ch, ant=ant, tri=k, src=src, P=GRID, center=c, width=w)))
    C = pd.concat(curves, ignore_index=True)
    C.to_csv(os.path.join(out, "static_curves.csv"), index=False, float_format="%.3f")

    for (ch, ant, k), g in C.groupby(["ch", "ant", "tri"]):
        R = g[g.src == "real"].set_index("P"); S = g[g.src == "sim"].set_index("P")
        ok = R.center.notna() & S.center.notna()
        if ok.sum() < 5:
            continue
        P = R.index[ok].values
        sr = np.polyfit(P, R.center[ok].values, 1)[0]; ss = np.polyfit(P, S.center[ok].values, 1)[0]
        rows.append(dict(ch=ch, ant=ant, tri=k, offset=float((S.center[ok] - R.center[ok]).mean()),
                         slope_real=sr, slope_sim=ss, slope_ratio=ss / sr if sr else np.nan,
                         width_real=float(R.width[ok].mean()), width_sim=float(S.width[ok].mean()),
                         P_range=f"{P.min():.2f}-{P.max():.2f}"))
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(out, "static_metrics.csv"), index=False, float_format="%.3f")

    # 拮抗筋の圧力によるずれ（P = 0.30 MPa での center、2周目）
    sh = []
    for ch in ("DF", "F"):
        for src in ("real", "sim"):
            g = C[(C.ch == ch) & (C.src == src) & (C.tri == 1) & np.isclose(C.P, 0.30)].set_index("ant").center
            if len(g) >= 2 and g.notna().sum() >= 2:
                lo_, hi_ = min(g.index), max(g.index)
                sh.append(dict(ch=ch, src=src, **{f"center_ant{x:.2f}": g.get(x, np.nan) for x in sorted(g.index)},
                               **{f"shift_{lo_:.2f}_to_{hi_:.2f}": g[hi_] - g[lo_]}))
    SH = pd.DataFrame(sh)

    with open(os.path.join(out, "static_summary.md"), "w", encoding="utf-8") as fo:
        fo.write("# 静的なつり合い：実機と sim（static_compare.py）\n\n")
        fo.write(f"sim: `{os.path.basename(a.sim)}`\n\n## 区間ごとの指標（offset = sim − 実機 [deg]、slope [deg/MPa]、width [deg]）\n\n```\n")
        fo.write(M.round(2).to_string(index=False) + "\n```\n\n## 拮抗筋の圧力によるずれ（掃引する筋 0.30 MPa、2周目）[deg]\n\n```\n")
        fo.write((SH.round(2).to_string(index=False) if len(SH) else "（データ不足）") + "\n```\n")
    print(open(os.path.join(out, "static_summary.md"), encoding="utf-8").read())

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif",
                         "font.serif": ["Liberation Serif", "Times New Roman", "TeX Gyre Termes", "DejaVu Serif"],
                         "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    ants = sorted(C.ant.unique())
    fig, axs = plt.subplots(2, len(ants), figsize=(6.7, 3.6), sharey="row")
    axs = np.atleast_2d(axs)
    for i, ch in enumerate(("DF", "F")):
        for j, ant in enumerate(ants):
            ax = axs[i, j]; seg = f"S_{ch}_ant{ant:.2f}_tri_1"
            ii = np.flatnonzero(r.segment.values == seg)
            if len(ii):
                ax.plot(smooth(r[f"P_{ch}"].values[ii]), smooth(r.wrist.values[ii]), color="#1b2430", lw=1.2, label="Measured")
                ax.plot(smooth(s[f"P_out_{ch}"].values[ii]), smooth(s.wrist_angle_deg.values[ii]), color="#c2410c", lw=1.0, label="Sim")
            other = "F" if ch == "DF" else "DF"
            ax.set_title(f"{ch} swept, {other} = {ant:.2f} MPa", fontsize=8, loc="left")
            ax.set_xlabel(f"{ch} pressure [MPa]"); ax.grid(alpha=0.25, lw=0.5)
        axs[i, 0].set_ylabel("Wrist angle [deg]")
    axs[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"static_compare.{ext}"), dpi=200)


if __name__ == "__main__":
    main()

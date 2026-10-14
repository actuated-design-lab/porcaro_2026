"""static_fit.py — 静的なつり合いだけの簡易モデルで、力の倍率とたるみの境目を実機に当てはめる（Isaac なし）

sim（common/actions/torque.py の Model B）の静的な部分だけを Python で解く：
  τ(θ) = r [ F_DF − F_F ],  F_i = 張っていれば max(s_i·M(P_i, h_i) + f_i, 0)、たるんでいれば 0
  h_DF = r(θ − θt_DF)/L,  h_F = r(θt_F − θ)/L,  張る条件 h_i ≤ h0(P_i)（力マップ・h0 マップは tasks/data/ の csv）
  f_i：ヒステリシスの摩擦 −(c0 + cP·P) g(d) d。掃引する筋は上げ・下げの枝の平均（d = ±1 の平均）、
       保持している筋は id_static_cocontraction の指令から求めた向き（play, w = 40 kPa）
  つり合い：τ(θ) = τg(θ)（重力。sim の中心曲線から a + bθ で推定）
static_compare.py の static_curves.csv（上げ・下げの枝の平均＝中心曲線）を読み、
  1) sim のパラメータで sim の中心曲線を再現できるか（簡易モデルの検証）。同時に重力 τg を推定
  2) 実機の中心曲線に s_DF, s_F, θt_DF, θt_F を当てはめる（重力は sim のまま）
注意：重力と力の倍率は、比しか効かない（一緒に大きくしても同じつり合いになる）ので、重力は sim の値に固定する。

使い方: python analysis/eval/static_fit.py [--curves data/user0/jfps2026/replay/static/summary/static_curves.csv]
出力: <curves のフォルダ>/static_fit.md
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(ROOT, "source/porcaro_2026/porcaro_2026/tasks/data")
R_PULLEY, L0 = 0.014, 0.150                       # user0 cfg/controller_cfg.py の r, L
C0, CP, GC, GE = 0.5, 15.0, 1.5, 1.0              # ヒステリシスの大きさと非対称（controller_cfg.py）
# id_static_cocontraction で、保持している筋の play の向き（指令に play w=40 kPa を通した値）
D_HELD = {("DF", 0.1): 0.0, ("DF", 0.2): 1.0, ("DF", 0.3): 1.0, ("F", 0.1): 0.25, ("F", 0.2): 1.0, ("F", 0.3): 1.0}
TH = np.arange(-20, 75.01, 0.25)

fm = pd.read_csv(os.path.join(DATA, "pam_force_map.csv"), index_col=0)
PAX, HAX = fm.index.values.astype(float), fm.columns.astype(float).values / 100
FINT = RegularGridInterpolator((PAX, HAX), fm.values, bounds_error=False, fill_value=None)
h0m = pd.read_csv(os.path.join(DATA, "pam_force_0_map.csv"), index_col=0)
H0P, H0V = h0m.index.values.astype(float), h0m.iloc[:, 0].values / 100


def fric(P, d):
    return -(C0 + CP * abs(P)) * (GC * (1 + d) / 2 + GE * (1 - d) / 2) * d


def muscle(P, h, s, f):
    hh = np.clip(h, HAX[0], HAX[-1])
    F = np.maximum(FINT(np.c_[np.full(np.size(h), np.clip(P, PAX[0], PAX[-1])), np.ravel(hh)]).reshape(np.shape(h)), 0)
    return np.where(h <= np.interp(P, H0P, H0V), np.maximum(s * F + f, 0), 0.0)


def tau(th_deg, PDF, PF, fDF, fF, sDF, sF, tDF, tF):
    th = np.radians(th_deg)
    hDF = R_PULLEY * (th - np.radians(tDF)) / L0
    hF = R_PULLEY * (np.radians(tF) - th) / L0
    return R_PULLEY * (muscle(PDF, hDF, sDF, fDF) - muscle(PF, hF, sF, fF))


def predict(d, sDF, sF, tDF, tF, a, b):
    out = []
    for ch, ant, P in zip(d.ch, d.ant, d.P):
        fsw = 0.5 * (fric(P, 1) + fric(P, -1)); fh = fric(ant, D_HELD[(ch, round(ant, 2))])
        PDF, PF, fD, fF = (P, ant, fsw, fh) if ch == "DF" else (ant, P, fh, fsw)
        Rr = tau(TH, PDF, PF, fD, fF, sDF, sF, tDF, tF) - (a + b * np.radians(TH))
        i = np.flatnonzero((Rr[:-1] > 0) & (Rr[1:] <= 0))
        if len(i):
            k = i[0]; out.append(TH[k] + (TH[k + 1] - TH[k]) * Rr[k] / (Rr[k] - Rr[k + 1]))
        else:
            out.append(TH[-1] if Rr[-1] > 0 else TH[0])
    return np.array(out)


def table(d, e):
    rows = []
    for ch in ("DF", "F"):
        for ant in (0.1, 0.2, 0.3):
            m = ((d.ch == ch) & np.isclose(d.ant, ant)).values
            if m.any():
                rows.append(dict(ch=ch, ant=ant, rmse=np.sqrt(np.mean(e[m] ** 2)), offset=e[m].mean()))
    return pd.DataFrame(rows).round(2).to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default=os.path.join(ROOT, "data/user0/jfps2026/replay/static/summary/static_curves.csv"))
    a_ = ap.parse_args()
    C = pd.read_csv(a_.curves); C = C[(C.tri == 1) & C.center.notna()]
    sim, real = C[C.src == "sim"], C[C.src == "real"]
    g = least_squares(lambda q: predict(sim, 0.2, 0.2, 7.0, 70.0, q[0], q[1]) - sim.center.values, [0.24, 0.0])
    a, b = g.x; e_sim = g.fun
    e_base = predict(real, 0.2, 0.2, 7.0, 70.0, a, b) - real.center.values
    f = least_squares(lambda p: predict(real, p[0], p[1], p[2], p[3], a, b) - real.center.values, [0.2, 0.2, 7, 70],
                      bounds=([0.05, 0.01, -30, 0], [0.5, 0.5, 40, 140]), diff_step=1e-2)
    e_fit = f.fun
    rm = lambda e: np.sqrt(np.mean(e ** 2))
    txt = [
        "# 静的なつり合いの当てはめ（static_fit.py）", "",
        f"入力: `{os.path.relpath(a_.curves, ROOT)}`（2周目の中心曲線）", "",
        "## 1. 簡易モデルの検証（sim のパラメータで sim を再現）", "",
        f"- 重力トルク τg = {a:.3f} + {b:.3f}·θ[rad] N·m", f"- 再現誤差 RMSE {rm(e_sim):.2f} deg", "",
        "```", table(sim, e_sim), "```", "",
        "## 2. 実機：sim のパラメータのまま", "", f"RMSE {rm(e_base):.2f} deg", "", "```", table(real, e_base), "```", "",
        "## 3. 実機に当てはめ（重力は sim のまま）", "",
        f"- force_scale DF = {f.x[0]:.3f}、F = {f.x[1]:.3f}（今は 0.2, 0.2）",
        f"- theta_t_DF_deg = {f.x[2]:.1f}、theta_t_F_deg = {f.x[3]:.1f}（今は 7, 70）",
        f"- RMSE {rm(e_fit):.2f} deg", "", "```", table(real, e_fit), "```", ""]
    out = os.path.join(os.path.dirname(os.path.abspath(a_.curves)), "static_fit.md")
    open(out, "w", encoding="utf-8").write("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    main()

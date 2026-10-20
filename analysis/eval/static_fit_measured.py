"""static_fit_measured.py — 実測の力マップ（予備の筋、10/20）を「縮み方」か「力」で弱めて、10/14 のロボットの静的なつり合いに合わせる（Isaac なし）

背景：10/20 の PAM 単体試験は予備の筋（新しめ）で、ロボットに付いている筋（使い込んだもの）より縮みやすい。
ロボットの DF は、同じ荷重なら単体試験の約 0.6 倍しか縮まない（PAM単体試験_結果_20261020）。
弱め方を 2 通り比べる（どちらも tasks/data/pam_force_map_measured.csv が元）:
  横（縮み方）: F_k(P, h) = F(P, h / k)、h0_k(P) = k · h0(P)。k < 1 で、同じ圧力・同じ力でも k 倍しか縮まない
  縦（力）    : F_s(P, h) = s · F(P, h)（今までの force_scale と同じ）
静的なつり合いの解き方は static_fit.py と同じ（重力 τg = 0.216 + 0.004 θ N·m は sim の値、θt は 7 / 70°、摩擦 cP 15）。
実機：data/user0/jfps2026/replay/static/summary/static_curves.csv（10/14 の id_static_cocontraction、2 周目の中心曲線）
      と、id_static_F_taut の DF だけの掃引（F = 0 MPa）の中央値（手首角 16.2 / 39.2 / 49.5 / 60.1° @ 0.2〜0.5 MPa）

  python analysis/eval/static_fit_measured.py
出力: data/user0/jfps2026/replay/static/summary/static_fit_measured.md
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(ROOT, "source/porcaro_2026/porcaro_2026/tasks/data")
CURVES = os.path.join(ROOT, "data/user0/jfps2026/replay/static/summary/static_curves.csv")
R, L0 = 0.014, 0.150
C0, CP, GC, GE = 0.5, 15.0, 1.5, 1.0
TAU_G = (0.216, 0.004)                      # static_fit.md の 1（sim の重力）
TT_DF, TT_F = 7.0, 70.0
D_HELD = {("DF", 0.1): 0.0, ("DF", 0.2): 1.0, ("DF", 0.3): 1.0, ("F", 0.1): 0.25, ("F", 0.2): 1.0, ("F", 0.3): 1.0}
DF_ONLY = {0.2: 16.2, 0.3: 39.2, 0.4: 49.5, 0.5: 60.1}   # id_static_F_taut、F = 0 の区間の手首角の中央値 [deg]
TH = np.arange(-20, 110.01, 0.25)

fm = pd.read_csv(os.path.join(DATA, "pam_force_map_measured.csv"), index_col=0)
PAX, HAX = fm.index.values.astype(float), fm.columns.astype(float).values / 100
FINT = RegularGridInterpolator((PAX, HAX), fm.values, bounds_error=False, fill_value=None)
h0m = pd.read_csv(os.path.join(DATA, "pam_force_0_map_measured.csv"), index_col=0)
H0P, H0V = h0m.index.values.astype(float), h0m.iloc[:, 0].values / 100


def fric(P, d):
    return -(C0 + CP * abs(P)) * (GC * (1 + d) / 2 + GE * (1 - d) / 2) * d


def muscle(P, h, k, s, f):
    """横に k、縦に s で弱めた筋の力（張っていなければ 0）。"""
    hk = np.clip(h / k, HAX[0], HAX[-1])
    F = np.maximum(FINT(np.c_[np.full(np.size(h), np.clip(P, PAX[0], PAX[-1])), np.ravel(hk)]).reshape(np.shape(h)), 0)
    return np.where(h <= k * np.interp(P, H0P, H0V), np.maximum(s * F + f, 0), 0.0)


def theta_eq(PDF, PF, fDF, fF, kDF, kF, s):
    th = np.radians(TH)
    hDF, hF = R * (th - np.radians(TT_DF)) / L0, R * (np.radians(TT_F) - th) / L0
    Rr = R * (muscle(PDF, hDF, kDF, s, fDF) - muscle(PF, hF, kF, s, fF)) - (TAU_G[0] + TAU_G[1] * th)
    i = np.flatnonzero((Rr[:-1] > 0) & (Rr[1:] <= 0))
    if len(i):
        j = i[0]
        return TH[j] + (TH[j + 1] - TH[j]) * Rr[j] / (Rr[j] - Rr[j + 1])
    return TH[-1] if Rr[-1] > 0 else TH[0]


def predict(d, kDF, kF, s):
    out = []
    for ch, ant, P in zip(d.ch, d.ant, d.P):
        fsw = 0.5 * (fric(P, 1) + fric(P, -1))
        fh = fric(ant, D_HELD[(ch, round(ant, 2))])
        PDF, PF, fD, fF = (P, ant, fsw, fh) if ch == "DF" else (ant, P, fh, fsw)
        out.append(theta_eq(PDF, PF, fD, fF, kDF, kF, s))
    return np.array(out)


def df_only(kDF, s):
    return {P: theta_eq(P, 0.0, 0.5 * (fric(P, 1) + fric(P, -1)), 0.0, kDF, 1.0, s) for P in DF_ONLY}


def main():
    C = pd.read_csv(CURVES)
    real = C[(C.src == "real") & (C.tri == 1) & C.center.notna()].reset_index(drop=True)
    y = real.center.values
    rm = lambda e: float(np.sqrt(np.mean(e ** 2)))
    cases = []

    def add(name, kDF, kF, s, note=""):
        e = predict(real, kDF, kF, s) - y
        g = real.assign(e=e).groupby(["ch", "ant"]).e.agg(rmse=lambda x: np.sqrt(np.mean(x ** 2)), offset="mean")
        do = df_only(kDF, s)
        cases.append(dict(name=name, kDF=kDF, kF=kF, s=s, rmse=rm(e), note=note, table=g, df_only=do,
                          df_only_rmse=rm(np.array([do[P] - DF_ONLY[P] for P in DF_ONLY]))))

    add("A. 実測マップのまま", 1.0, 1.0, 1.0)
    r = least_squares(lambda p: predict(real, 1.0, 1.0, p[0]) - y, [0.5], bounds=([0.05], [2.0]), diff_step=1e-2)
    add("B. 縦（力）だけ弱める", 1.0, 1.0, float(r.x[0]))
    r = least_squares(lambda p: predict(real, p[0], p[0], 1.0) - y, [0.7], bounds=([0.3], [1.2]), diff_step=1e-2)
    add("C. 横（縮み方）だけ、2 本共通", float(r.x[0]), float(r.x[0]), 1.0)
    r = least_squares(lambda p: predict(real, p[0], p[1], 1.0) - y, [0.7, 0.7], bounds=([0.3, 0.3], [1.2, 1.2]),
                      diff_step=1e-2)
    add("D. 横（縮み方）、DF と F で別", float(r.x[0]), float(r.x[1]), 1.0)
    r = min((least_squares(lambda p: predict(real, p[0], p[1], p[2]) - y, x0, bounds=([0.3, 0.3, 0.05], [1.2, 1.2, 2.0]),
                           diff_step=1e-2) for x0 in ([0.7, 0.7, 1.0], [0.6, 0.6, 0.5])), key=lambda q: q.cost)
    add("E. 横（DF・F 別）＋縦（共通）", float(r.x[0]), float(r.x[1]), float(r.x[2]))
    # 横の k を振ったときの誤差（DR の範囲を考える材料）
    sweep = [(k, rm(predict(real, k, k, 1.0) - y), rm(np.array([v - DF_ONLY[P] for P, v in df_only(k, 1.0).items()])))
             for k in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0)]

    L = ["# 実測マップを弱めて、10/14 のロボットの静的なつり合いに合わせる（static_fit_measured.py）", "",
         f"元のマップ: `tasks/data/pam_force_map_measured.csv`（予備の筋、10/20）。重力 τg = {TAU_G[0]} + {TAU_G[1]}·θ N·m、"
         f"θt = {TT_DF:g} / {TT_F:g}°、摩擦 cP {CP:g}", "",
         "| 当てはめ | k_DF | k_F | 力の倍率 s | 共収縮 RMSE [deg] | DF だけ RMSE [deg] |", "|---|---|---|---|---|---|"]
    for c in cases:
        L.append(f"| {c['name']} | {c['kDF']:.3f} | {c['kF']:.3f} | {c['s']:.3f} | {c['rmse']:.2f} | {c['df_only_rmse']:.2f} |")
    L += ["", "k：横（縮み方）の倍率（1 = 予備の筋のまま）。s：力の倍率（今までの force_scale と同じ）", "",
          "## DF だけの掃引（F = 0 MPa）の手首角 [deg]", "", "| 当てはめ | " + " | ".join(f"{P} MPa" for P in DF_ONLY) + " |",
          "|---|" + "---|" * len(DF_ONLY),
          "| 実機 | " + " | ".join(f"{v:.1f}" for v in DF_ONLY.values()) + " |"]
    for c in cases:
        L.append(f"| {c['name']} | " + " | ".join(f"{c['df_only'][P]:.1f}" for P in DF_ONLY) + " |")
    L += ["", "## 共収縮：区間ごとの誤差（sim − 実機）", ""]
    for c in cases:
        L += [f"### {c['name']}", "", "```", c["table"].round(2).to_string(), "```", ""]
    L += ["## 横の k を 2 本共通で振ったとき", "", "| k | 共収縮 RMSE | DF だけ RMSE |", "|---|---|---|"]
    L += [f"| {k:.2f} | {a:.2f} | {b:.2f} |" for k, a, b in sweep]
    out = os.path.join(os.path.dirname(CURVES), "static_fit_measured.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

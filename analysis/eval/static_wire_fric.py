"""static_wire_fric.py — ワイヤの摩擦（張力に比例）を入れた準静的な経路計算で、10/14 の静的試験に当てはめる（Isaac なし）

static_fit.py は上げ・下げの枝の平均（中心曲線）だけを見ていたので、引っかかり（摩擦で止まる）の効果は打ち消されて見えなかった。
ここでは実測の圧力の時系列をそのまま入れて、手首角の経路を1ステップずつ追い、実機の角度の時系列と直接比べる。

比べる案：A 今の sim／B 倍率だけ／C 倍率＋ワイヤ摩擦／E 倍率＋F の張り始め（実機で触った観察 36°→70° をそのまま使う）／
F E＋PAM の内部摩擦の倍率

モデル（θt は sim のまま DF 7°、F 70° に固定）
  PAM の張力 T_i = 張っていれば max(s_i·M(P_i, h_i) + f_i, 0)、たるんでいれば 0     （static_fit.py と同じ。f_i は PAM の内部摩擦、
                                                                                 向きは play w = 40 kPa で実測圧力から求める）
  ワイヤの摩擦：関節側に伝わる張力は、ワイヤが PAM 側へ引き込まれるとき T(1 − κ)、関節側へ引き出されるとき T(1 + κ)
                 （巻き付け摩擦 T_out = T_in·e^{±μφ} を1次で近似。κ ≈ μφ、φ はプーリ・ガイドへの巻き付き角の合計）
  手首が上がる向き（θ 増）：DF は引き込まれ、F は引き出される
      τ_up(θ) = r[T_DF(1 − κ_DF) − T_F(1 + κ_F)] − τg(θ)
  手首が下がる向き：
      τ_dn(θ) = r[T_DF(1 + κ_DF) − T_F(1 − κ_F)] − τg(θ)
  経路：τ_up(θ) > 0 なら τ_up = 0 になる角度まで上がる。τ_dn(θ) < 0 なら τ_dn = 0 になる角度まで下がる。どちらでもなければ止まったまま
  τg：重力（static_fit.py と同じく sim の再生から推定した a + bθ）

使い方: python analysis/eval/static_wire_fric.py [--jetson ../jetson_project]
出力: data/user0/jfps2026/replay/static/wire_fric/（static_wire_fric.md, static_wire_fric.png, paths.csv）
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import static_compare as SC  # noqa: E402
import static_fit as SF  # noqa: E402

R, L0 = SF.R_PULLEY, SF.L0
TDF, TF = 7.0, 70.0                      # θt [deg]（sim の値。実機を見て決めた値なので固定）
TH = np.arange(-20.0, 75.001, 0.1)       # 角度の格子 [deg]
PG = np.round(np.arange(0.0, 0.7001, 0.0025), 4)
STEP = 8                                  # 200 Hz の実機ログを 25 Hz に間引いて経路を追う
PLAY_W = 0.04

# 力マップを (圧力, 角度) の格子で前計算（倍率 1、摩擦なし）。張っているかどうかのマスクも別に持つ
_hDF = R * np.radians(TH - TDF) / L0
_hF = R * np.radians(TF - TH) / L0


def _table(h):
    Fc = np.empty((len(PG), len(TH))); Mk = np.empty_like(Fc)
    for i, P in enumerate(PG):
        hh = np.clip(h, SF.HAX[0], SF.HAX[-1])
        Fc[i] = np.maximum(SF.FINT(np.c_[np.full(len(h), P), hh]), 0)
        Mk[i] = (h <= np.interp(P, SF.H0P, SF.H0V)).astype(float)
    return Fc, Mk


FC_DF, MK_DF = _table(_hDF)
FC_F, MK_F = _table(_hF)
_FC_F0 = FC_F.copy()


def _row(tab, P):
    x = np.clip(P, PG[0], PG[-1]) / (PG[1] - PG[0]); i = min(int(x), len(PG) - 2); w = x - i
    return tab[i] * (1 - w) + tab[i + 1] * w


def play_dir(P, w=PLAY_W):
    z = P[0]; d = np.empty(len(P))
    for k, p in enumerate(P):
        z = min(max(z, p - w), p + w); d[k] = (p - z) / w
    return d


FRIC_K = 1.0          # PAM の内部摩擦（c0, cP）の倍率
RAMP_F = None         # F のワイヤの張り始め [deg]（None = なし。θt_F で張り切る。torque.py の pam_engage_width_deg = θt_F − これ）


def set_case(fric_k=1.0, ramp_f=None):
    global FRIC_K, RAMP_F, FC_F
    FRIC_K, RAMP_F = fric_k, ramp_f
    FC_F = _FC_F0 * (np.clip((TH - ramp_f) / (TF - ramp_f), 0, 1) if ramp_f is not None else 1.0)


def tension(P, d, s, fc, mk):
    f = FRIC_K * SF.fric(P, d)
    return np.where(_row(mk, P) > 0.5, np.maximum(s * _row(fc, P) + f, 0.0), 0.0)


def run_path(seg, p, ga, gb):
    """seg: dict(PDF, PF, dDF, dF, th0)。p = (sDF, sF, kDF, kF)。戻り値：角度の時系列 [deg]"""
    sDF, sF, kDF, kF = p
    tg = ga + gb * np.radians(TH)
    th = seg["th0"]; out = np.empty(len(seg["PDF"]))
    for k in range(len(out)):
        TDFv = tension(seg["PDF"][k], seg["dDF"][k], sDF, FC_DF, MK_DF)
        TFv = tension(seg["PF"][k], seg["dF"][k], sF, FC_F, MK_F)    # FC_F は set_case で張り始めを掛けたもの
        up = R * (TDFv * (1 - kDF) - TFv * (1 + kF)) - tg
        dn = R * (TDFv * (1 + kDF) - TFv * (1 - kF)) - tg
        u0, d0 = np.interp(th, TH, up), np.interp(th, TH, dn)
        if u0 > 0:
            j = np.flatnonzero((TH > th) & (up <= 0))
            th = TH[j[0]] if len(j) else TH[-1]
        elif d0 < 0:
            j = np.flatnonzero((TH < th) & (dn >= 0))
            th = TH[j[-1]] if len(j) else TH[0]
        out[k] = th
    return out


def load_segments(jetson):
    r = SC.load_real(jetson)
    segs = []
    for name in pd.unique(r.segment):
        mm = SC.SEG_RE.match(str(name))
        if not mm:
            continue
        ii = np.flatnonzero(r.segment.values == name)[::STEP]
        PDF, PF = r.P_DF.values[ii], r.P_F.values[ii]
        segs.append(dict(name=name, ch=mm["ch"], ant=float(mm["ant"]), tri=int(mm["k"]), PDF=PDF, PF=PF,
                         dDF=play_dir(PDF), dF=play_dir(PF), real=r.wrist.values[ii], th0=float(r.wrist.values[ii][0])))
    return segs


def resid(p, segs, ga, gb):
    return np.concatenate([run_path(s, p, ga, gb) - s["real"] for s in segs])


def per_seg(p, segs, ga, gb):
    rows = []
    for s in segs:
        sim = run_path(s, p, ga, gb); e = sim - s["real"]
        sw = s["PDF"] if s["ch"] == "DF" else s["PF"]
        cr, wr = SC.branches(sw, s["real"]); cs, ws = SC.branches(sw, sim)
        ok = ~np.isnan(cr) & ~np.isnan(cs)
        slope_r = np.polyfit(SC.GRID[ok], cr[ok], 1)[0] if ok.sum() > 5 else np.nan
        slope_s = np.polyfit(SC.GRID[ok], cs[ok], 1)[0] if ok.sum() > 5 else np.nan
        rows.append(dict(ch=s["ch"], ant=s["ant"], tri=s["tri"], rmse=np.sqrt(np.mean(e ** 2)),
                         slope_real=slope_r, slope_sim=slope_s, slope_ratio=slope_s / slope_r if slope_r else np.nan,
                         width_real=np.nanmean(wr[ok]), width_sim=np.nanmean(ws[ok]),
                         c30_real=np.interp(0.30, SC.GRID[ok], cr[ok]) if ok.sum() > 5 else np.nan,
                         c30_sim=np.interp(0.30, SC.GRID[ok], cs[ok]) if ok.sum() > 5 else np.nan))
    return pd.DataFrame(rows)


def shift_table(T):
    rows = []
    for ch in ("DF", "F"):
        g = T[(T.ch == ch) & (T.tri == 1)].set_index("ant")
        if {0.1, 0.3} <= set(g.index):
            rows.append(dict(ch=ch, real=g.c30_real[0.3] - g.c30_real[0.1], sim=g.c30_sim[0.3] - g.c30_sim[0.1]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jetson", default=os.environ.get("JETSON_PROJECT", os.path.join(os.path.dirname(ROOT), "jetson_project")))
    ap.add_argument("--curves", default=os.path.join(ROOT, "data/user0/jfps2026/replay/static/summary/static_curves.csv"))
    a_ = ap.parse_args()
    out = os.path.join(ROOT, "data/user0/jfps2026/replay/static/wire_fric"); os.makedirs(out, exist_ok=True)

    # 重力（static_fit.py と同じ：sim の中心曲線を sim のパラメータで再現するように推定）
    C = pd.read_csv(a_.curves); C = C[(C.tri == 1) & C.center.notna()]
    sim = C[C.src == "sim"]
    g = least_squares(lambda q: SF.predict(sim, 0.2, 0.2, TDF, TF, q[0], q[1]) - sim.center.values, [0.24, 0.0])
    ga, gb = g.x

    segs = load_segments(a_.jetson)
    cases = {}   # 名前 → (倍率DF, 倍率F, κ_DF, κ_F, PAM摩擦の倍率, F の張り始め)
    set_case()
    cases["A 今の sim（倍率 0.2/0.2、ワイヤ摩擦なし）"] = (0.2, 0.2, 0.0, 0.0, 1.0, None)
    fB = least_squares(lambda q: resid([q[0], q[1], 0, 0], segs, ga, gb), [0.2, 0.2],
                       bounds=([0.02, 0.005], [1.5, 1.5]), diff_step=1e-2)
    cases["B 倍率だけ当てはめ"] = (*fB.x, 0.0, 0.0, 1.0, None)
    fC = min((least_squares(lambda q: resid(q, segs, ga, gb), x0, bounds=([0.02, 0.005, 0, 0], [1.5, 1.5, 0.9, 0.9]),
                            diff_step=1e-2) for x0 in ([0.2, 0.2, 0.2, 0.2], [0.3, 0.2, 0.1, 0.4])), key=lambda r_: r_.cost)
    cases["C 倍率＋ワイヤ摩擦 κ（筋ごと）"] = (*fC.x, 1.0, None)
    # F のワイヤの張り始め：実機で触った観察（36° で張り始め、70° で張り切る）をそのまま使う（当てはめない）。θt は変えない
    set_case(ramp_f=36.0)
    fE = least_squares(lambda q: resid([q[0], q[1], 0, 0], segs, ga, gb), [0.15, 0.15],
                       bounds=([0.02, 0.005], [1.5, 1.5]), diff_step=1e-2)
    cases["E 倍率＋F の張り始め 36°→70°（観察値）"] = (*fE.x, 0.0, 0.0, 1.0, 36.0)

    def rF(q):
        set_case(fric_k=q[2], ramp_f=36.0); return resid([q[0], q[1], 0, 0], segs, ga, gb)
    fF = least_squares(rF, [0.15, 0.15, 0.5], bounds=([0.02, 0.005, 0.0], [1.5, 1.5, 3.0]), diff_step=1e-2)
    cases["F 倍率＋F の張り始め 36°→70°＋PAM 摩擦の倍率"] = (fF.x[0], fF.x[1], 0.0, 0.0, fF.x[2], 36.0)

    txt = ["# ワイヤの摩擦を入れた準静的な経路計算（static_wire_fric.py）", "",
           f"θt は固定（DF {TDF}°、F {TF}°）。重力 τg = {ga:.3f} + {gb:.3f}·θ[rad] N·m（sim の再生から推定）。",
           "実測の圧力（25 Hz に間引き）を入れて手首角の経路を追い、実機の角度と時系列で比べる（tri 0・1 の両方、各区間の最初は実機の角度から）。", ""]
    allpaths = []
    for name, c in cases.items():
        set_case(fric_k=c[4], ramp_f=c[5]); p = np.array(c[:4])
        e = resid(p, segs, ga, gb); T = per_seg(p, segs, ga, gb); SH = shift_table(T)
        T1 = T[T.tri == 1]
        txt += [f"## {name}", "",
                f"- 倍率 DF {p[0]:.3f}、F {p[1]:.3f}、ワイヤ摩擦 κ_DF {p[2]:.3f}、κ_F {p[3]:.3f}、"
                f"PAM 摩擦 ×{c[4]:.2f}、F の張り始め {'なし' if c[5] is None else f'{c[5]:.0f}° → {TF:.0f}°'}",
                f"- 角度の RMSE（全区間）{np.sqrt(np.mean(e ** 2)):.2f} deg", "", "```",
                T1[["ch", "ant", "rmse", "slope_real", "slope_sim", "slope_ratio", "width_real", "width_sim"]].round(2).to_string(index=False),
                "", "拮抗筋 0.1→0.3 MPa での中心曲線のずれ（掃引する筋 0.30 MPa、2周目）[deg]",
                SH.round(2).to_string(index=False) if len(SH) else "（データ不足）", "```", ""]
        for s in segs:
            allpaths.append(pd.DataFrame(dict(case=name[:1], seg=s["name"], P_DF=s["PDF"], P_F=s["PF"], real=s["real"],
                                              sim=run_path(s, p, ga, gb))))
    open(os.path.join(out, "static_wire_fric.md"), "w", encoding="utf-8").write("\n".join(txt))
    Pth = pd.concat(allpaths); Pth.to_csv(os.path.join(out, "paths.csv"), index=False, float_format="%.4f")
    print("\n".join(txt))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif",
                         "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 3, figsize=(6.7, 4.1), sharey="row")
    col = {"A": "#9aa5b1", "C": "#2563eb", "F": "#c2410c"}
    for i, ch in enumerate(("DF", "F")):
        for j, ant in enumerate((0.1, 0.2, 0.3)):
            ax = axs[i, j]; seg = f"S_{ch}_ant{ant:.2f}_tri_1"
            g = Pth[Pth.seg == seg]
            if len(g) == 0:
                continue
            x = g[g.case == "A"][f"P_{ch}"].values
            ax.plot(x, g[g.case == "A"].real.values, color="#1b2430", lw=1.3, label="Measured")
            for c in ("A", "C", "F"):
                ax.plot(x, g[g.case == c].sim.values, color=col[c], lw=0.9,
                        label={"A": "Sim now", "C": "Scale + wire friction", "F": "Scale + F engagement 36-70 deg + PAM friction"}[c])
            other = "F" if ch == "DF" else "DF"
            ax.set_title(f"{ch} swept, {other} = {ant:.2f} MPa", fontsize=8, loc="left")
            ax.set_xlabel(f"{ch} pressure [MPa]"); ax.grid(alpha=0.25, lw=0.5)
        axs[i, 0].set_ylabel("Wrist angle [deg]")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"static_wire_fric.{ext}"), dpi=200)


if __name__ == "__main__":
    main()

"""verify_ral_numbers.py — 本文 00_main.tex に書かれている数値を、集計済みCSVから
再計算して突き合わせる。合わない項目は FAIL として列挙する。

★2026-08-06 更新。打点判定を argmax 方式から peak_match 方式（実打撃をピーク検出し、
  各目標に最近傍を1対1で割り当てる。IROS の create_fig7.py と同じ）に切り替えたため、
  期待値を全面的に入れ替えた。`claude/RAL_hardware_results_FINAL.md` の数値は
  argmax 時代のもので **すべて古い**。参照しないこと。

判定は sim・実機とも 1N（min_strike_frac=0.05 x target_ref=20N）、許容 +/-30ms。
解析単位は学習シード（n=5）。本文の比較は double_160 と gmd_03 の2条件。

Usage:
  python analysis/summary/verify_ral_numbers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ral_data as D  # noqa: E402

RESULTS: list[tuple[str, str, str, bool]] = []


def chk(name, got, want, tol=0.0015):
    RESULTS.append((name, f"{float(got):.4f}", f"{float(want):.4f}",
                    abs(float(got) - float(want)) <= tol))


def rng(name, got, lo, hi):
    RESULTS.append((name, f"{float(got):.3f}", f"[{lo}, {hi}]", lo <= float(got) <= hi))


def main() -> int:
    sim, hw = D.sim_summary(), D.hw_summary()
    S = D.seed_level(sim, "task", D.MAIN_SIM)
    H = D.seed_level(hw, "midi", D.MAIN_HW)
    Sg = D.seed_level(sim, "task", D.GMD03_SIM)
    Hg = D.seed_level(hw, "midi", D.GMD03_HW)

    # --- §IV-C2 1) 予期 / 2) 記憶 / 3) 再帰 の本文値 ---
    for m, mu, sd in [("A", 0.123, 0.048), ("B", 0.445, 0.118), ("C", 0.453, 0.165),
                      ("D", 0.264, 0.201), ("E", 0.505, 0.054)]:
        a, b = D.mean_sd(H, m)
        chk(f"IV-C2 実機 {m} mean", a, mu)
        chk(f"IV-C2 実機 {m} sd", b, sd)

    d, p = D.ttest(H, "A", "B"); chk("1) A→B 差", d, 0.322); rng("1) p(A,B)", p, 0.0, 0.001)
    d, p = D.ttest(H, "B", "C"); chk("2) B vs C 差", d, 0.008); chk("2) p(B,C)", p, 0.931, 0.002)
    d, p = D.ttest(H, "D", "E"); chk("3) D vs E 差", d, 0.240); chk("3) p(D,E)", p, 0.033, 0.002)
    vd = H[H.model == "D"].success_rate.values; ve = H[H.model == "E"].success_rate.values
    chk("3) Cohen d", (ve.mean() - vd.mean()) / np.sqrt((vd.var(ddof=1) + ve.var(ddof=1)) / 2), 1.63, 0.02)
    d, p = D.ttest(H, "B", "E"); chk("4) E vs B 差", d, 0.060); chk("4) p(E,B)", p, 0.332, 0.003)
    vb = H[H.model == "B"].success_rate.values
    se = np.sqrt(vb.var(ddof=1) / 5 + ve.var(ddof=1) / 5)
    chk("4) 95%CI 下", d - 2.306 * se, -0.074, 0.002)
    chk("4) 95%CI 上", d + 2.306 * se, 0.194, 0.002)

    # --- §IV-C 検出力 ---
    # ★2026-08-07: 本文が「転移した3方策」と書いているので B/C/E でプールする
    #   （旧実装は B/E だけで 0.092 だった）。検出力側の係数も t(0.80, df=8)=0.889 に
    #   修正（旧 1.533 は t(0.90, df=4) で分位も自由度も誤り）。0.092x3.839=0.223 と
    #   0.121x3.195=0.245 は近いので、本文の結論は変わらない。
    vc = H[H.model == "C"]["success_rate"].values
    pooled = np.sqrt(np.mean([vb.var(ddof=1), vc.var(ddof=1), ve.var(ddof=1)]))
    chk("IV-C pooled SD (B,C,E)", pooled, 0.121, 0.002)
    chk("IV-C MDE", (2.306 + 0.889) * pooled * np.sqrt(2 / 5), 0.245, 0.005)

    # --- §IV-C2 打てない率 ---
    st = D.hw_strikes(); st = st[st.midi.isin(D.MAIN_HW)]
    ns = (st.assign(m=st.peak_force < 1.0).groupby(["model", "seed", "midi"])["m"].mean()
          .groupby(["model", "seed"]).mean().unstack(0))
    chk("2) 打てない率 D", ns["D"].mean(), 0.51, 0.006)
    chk("2) 打てない率 E", ns["E"].mean(), 0.20, 0.006)
    chk("2) p(打てない率)", stats.ttest_ind(ns["D"], ns["E"]).pvalue, 0.110, 0.004)

    # --- §IV-C2 D の破綻シード（sim と実機で同一）---
    for lab, df in [("sim", Sg), ("実機", Hg)]:
        v = df[df.model == "D"].sort_values("seed").success_rate.values
        RESULTS.append((f"2) D 破綻シード {lab}", str([i + 1 for i, x in enumerate(v) if x < 0.005]),
                        "[3, 5]", [i + 1 for i, x in enumerate(v) if x < 0.005] == [3, 5]))

    # --- §IV-C2 4) レンジと順位相関 ---
    rng("4) 実機レンジ 下", min(D.mean_sd(H, m)[0] for m in D.MODELS), 0.12, 0.13)
    rng("4) 実機レンジ 上", max(D.mean_sd(H, m)[0] for m in D.MODELS), 0.500, 0.510)
    rng("4) sim レンジ 下", min(D.mean_sd(S, m)[0] for m in D.MODELS), 0.565, 0.575)
    rng("4) sim レンジ 上", max(D.mean_sd(S, m)[0] for m in D.MODELS), 0.782, 0.790)
    rho, prho = stats.spearmanr([D.mean_sd(S, m)[0] for m in D.MODELS],
                                [D.mean_sd(H, m)[0] for m in D.MODELS])
    chk("4) 順位相関 rho", rho, 0.90, 0.01); rng("4) p(rho)", prho, 0.0, 0.05)
    drops = [(D.mean_sd(S, m)[0] - D.mean_sd(H, m)[0]) / D.mean_sd(S, m)[0] for m in D.MODELS]
    chk("4) A の低下率", drops[0], 0.78, 0.006)
    rng("4) 他方策の低下率 下", min(drops[1:]), 0.355, 0.365)
    rng("4) 他方策の低下率 上", max(drops[1:]), 0.575, 0.585)

    # --- マスクアブレーション ---
    b0 = sim[sim.model == "C"].groupby("seed").success_rate.mean()
    for tag, dm, pm in [("zero", -0.032, 0.578), ("shuffle", 0.000, 0.990), ("noise", -0.186, 0.224)]:
        v = D.mask_summary(tag).groupby("seed").success_rate.mean().reindex(b0.index)
        chk(f"マスク {tag} Δ", (v - b0).mean(), dm, 0.002)
        chk(f"マスク {tag} p", stats.ttest_rel(v, b0).pvalue, pm, 0.003)

    # --- §IV-B 妥当性 ---
    HW = Path(__file__).resolve().parents[2] / "paper_data" / "hardware"
    anchors = {}
    for f, lab in [("hw_summary_s4_1N.csv", "S4"), ("summary_s2_1N.csv", "S2"), ("summary_s3_1N.csv", "S3")]:
        d_ = pd.read_csv(HW / f)
        d_["model"] = d_.model_key.str.extract(r"/([A-E])_seed")[0]
        d_["seed"] = d_.model_key.str.extract(r"seed(\d)")[0].astype(int)
        x = d_[d_.midi == D.GMD03_HW]
        anchors[lab] = x[x.model == "B"].groupby("seed").success_rate.mean().mean()
    rng("IV-B アンカー他3セッション 下", min(anchors.values()), 0.445, 0.455)
    rng("IV-B アンカー他3セッション 上", max(anchors.values()), 0.530, 0.540)

    # --- §IV-C2 1) 打点分解（Model A の「打てない率」が実機で倍増）---
    st_a = D.hw_strikes(); st_a = st_a[st_a.midi.isin(D.MAIN_HW)]
    ss_a = D.sim_strikes(); ss_a = ss_a[ss_a.task.isin(D.MAIN_SIM)]
    # ★シード単位の二段平均で取る（本文が宣言している解析単位。プール平均だと
    #   実機 64.9% / sim 29.6% になり、本文の規約と食い違う）。
    def _nostrike(d, tcol, model="A"):
        x = d[d.model == model]
        return float(x.assign(m=x.peak_force < 1.0).groupby(["seed", tcol]).m.mean()
                     .groupby("seed").mean().mean())
    chk("1) A 打てない率 実機", _nostrike(st_a, "midi"), 0.621, 0.003)
    chk("1) A 打てない率 sim", _nostrike(ss_a, "task"), 0.331, 0.003)

    # --- §IV-B フラグ付きランを除外した場合の Model D（二段平均） ---
    import glob, os
    fl = {}
    for f in glob.glob("/home/claude/research/jetson_project/results/RAL/**/deploy_*.csv",
                       recursive=True):
        try:
            c = pd.read_csv(f, usecols=["force_N"])
        except Exception:
            continue
        if len(c) <= 10:
            continue
        fl[os.path.basename(f)] = (float(c.force_N.median()) < -5.0
                                   or float(c.force_N.max() - c.force_N.min()) < 10.0)
    if fl:
        hwf = hw.copy()
        hwf["flag"] = hwf.file.astype(str).str.replace(r"^.*/", "", regex=True).map(fl)
        keep = hwf[(hwf.midi.isin(D.MAIN_HW)) & (hwf.flag != True)]  # noqa: E712
        vD = D.seed_level(keep, "midi", D.MAIN_HW)
        vD = vD[vD.model == "D"].success_rate
        chk("IV-B フラグ除外時の D", vD.mean(), 0.357, 0.002)
        RESULTS.append(("IV-B フラグ除外で残る D のシード数", str(len(vD)), "4", len(vD) == 4))

    # --- §IV-C2 3) ダブルストロークの1打目/2打目 打撃力 ---
    dd = D.hw_strikes(); dd = dd[(dd.midi == D.DOUBLE_HW) & dd.model.isin(["B", "C", "E"])]
    dd = dd[dd.peak_force >= 1.0].copy()
    dd["pos"] = np.where(dd.strike_idx % 2 == 0, "1st", "2nd")
    # ★本文が宣言する解析単位（シード, n=5）で対にする。model-seed の15対だと
    #   27.1/11.3, p=0.00015 になるが、それは宣言した規約と違う単位になる。
    g = dd.groupby(["seed", "pos"]).peak_force.mean().unstack(-1).dropna()
    chk("3) 1打目 打撃力[N]", g["1st"].mean(), 26.9, 0.05)
    chk("3) 2打目 打撃力[N]", g["2nd"].mean(), 11.0, 0.05)
    chk("3) p(1打目,2打目)", stats.ttest_rel(g["2nd"], g["1st"]).pvalue, 0.008, 0.001)

    # --- §IV-D tau スイープ ---
    tp = D.tau_summary().groupby(["tau", "lh", "seed"], as_index=False).success_rate.mean()
    cell = tp.groupby(["tau", "lh"]).success_rate.mean()
    for (t, l), want in [((0.5, 0.5), 0.903), ((2.0, 0.5), 0.545), ((2.0, 1.0), 0.726),
                         ((2.0, 2.0), 0.738), ((0.5, 0.25), 0.847), ((0.5, 0.1), 0.594)]:
        chk(f"IV-D tau={t} lh={l}", cell[(t, l)], want, 0.002)
    lo = tp[(tp.tau == 0.5) & (tp.lh == 0.5)].success_rate.values
    hi = tp[(tp.tau == 2.0) & (tp.lh == 0.5)].success_rate.values
    chk("IV-D p(tau0.5,tau2.0 @lh0.5)", stats.ttest_ind(lo, hi).pvalue, 0.145, 0.003)

    w = max(len(r[0]) for r in RESULTS) + 2
    print(f"{'項目':<{w}} {'算出値':>18} {'本文の記載':>18}   判定")
    print("-" * (w + 44))
    for n, got, want, ok in RESULTS:
        print(f"{n:<{w}} {got:>18} {want:>18}   {'OK' if ok else '** FAIL **'}")
    nf = sum(1 for r in RESULTS if not r[3])
    print("-" * (w + 44))
    print(f"{len(RESULTS)} 項目中 {nf} 件 不一致")
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())

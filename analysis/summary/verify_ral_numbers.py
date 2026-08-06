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
    for m, mu, sd in [("A", 0.123, 0.048), ("B", 0.446, 0.126), ("C", 0.453, 0.165),
                      ("D", 0.292, 0.247), ("E", 0.511, 0.103)]:
        a, b = D.mean_sd(H, m)
        chk(f"IV-C2 実機 {m} mean", a, mu)
        chk(f"IV-C2 実機 {m} sd", b, sd)

    d, p = D.ttest(H, "A", "B"); chk("1) A→B 差", d, 0.323); rng("1) p(A,B)", p, 0.0, 0.001)
    d, p = D.ttest(H, "B", "C"); chk("2) B vs C 差", d, 0.007); chk("2) p(B,C)", p, 0.945, 0.002)
    d, p = D.ttest(H, "D", "E"); chk("3) D vs E 差", d, 0.219); chk("3) p(D,E)", p, 0.104, 0.002)
    vd = H[H.model == "D"].success_rate.values; ve = H[H.model == "E"].success_rate.values
    chk("3) Cohen d", (ve.mean() - vd.mean()) / np.sqrt((vd.var(ddof=1) + ve.var(ddof=1)) / 2), 1.16, 0.02)
    d, p = D.ttest(H, "B", "E"); chk("4) E vs B 差", d, 0.065); chk("4) p(E,B)", p, 0.398, 0.003)
    vb = H[H.model == "B"].success_rate.values
    se = np.sqrt(vb.var(ddof=1) / 5 + ve.var(ddof=1) / 5)
    chk("4) 95%CI 下", d - 2.306 * se, -0.103, 0.002)
    chk("4) 95%CI 上", d + 2.306 * se, 0.233, 0.002)

    # --- §IV-C 検出力 ---
    pooled = np.sqrt((vb.var(ddof=1) + ve.var(ddof=1)) / 2)
    chk("IV-C pooled SD", pooled, 0.115, 0.002)
    chk("IV-C MDE", (2.306 + 1.533) * pooled * np.sqrt(2 / 5), 0.28, 0.005)

    # --- §IV-C2 打てない率 ---
    st = D.hw_strikes(); st = st[st.midi.isin(D.MAIN_HW)]
    ns = st.assign(m=st.peak_force < 1.0).groupby(["model", "seed"])["m"].mean().unstack(0)
    chk("2) 打てない率 D", ns["D"].mean(), 0.47, 0.006)
    chk("2) 打てない率 E", ns["E"].mean(), 0.14, 0.006)
    chk("2) p(打てない率)", stats.ttest_ind(ns["D"], ns["E"]).pvalue, 0.157, 0.004)

    # --- §IV-C2 D の破綻シード（sim と実機で同一）---
    for lab, df in [("sim", Sg), ("実機", Hg)]:
        v = df[df.model == "D"].sort_values("seed").success_rate.values
        RESULTS.append((f"2) D 破綻シード {lab}", str([i + 1 for i, x in enumerate(v) if x < 0.005]),
                        "[3, 5]", [i + 1 for i, x in enumerate(v) if x < 0.005] == [3, 5]))

    # --- §IV-C2 4) レンジと順位相関 ---
    rng("4) 実機レンジ 下", min(D.mean_sd(H, m)[0] for m in D.MODELS), 0.12, 0.13)
    rng("4) 実機レンジ 上", max(D.mean_sd(H, m)[0] for m in D.MODELS), 0.505, 0.515)
    rng("4) sim レンジ 下", min(D.mean_sd(S, m)[0] for m in D.MODELS), 0.565, 0.575)
    rng("4) sim レンジ 上", max(D.mean_sd(S, m)[0] for m in D.MODELS), 0.782, 0.790)
    rho, prho = stats.spearmanr([D.mean_sd(S, m)[0] for m in D.MODELS],
                                [D.mean_sd(H, m)[0] for m in D.MODELS])
    chk("4) 順位相関 rho", rho, 0.90, 0.01); rng("4) p(rho)", prho, 0.0, 0.05)
    drops = [(D.mean_sd(S, m)[0] - D.mean_sd(H, m)[0]) / D.mean_sd(S, m)[0] for m in D.MODELS]
    chk("4) A の低下率", drops[0], 0.78, 0.006)
    rng("4) 他方策の低下率 下", min(drops[1:]), 0.345, 0.355)
    rng("4) 他方策の低下率 上", max(drops[1:]), 0.535, 0.545)

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

    # --- §IV-C2 冒頭 single8 の幅 ---
    g = hw[hw.midi == D.SINGLE_HW].groupby(["model", "seed"]).success_rate.mean().unstack(0)
    chk("IV-C2 single8 の幅", g.mean().max() - g.mean().min(), 0.35, 0.006)

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

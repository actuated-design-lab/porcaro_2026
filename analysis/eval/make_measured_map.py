"""make_measured_map.py — PAM 単体試験（2026/10/20、DF_robot）の実測から、sim の力マップと自由収縮マップを作る

入力: jetson_project の data/user0/pamtest/summary/（analysis/pamtest/force_map.py の出力）
  isometric_table.csv：等尺性。長さ（c_set）× 圧力 × 上げ/下げ の 3 往復平均（P, h [%], F [N]）
  isotonic_table.csv ：等張性。おもり × 足す/外す × 圧力 × 上げ/下げ（h [%]、力 = おもり × g）
作り方（圧力の行 0.1〜0.6 MPa ごと）:
  1. ヒステリシスの真ん中を取る：等尺性は上げ・下げの力の平均（0.6 MPa は上げのみ）、等張性は上げ・下げ・足す・外すの
     収縮率の平均。sim はこのマップに摩擦（pam_hys_*）を足すので、マップは真ん中にする
  2. 等尺性は力 > 2 N の点だけ（低い圧力で筋が治具に届かず力がほぼ 0 の点は、その長さの力ではないので除く）
  3. 自由収縮 h0(P) = 等張性 0 kg の真ん中
  4. 点を収縮率の順に並べ、力が収縮率とともに減るように単調化（隣り合う逆転は平均でならす）。h0 で 0
  5. 列（-15〜25 %、0.5 % 刻み）へ直線補間。最小の収縮率（等尺性 0 %）より伸びた側は 0→2 % の傾きで延ばし、630 N で頭打ち
  0 MPa の行はカタログのまま（伸ばしたときの力は測っていない）。0.7・0.8 MPa は 0.5→0.6 MPa の変化を延ばした外挿
  測った圧力の間は 0.025 MPa 刻みの行を足す。h ≥ 0 は曲線を h / h0(P) でそろえて補間するので、どの圧力でも力がちょうど
  h0(P) で 0 になる（sim は力マップを双 3 次で細かくしたうえで h > h0 で力を切るので、行が粗いと h0 で数 N の段差が出ていた）
  最後に、どの収縮率でも圧力が高いほど力が小さくならないようにする（伸ばした側の外挿が 0 MPa の行を下回らないように）
  （ITV は 0.6 MPa までなので sim では使わない。補間の端のためだけ）
  自由収縮マップ（pam_force_0_map の形：行 = 圧力、列 = 荷重 0〜620 N、値 = その荷重での収縮率 %、下限 -3）は
  力マップを圧力ごとに逆に引いて作る（列 0 N が h0）
出力: source/porcaro_2026/porcaro_2026/tasks/data/pam_force_map_measured.csv, pam_force_0_map_measured.csv
  使うときは force_scale = 1（カタログ × 0.2 の代わり）。tasks/data/README.md

  python3 analysis/eval/make_measured_map.py --jetson ../jetson_project
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
D = os.path.join(ROOT, "source/porcaro_2026/porcaro_2026/tasks/data")
G = 9.80665
F_CAP = 630.0
P_MEAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
P_STEP = 0.025          # 出力する圧力の行の刻み（測った 0.1 刻みの間を埋める）
H_STEP = 0.5            # 出力する収縮率の列の刻み [%]（h0 での 0 への落ち方を細かく表すため）


def decreasing(h, F):
    """F が h とともに減るように、隣り合う逆転を平均でならす（pool adjacent violators）。"""
    blocks = [[f, 1] for f in F]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            v, n = blocks[i]
            v2, n2 = blocks[i + 1]
            blocks[i] = [(v * n + v2 * n2) / (n + n2), n + n2]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, n in blocks:
        out += [v] * n
    return np.asarray(h), np.asarray(out)


def curve(gi, gt, P):
    """圧力 P の行：収縮率の点と力の点（真ん中）、h0。"""
    i = gi[(gi.P_cmd == P) & (gi.F > 2)]
    iso = i.groupby("c_set").agg(h=("h", "mean"), F=("F", "mean")).reset_index()   # 上げ・下げの平均
    t = gt[gt.P_cmd == P].groupby("load_kg").agg(h=("h", "mean")).reset_index()  # 上げ・下げ・足す・外すの平均
    t["F"] = t.load_kg * G
    h0 = float(t.loc[t.load_kg == 0.0, "h"].iloc[0])
    pts = pd.concat([iso[["h", "F"]], t[["h", "F"]]]).sort_values("h")
    pts = pts[pts.h < h0]
    h, F = decreasing(pts.h.values, pts.F.values)
    return np.r_[h, h0], np.r_[F, 0.0], h0, len(iso), len(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jetson", default=os.environ.get("JETSON_PROJECT", os.path.join(os.path.dirname(ROOT), "jetson_project")))
    a = ap.parse_args()
    S = os.path.join(a.jetson, "data", "user0", "pamtest", "summary")
    gi = pd.read_csv(os.path.join(S, "isometric_table.csv"))
    gt = pd.read_csv(os.path.join(S, "isotonic_table.csv"))
    cat = pd.read_csv(os.path.join(D, "pam_force_map.csv"), index_col=0)
    cat0 = pd.read_csv(os.path.join(D, "pam_force_0_map.csv"), index_col=0)
    Hc = cat.columns.astype(float).values
    H = np.round(np.arange(Hc[0], Hc[-1] + 1e-9, H_STEP), 3)   # 列（収縮率 %）。カタログの 1 % より細かく
    cat_row0 = np.interp(H, Hc, cat.loc[0.0].values.astype(float))
    Fcols = cat0.columns.astype(float).values

    rows, h0s = {}, {0.0: float(cat0.loc[0.0].iloc[0])}
    for P in P_MEAS:
        h, F, h0, ni, nt = curve(gi, gt, P)
        slope = (F[1] - F[0]) / (h[1] - h[0]) if len(h) > 2 and h[1] > h[0] else 0.0
        row = np.interp(H, h, F)
        lo = H < h[0]
        row[lo] = F[0] + slope * (H[lo] - h[0])
        rows[P] = np.clip(row, 0.0, F_CAP)
        h0s[P] = h0
        print(f"  {P:.1f} MPa：等尺性 {ni} 点・等張性 {nt} 点、h0 {h0:.2f} %、F(0 %) {np.interp(0.0, H, rows[P]):.1f} N")
    for P in (0.7, 0.8):                                  # 外挿（sim では使わない）
        k = round((P - 0.6) / 0.1)
        rows[P] = np.clip(rows[0.6] + k * (rows[0.6] - rows[0.5]), 0.0, F_CAP)
        h0s[P] = h0s[0.6] + k * (h0s[0.6] - h0s[0.5])
    rows[0.0] = cat_row0

    # 測った圧力の間の行を足す（P_STEP 刻み）。h ≥ 0 は曲線を h / h0(P) でそろえて隣の行から補間するので、
    # どの圧力でも力がちょうど h0(P) で 0 になる（自由収縮マップと食い違わない。sim は h > h0 で力を 0 にする）。
    # h < 0（伸ばした側）は同じ収縮率どうしで直線補間。h0(P) は測った点を単調な 3 次（PCHIP）でつなぐ
    known = sorted(rows)
    h0f = PchipInterpolator(known, [h0s[P] for P in known])
    for P in np.round(np.arange(0.0, 0.6 + 1e-9, P_STEP), 4):
        if any(abs(P - k) < 1e-9 for k in known):
            continue
        lo = max(k for k in known if k < P)
        hi = min(k for k in known if k > P)
        t = (P - lo) / (hi - lo)
        h0P = float(h0f(P))
        h0s[P] = h0P
        row = (1 - t) * rows[lo] + t * rows[hi]                       # h < 0 の側
        pos = H >= 0
        x = H[pos] / h0P                                              # h0(P) で正規化した収縮
        Flo = np.where(x < 1, np.interp(x * h0s[lo], H, rows[lo]), 0.0)
        Fhi = np.where(x < 1, np.interp(x * h0s[hi], H, rows[hi]), 0.0)
        row[pos] = (1 - t) * Flo + t * Fhi
        rows[P] = np.clip(row, 0.0, F_CAP)
    fm = pd.DataFrame(rows, index=[f'{h:g}' for h in H]).T.sort_index()
    fm.index.name = None
    # 圧力が高いほど力は小さくならない（伸ばした側の外挿が 0 MPa のカタログより弱くならないように）
    fm = pd.DataFrame(np.maximum.accumulate(fm.values, axis=0), index=fm.index, columns=fm.columns)

    # 自由収縮マップ：荷重 F での収縮率（力マップを逆に引く）。下限 -3 %（カタログと同じ）
    hf = np.linspace(-15, 25, 4001)
    z = {}
    for P in fm.index:
        if P == 0.0:
            z[P] = cat0.loc[0.0].values
            continue
        Fr = np.interp(hf, H, fm.loc[P].values.astype(float))
        Fr = np.where(hf >= h0s[P], 0.0, Fr)
        hh = []
        for f in Fcols:
            if f == 0:
                hh.append(h0s[P])
                continue
            idx = np.where(Fr >= f)[0]                     # 力が f 以上ある一番縮んだ位置
            hh.append(max(hf[idx[-1]], -3.0) if len(idx) else -3.0)
        z[P] = np.asarray(hh)
    f0 = pd.DataFrame(z, index=cat0.columns).T.sort_index()
    f0.index.name = None

    fm.round(3).to_csv(os.path.join(D, "pam_force_map_measured.csv"))
    f0.round(3).to_csv(os.path.join(D, "pam_force_0_map_measured.csv"))
    print("保存:", os.path.join(D, "pam_force_map_measured.csv"))
    print("保存:", os.path.join(D, "pam_force_0_map_measured.csv"))
    show = ["0", "2", "4", "6", "8", "10", "12", "14", "16", "18", "20"]
    print("\n力マップ [N]（列 = 収縮率 %）\n" + fm[show].round(1).to_string())
    print("\n自由収縮 h0 [%]：" + ", ".join(f"{P:g} {f0.loc[P].iloc[0]:.2f}" for P in f0.index))


if __name__ == "__main__":
    main()

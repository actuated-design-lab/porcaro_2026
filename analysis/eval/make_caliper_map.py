"""make_caliper_map.py — カタログの力マップを、ノギスで測った自由収縮に合わせて圧力ごとに横に縮めた「予想マップ」を作る

元: tasks/data/pam_force_map.csv（カタログ DMSP-10-100N を読み取ったもの）、pam_force_0_map.csv（荷重ごとの収縮率、力0の列が自由収縮）
実測: 初期にノギスで測った PAM（PA）の自由収縮 [%]（150 mm → 0.6 MPa で 128 mm）
  0.1: 0.667（別測定 3 回の平均）, 0.2: 2.7, 0.3: 5.3, 0.4: 8.0, 0.5: 12.0, 0.6: 14.7
  0.7, 0.8 MPa は未測定。カタログ比 0.70, 0.75 で仮置き（ITV は 0.6 MPa までなので sim では使わない）
変換: F_pred(P, h) = F_cat(P, h × h0_cat(P) / h0_meas(P))（力が0になる位置を実測に合わせ、曲線の形はカタログのまま）
出力: tasks/data/pam_force_map_caliper.csv, pam_force_0_map_caliper.csv
注意: 荷重をかけたときの形は予想。吊るす試験（荷重 × 圧力）で置き換える前提の暫定マップ
"""
import os
import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "source/porcaro_2026/porcaro_2026/tasks/data")
H0_MEAS = {0.0: 0.011, 0.1: 0.667, 0.2: 2.7, 0.3: 5.3, 0.4: 8.0, 0.5: 12.0, 0.6: 14.7}

fm = pd.read_csv(os.path.join(D, "pam_force_map.csv"), index_col=0)
h0 = pd.read_csv(os.path.join(D, "pam_force_0_map.csv"), index_col=0)
H = fm.columns.astype(float).values
h0c = h0["0"].to_dict()
meas = {P: H0_MEAS.get(round(P, 1), h0c[P] * {0.7: 0.70, 0.8: 0.75}.get(round(P, 1), 1.0)) for P in fm.index}
out = fm.copy()
for P in fm.index:
    k = h0c[P] / meas[P] if P > 0 else 1.0
    out.loc[P] = np.interp(H * k, H, fm.loc[P].values)
out.round(3).to_csv(os.path.join(D, "pam_force_map_caliper.csv"))
h0o = h0.copy()
for P in h0.index:
    h0o.loc[P] = (h0.loc[P] * (meas[P] / h0c[P] if P > 0 else 1.0)).values
h0o.round(3).to_csv(os.path.join(D, "pam_force_0_map_caliper.csv"))
print("保存:", os.path.join(D, "pam_force_map_caliper.csv"), os.path.join(D, "pam_force_0_map_caliper.csv"))

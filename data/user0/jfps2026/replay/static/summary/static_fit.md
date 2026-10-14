# 静的なつり合いの当てはめ（static_fit.py）

入力: `data/user0/jfps2026/replay/static/summary/static_curves.csv`（2周目の中心曲線）

## 1. 簡易モデルの検証（sim のパラメータで sim を再現）

- 重力トルク τg = 0.216 + 0.004·θ[rad] N·m
- 再現誤差 RMSE 2.34 deg

```
ch  ant  rmse  offset
DF  0.1  1.77    0.66
DF  0.2  2.70    0.53
DF  0.3  3.14    2.12
 F  0.1  1.14    0.54
 F  0.2  0.91   -0.82
 F  0.3  3.24   -2.95
```

## 2. 実機：sim のパラメータのまま

RMSE 4.71 deg

```
ch  ant  rmse  offset
DF  0.1  5.72    4.72
DF  0.2  3.51    2.01
DF  0.3  3.02    0.23
 F  0.1  2.56    0.43
 F  0.2  3.96   -1.23
 F  0.3  7.52   -4.00
```

## 3. 実機に当てはめ（重力は sim のまま）

- force_scale DF = 0.241、F = 0.048（今は 0.2, 0.2）
- theta_t_DF_deg = 3.2、theta_t_F_deg = 35.7（今は 7, 70）
- RMSE 2.71 deg

```
ch  ant  rmse  offset
DF  0.1  3.16   -1.26
DF  0.2  3.06   -0.55
DF  0.3  4.17    2.93
 F  0.1  1.09   -0.39
 F  0.2  1.45   -0.13
 F  0.3  2.00   -0.54
```

## 4. 実機に当てはめ：F のワイヤの張り始めを入れる（θt_F = 70 固定）

- force_scale DF = 0.164、F = 0.226
- theta_t_DF_deg = 7.1
- pam_engage_width_deg（F）= 46.8（23.2 deg で張り始めて 70 deg で張り切る）
- RMSE 2.14 deg

```
ch  ant  rmse  offset
DF  0.1  2.55    0.86
DF  0.2  2.71   -1.19
DF  0.3  2.58   -0.01
 F  0.1  1.66    1.57
 F  0.2  1.01    0.17
 F  0.3  1.76   -1.00
```

倍率は重力との比しか決まらない（重力を変えると倍率が一緒に動き、RMSE はほぼ同じ）:

- 重力 ×2: DF 0.297、F 0.303、θt_DF 7.5、張り始め 17.9 deg、RMSE 2.21 deg
- 重力 ×3: DF 0.426、F 0.382、θt_DF 7.9、張り始め 15.2 deg、RMSE 2.40 deg

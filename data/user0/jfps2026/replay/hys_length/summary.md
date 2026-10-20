# PAM 摩擦の向きの決め方の比較（compare_hys_modes.py）

手首角 [deg]、最初の 2 s を除く。止まっている＝実機の角速度 < 5 deg/s、動いている＝ > 50 deg/s

## RMSE

```
tag                         d0  len001  len003   meas  play_p    s04  s04d0
motion                                                                     
id_static_F_taut           NaN   10.07    8.09  16.19   10.46    NaN    NaN
id_static_cocontraction    NaN    5.76    5.56  10.92    4.92    NaN    NaN
tm_B_steps               16.03    4.13    3.88  10.89    5.10  11.57  21.14
tm_C_sine                25.63    8.42    7.68  16.27    9.29  15.63  29.39
```

## 止まっているときの RMSE

```
tag                         d0  len001  len003   meas  play_p    s04  s04d0
motion                                                                     
id_static_F_taut           NaN   10.17    8.00  16.23   10.62    NaN    NaN
id_static_cocontraction    NaN    5.59    5.44  10.82    4.85    NaN    NaN
tm_B_steps               15.62    3.59    3.34  10.54    4.81  11.11  20.42
tm_C_sine                22.98    4.93    3.90  14.99    7.48  13.02  27.24
```

## 動いているときの RMSE

```
tag                         d0  len001  len003   meas  play_p    s04  s04d0
motion                                                                     
id_static_F_taut           NaN   11.04   10.19  20.36   12.11    NaN    NaN
id_static_cocontraction    NaN    9.43    9.58  14.95    4.70    NaN    NaN
tm_B_steps               22.94   10.15    9.63  16.01    9.80  17.34  30.66
tm_C_sine                27.63   11.15   10.43  17.88   11.38  17.83  30.05
```

## 平均ずれ（sim − 実機）

```
tag                        d0  len001  len003   meas  play_p    s04  s04d0
motion                                                                    
id_static_F_taut          NaN    5.42    5.05  13.60    4.86    NaN    NaN
id_static_cocontraction   NaN   -1.24   -1.47   7.17    0.44    NaN    NaN
tm_B_steps               3.19    1.88    1.29   7.89    2.44   8.47  10.32
tm_C_sine                3.81    2.96    2.17  14.34    4.78  13.67  10.47
```

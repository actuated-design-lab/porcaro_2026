# replay/ — 開ループ再生の結果

`scripts/replay_open_loop.py --out data/user0/jfps2026/replay/<名前>.csv` の出力。

ファイル名は `sim_tm_<テスト動作>_<圧力モデル>.{csv,json}`:
- テスト動作: `A_quasistatic`, `B_steps`, `C_sine`, `D_antagonist`, `E_<曲>_seed<n>`
- 圧力モデル: `table`, `lag`, `orifice`, `measured`
  - 10/8 追加（`scripts/itv_models.py`、ITV の小振幅・高周波のしきい値入り）:
    - `shaped`：モデルA（現象論）。指令の小さく速い振動を、実測の振幅比マップで縮めてから流量上限モデル O に通す
    - `pilot`：モデルB。パイロット段（積分）＋重なりを持つ主弁
    - `pilot_leak`：モデルB＋漏れ。重なりの中でも主弁が少し漏れる
    - `orificeV2`：`--pmodel orifice` を shaped の土台と同じ O パラメータで回したもの（shaped との差がしきい値の効果だけになる比較の基準）。
      `--params '{"tau":0.0833,"L":0.045,"c_in":4.8542,"c_out":9.962,"ps":0.5776,"b":0.1323}'`

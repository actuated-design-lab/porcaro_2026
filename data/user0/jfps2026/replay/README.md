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
  - 10/9 追加：
    - `measuredV2`：`--pmodel measured` を今の既定（実測圧力に 50 ms の移動平均＋±10 kPa の遊び, 10/6 追加）で回し直したもの。
      `measured` は 10/6 のフィルタ追加前のスクリプトで作ったもので、実測圧力を生のまま入れている（json に `meas_filter` がない。生だと圧力ノイズでヒステリシスの向きが毎ステップ反転し、sim のヒステリシスが実質消える）

## compare/ — 実機との比較（`analysis/eval/compare_replay.py`）

`python analysis/eval/compare_replay.py` で作り直せる（実機ログは `../jetson_project/test_signals/` をパスで参照。別の場所なら `--jetson`）。
- `summary.md`：動作 × 圧力モデルの表（角度の RMSE・平均ずれ・平均ずれを引いた RMSE・相互相関の遅れ、圧力の NRMSE）
- `metrics.csv`：同じ指標の全部（1 行 = 動作 × 圧力モデル）
- `wire_screen.csv`：ワイヤー外れの洗い出し（実機の角度の跳び、実機だけ張り付いて measured の sim が動いた区間）。除外はしていない
- `angles_tm_C_sine.png`, `angles_tm_C_sine_zoom.png`, `angles_tm_E_dbl160_seed2.png`：角度波形（実機と measured / measuredV2 / orificeV2 / shaped / pilot_leak）
- `overview_measured.png`：全動作の実機と measuredV2（なければ measured）

## hys/ — ヒステリシス力のモデル（relay / play）の比較（10/11）

`analysis/eval/run_hys_sweep.py` の出力。ファイル名は `sim_<動作>__<圧力モデル>__<ヒステリシス>.{csv,json}`
（圧力モデル：`measured_raw`＝実測圧力そのまま、`measured_v2`＝平滑化＋遊び、`orificeV2`。ヒステリシス：`relay`、`play_w<幅kPa>_cp<cP>`）。
集計は `analysis/eval/hys_loops.py` → `hys/summary/`。モデルの式と計画は `docs/hysteresis_play_20261011.md`。

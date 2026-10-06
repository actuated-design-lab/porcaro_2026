# replay/ — 開ループ再生の結果

`scripts/replay_open_loop.py --out data/user0/jfps2026/replay/<名前>.csv` の出力。

ファイル名は `sim_tm_<テスト動作>_<圧力モデル>.{csv,json}`:
- テスト動作: `A_quasistatic`, `B_steps`, `C_sine`, `D_antagonist`, `E_<曲>_seed<n>`
- 圧力モデル: `table`, `lag`, `orifice`, `measured`

# ral2026/ — IEEE RA-L 投稿

| 項目 | 内容 |
|---|---|
| 論文誌 | IEEE Robotics and Automation Letters (RA-L) |
| 投稿日 | 2026-09-02 |
| 投稿時のタグ | `ral2026-submit`（コミット `c4f9b6c`） |
| モデル | A=先読み0.1s/LSTM, B=0.5s/LSTM, C=1.0s/LSTM, D=0.5s/記憶なしMLP, E=0.5s/フレームスタックMLP |
| 一括実行 | `analysis/run_all_offline.sh` |
| 図表 | `analysis/summary/fig*.py`（`analysis/summary/README.md` 参照） |

## 中身

| フォルダ | 入れるもの |
|---|---|
| `eval/` | シミュレーション評価の生の出力（`main`, `nondr`, `trials5`, `tau_sweep`, `mask_*`） |
| `paper/` | 論文の図表が読む確定CSV（`sim/`, `tau/`, `hardware/`, `ablation/`, `validation/`） |
| `models/` | 実機に渡したONNXと `manifest_fragment.yaml` |

`analysis/` の既定パスはこのフォルダを向いている。**新しい実験の出力をここに混ぜないこと。**

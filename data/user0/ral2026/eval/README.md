# eval/ — RA-L のシミュレーション評価（生の出力）

| フォルダ | 実験 | 書き出すもの |
|---|---|---|
| `main/` | Model A〜E の本評価（75ジョブ） | `analysis/eval/run_eval_matrix.py` |
| `nondr/` | ドメインランダム化なしで再評価 | `run_eval_matrix.py --task_override` |
| `trials5/` | 評価ノイズ見積もりの反復試行 | `run_eval_matrix.py --trials_per_condition` |
| `tau_sweep/` | PAM 時定数 τ × 先読みのスイープ（81ジョブ） | `analysis/eval/run_eval_tau_sweep.py` |
| `mask_{zero,noise,shuffle}/` | 遠い未来の観測マスクのアブレーション | `run_eval_matrix.py --mask_mode` |

各フォルダの `*_manifest.json` は完了ジョブの記録（キーにパスは含まれない）。CSV は git に入っていない。

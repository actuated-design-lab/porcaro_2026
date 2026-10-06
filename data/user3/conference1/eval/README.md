# eval/ — シミュレーション評価の生の出力

実験ごとにサブフォルダを作る（例: `eval/main/`, `eval/tau_sweep/`）。
評価スクリプトには出力先を明示する:

```bash
python scripts/rsl_rl/play_sim_rhythm.py ... --eval_logs_root data/<user>/<フォルダ名>/eval/<実験名>
```

- 中身は `<run_tag>/<ckpt>/<条件>_trial<t>/simulation_log.csv` の木になる。
- CSV は `.gitignore` 対象なので git には入らない（研究室のマシンにだけ残る）。

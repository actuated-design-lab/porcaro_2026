# data/common/midi/ — 評価用MIDI

`analysis/eval/run_eval_matrix.py` と `scripts/rsl_rl/play_sim_midi.py` が読む評価用のMIDIファイル。

| ファイル | 用途 |
|---|---|
| `gmd_01_low_bpm80.mid` 〜 `gmd_04_extreme_bpm170.mid` | GMD由来の評価曲4曲（テンポ帯別） |
| `test_*.mid` | 単純リズムの動作確認用 |

- 既存ファイルの上書き禁止（過去の評価結果の再現性が崩れる）。新しい曲は別名で追加する。

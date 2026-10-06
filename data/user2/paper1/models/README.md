# models/ — 実機に渡すモデル

ONNX など、実機（`jetson_project`）で動かすために書き出したモデルを置く。

- `.onnx` / `.pt` は `.gitignore` 対象。どのチェックポイントから書き出したかを、manifest か README に必ず残す。
- 書き出しは `scripts/rsl_rl/export_onnx.py --out_dir data/<user>/<フォルダ名>/models/...`。

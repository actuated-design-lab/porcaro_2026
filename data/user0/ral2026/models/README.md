# models/ — RA-L で実機に渡したモデル

`analysis/export/export_onnx_matrix.py` が `staging/` に ONNX を書き出し、`manifest_fragment.yaml` を作る。
`manifest_fragment.yaml` の内容は `jetson_project` の `models/manifest.yaml` にマージする。
ONNX 本体は git 管理外。

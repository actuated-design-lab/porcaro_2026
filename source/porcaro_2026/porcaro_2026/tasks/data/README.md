# tasks/data/ — PAM の力マップ

sim（`common/actions/torque.py`）は 2 つの表で PAM の発揮力を決める。

- **力マップ**（`pam_force_map*.csv`）：行 = 内圧 P [MPa]、列 = 収縮率 h [%]（自然長 150 mm に対する縮み、負は伸び）、値 = 力 [N]
- **自由収縮マップ**（`pam_force_0_map*.csv`）：行 = 内圧 P、列 = 荷重 [N]（0〜620、20 刻み）、値 = その荷重で釣り合う収縮率 [%]。
  sim が使うのは列 0 N（自由収縮 h0）だけで、h > h0 で力を 0 にする（ワイヤのたるみ）

最終的な力は `力マップ(P, h) × force_scale`（`user0/cfg/controller_cfg.py`）に摩擦（`pam_hys_*`）を足したもの。

## ファイル

| ファイル | 中身 | 使い方 | 状態 |
|---|---|---|---|
| `pam_force_map.csv`<br>`pam_force_0_map.csv` | **旧（カタログ）**。Festo DMSP-10-100N のカタログ値を読み取ったもの | `force_scale = 0.2` | 今の既定（RA-L 投稿版・JFPS 2026 の結果はこれ） |
| `pam_force_map_caliper.csv`<br>`pam_force_0_map_caliper.csv` | 暫定（10/14）。カタログを、初期にノギスで測った自由収縮に合わせて横に縮めたもの（PR #33、`analysis/eval/make_caliper_map.py`） | `force_scale = 0.2` | 実測マップに置き換え。使わない |
| `pam_force_map_measured.csv`<br>`pam_force_0_map_measured.csv` | **新（実測）**。PAM 単体試験（2026/10/20、ロボットから外した DF の筋）の等尺性・等張性から作ったもの（`analysis/eval/make_measured_map.py`） | **`force_scale = 1.0`** | これから使う。既定はまだ旧のまま |

## 旧（カタログ × 0.2）と新（実測）の違い

| 内圧 | 収縮 0 % の力 旧 → 新 [N] | 収縮 8 % の力 旧 → 新 [N] | 自由収縮 h0 旧 → 新 [%] |
|---|---|---|---|
| 0.2 MPa | 33 → 74 | 0 → 0 | 6.7 → 4.8 |
| 0.3 MPa | 51 → 127 | 10 → 13 | 13.0 → 9.5 |
| 0.4 MPa | 67 → 182 | 22 → 64 | 17.4 → 13.8 |
| 0.5 MPa | 83 → 237 | 34 → 113 | 20.4 → 16.7 |
| 0.6 MPa | 99 → 292 | 46 → 155 | 22.1 → 18.4 |

- 実測の力はカタログの 0.3〜0.7 倍（中央値 0.58）。今の 0.2 倍は、縮みの小さい（力の大きい）側で実測の約 1/3
- 自由収縮はカタログより手前（0.6 MPa で 18.4 % 対 22.1 %）。自由収縮の近く（力が小さい側）では、旧（× 0.2）と新はだいたい重なる
- 曲線の形がカタログと違う（実測は直線に近い）ので、1 つの倍率ではカタログを実測に合わせられない

出典：jetson_project の `data/user0/pamtest/summary/summary.md`、`fig_force_length.png`（`analysis/pamtest/force_map.py`）

## 新（実測）マップの作り方と注意

- 圧力 0.1〜0.6 MPa の各行：等尺性（収縮 0〜14 %、力 > 2 N の点）と等張性（おもり 0〜3.1 kg）の点を、ヒステリシスの
  真ん中（上げ・下げの平均）で取り、力が収縮とともに減るように単調化して列へ直線補間。h0 は等張性 0 kg の真ん中
- **0 MPa の行と、伸ばした側（h < 0）は測っていない**。0 MPa はカタログのまま、h < 0 は 0→2 % の傾きで延ばして
  630 N で頭打ちにし、0 MPa のカタログを下回らないようにした。手首を下げたとき（DF は手首 −20° で約 −4 %）に効くので、
  ここが効く動作では注意
- **0.7・0.8 MPa は外挿**（0.5→0.6 MPa の変化を延ばした）。ITV は 0.6 MPa までなので sim では使わない
- 測った筋は DF の 1 本、1 日分。F・G は測っていない（3 本とも同じマップを使う）
- 内圧は試験機の G の圧力センサで測った（DF のセンサが故障）。0.6 MPa の段の実測は 0.59〜0.60 MPa
- 等張性の吊り具の重さは 0 として扱った
- ヒステリシス：長さのヒステリシスが大きい（等張性 0 kg で、上げ・下げの収縮率の差が最大 3.1 %、圧力の幅で 25〜70 kPa）。
  同じ長さでの力のヒステリシスは 3.5〜10 N で、圧力が上がると小さくなった（sim の摩擦 2.5 × (0.5 + 15 P) N は圧力とともに
  大きくなる）。摩擦のモデルは別に見直す

## 新マップで sim を回す

既定を変えずに、`replay_open_loop.py` などで controller の設定を上書きする（パスはリポジトリのルートから。ルートで実行する）。

```bash
python scripts/tools/user0/jfps2026/replay_open_loop.py ... --ctrl '{
  "force_map_csv": "source/porcaro_2026/porcaro_2026/tasks/data/pam_force_map_measured.csv",
  "h0_map_csv":    "source/porcaro_2026/porcaro_2026/tasks/data/pam_force_0_map_measured.csv",
  "force_scale": 1.0}'
```

既定を新マップに切り替えるのは、静的な試験（10/14 の共収縮・id_static_F_taut）と tm_B・tm_C の再生で確かめてから。

## 作り直す

```bash
# jetson_project 側で試験のまとめを作る
python3 analysis/pamtest/force_map.py --porcaro ../porcaro_2026     # jetson_project で
# porcaro_2026 側でマップを作る
python3 analysis/eval/make_measured_map.py --jetson ../jetson_project
```

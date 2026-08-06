# 図の生成スクリプト

**グラフ1枚につき1スクリプト。** 共通のスタイルとデータ読み込みだけ切り出してある。

| ファイル | 役割 |
|---|---|
| `ral_figstyle.py` | **フォント・配色・線幅はここだけ触れば全図に効く** |
| `ral_data.py` | 集計済みCSVの読み込みと、シード単位への集約・検定 |
| `fig4_simresults.py` | 本文 Fig.4（(a) 学習曲線 (b) sim 成功率）2段幅 |
| `fig5_quanteval.py` | 本文 Fig.5（(a) 5方策 sim/実機 (b) 誤差分布 (c) D のシード別 (d) マスク）1段幅 |
| `figS_tau_sweep.py` | 補足: τスイープのヒートマップ |
| `figS_mask_ablation.py` | 補足: マスク単体（本文では Fig.5(d) に統合済み。スライド用） |
| `verify_ral_numbers.py` | 数値を FINAL の記載と突き合わせる（82項目） |

## 必要なパッケージ

```
pip install numpy pandas scipy matplotlib
pip install tensorboard          # fig4 の (a) 学習曲線だけに必要
```

GPU も isaaclab も torch も要らない。Python 3.9 以降。
`tensorboard` を入れない場合は `fig4_simresults.py --skip_curves` で (b) だけ作れる。

## 使い方

```
cd porcaro_2026
python analysis/summary/fig4_simresults.py --outdir ../RALpaper/figures
python analysis/summary/fig5_quanteval.py  --outdir ../RALpaper/figures
python analysis/summary/verify_ral_numbers.py
```

共通オプション:

| オプション | 既定 | 意味 |
|---|---|---|
| `--outdir` | `figures` | 出力先 |
| `--format` | `pdf` | `pdf` / `png`(300dpi) / `both` |
| `--fontsize` | `8.0` | 本文相当のポイント数。全要素がこれに追従する |
| `--height` | 図による | 図の高さ [in]。幅は1段3.5in / 2段7.16inで固定 |

`fig5_quanteval.py` だけ `--panel a|b|c|d` があり、単一パネルを別ファイルに出せる（スライド用）。

## フォント

Times 系。`ral_figstyle.py` の `SERIF_STACK` の先頭が `Times New Roman` なので、
**Windows では実物の Times New Roman が使われる。** 無い環境では
STIXGeneral → TeX Gyre Termes → Liberation Serif の順にフォールバックする
（いずれも Times メトリック互換）。数式は mathtext の STIX セットで字面が揃う。

別のフォントにしたいときは `SERIF_STACK` の先頭に足すだけでよい。
入っているフォント名は次で確認できる:

```
python -c "import matplotlib.font_manager as fm; print(sorted({f.name for f in fm.fontManager.ttflist}))"
```

## 微調整するときに触る場所

**配色** — `ral_figstyle.py` の `MODEL_COLORS` / `DOMAIN_COLORS` / `MASK_COLORS`。
Okabe-Ito から選ぶこと（色覚多様性対応、全ペア検証済み）。

**★紙面** — 本文は8ページちょうど。図の高さを変えたら必ずページ数を確認すること。
- `fig4_simresults.py`: `--height 1.7`（既定）
- `fig5_quanteval.py`: `--height 4.42`（既定）。**4.7 にすると9ページになる。ここが一番シビア**

## 依存するデータ

```
paper_data/sim/sim_summary_1N.csv          sim 本評価 75ラン
paper_data/sim/sim_strikes_1N.csv          同・打点ごと
paper_data/tau/tau_summary_1N.csv          τスイープ 81ラン
paper_data/hardware/hw_summary_s4_1N.csv   実機 第4セッション 105ラン
paper_data/hardware/hw_strikes_s4_1N.csv   同・打点ごと
paper_data/ablation/mask_*_summary_1N.csv  マスク 各15ラン
```

`fig4` の (a) だけ TensorBoard の event ファイル（`logs/rsl_rl/`）が要る。
展開後は `logs/` の更新日時を過去にすること（`discover.py` が最新更新のランを
「学習中」とみなして1件落とすため）。

```
Windows: Get-ChildItem -Recurse logs | %{ $_.LastWriteTime = "2026-01-01" }
Linux  : find logs -depth -exec touch -d "2026-01-01" {} +
```

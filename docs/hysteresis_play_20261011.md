# PAM のヒステリシス力：今のモデル（relay）と遊び型（play）、10/11 の実験計画

2026/10/10 作成。JFPS 2026秋季（data/user0/jfps2026/）と修論の機構モデルの見直し用。
実装は `common/actions/pam.py`（`calculate_simple_latched_friction` / `calculate_play_friction` ほか）と
`common/actions/torque.py`（`pam_hys_mode` で切り替え）。

---

## 1. 共通：PAM 1 本の発揮力

筋 $i \in \{\mathrm{DF}, \mathrm{F}, \mathrm{G}\}$ の発揮力は

$$
F_i = E\!\left(\max\!\big(0,\; F_{s,i}(P_i, h_i) + F_{f,i}\big),\; h_i\right)
$$

- $P_i$：PAM の内圧 [MPa]（圧力モデルの出力）
- $h_i$：収縮率（関節角とワイヤーのたるみから計算）
- $F_{s,i}(P,h) = k_i\, M(P,h)$：静的な力。$M$ は力マップ（CSV）、$k_i$ は `force_scale`（RA-L は 0.2）
- $E(\cdot)$：たるみ→張りの切り替え（`apply_soft_engagement`。$h > h_0(P)$ で 0）
- $F_{f,i}$：摩擦（ヒステリシス）力。以下で「大きさ」と「向き」に分ける

$$
F_{f} = -\eta\,\dot h \;-\; \underbrace{\left(c_0 + c_P\,|P|\right)}_{\text{大きさ}}\; g(d)\; d,
\qquad
g(d) = \gamma_c\,\frac{1+d}{2} + \gamma_e\,\frac{1-d}{2}
$$

- $d \in [-1, 1]$：ヒステリシスの向き。$+1$ ＝ 加圧の枝（力が静的値より小さい）、$-1$ ＝ 減圧の枝（大きい）
- $c_0 = 0.5$ N、$c_P = 15$ N/MPa（`pam_hys_const`, `pam_hys_coef_p`）。0.3 MPa で 5 N
- $\gamma_c = 1.5$、$\gamma_e = 1.0$（収縮側・伸長側の非対称。`pam_contract_gain`, `pam_extend_gain`）
- $\eta = 0$（粘性。`pam_viscosity`）

**向き $d$ を内圧の変化で決める理由**（IROS のときの判断）：PAM のヒステリシスの主因は編み込み繊維どうしの摩擦で、
その大きさは内圧で変わる。関節角はワイヤーのたるみがあると PAM の長さと一対一にならないので、向きの決定には使えない。
→ 内圧の変化の向きをヒステリシスの向きとみなす。この判断は文献とも整合する（PAM のヒステリシスは圧力を入力にした
ヒステリシス作用素で表すのが一般的。Tondu 2012、Liu ら 2017、Kastner ら 2012）。

---

## 2. 今のモデル（relay、`pam_hys_mode="relay"`、既定・RA-L まで）

向きを、内圧の**変化率**の符号で決める。

$$
v_k = \mathrm{clip}\!\left(\kappa\,\frac{P_k - P_{k-1}}{\Delta t},\; -1,\; 1\right),
\qquad
d_k =
\begin{cases}
\tanh\!\left(v_k / 0.1\right) & |v_k| > 0.25 \\[2pt]
d_{k-1} & \text{それ以外（保持）}
\end{cases}
$$

- $\kappa = 100$ s/MPa（`pam_p_dot_scale`）、$\Delta t$ = 制御周期
- 切り替わる条件は $|\dot P| > 2.5$ kPa/s。そのとき $\tanh(2.5) = 0.99$ なので、**ほぼ符号関数**
- つまり、圧力が少しでも反転すると、摩擦力が $2(c_0 + c_P P)$（0.3 MPa で約 10 N、非対称込みで最大 12.5 N）だけ**一瞬で**跳ぶ

### 問題（10/10 に確認）

tm_A（DF の 0.05 Hz 三角波）で、手首角−DF 圧力のループを比べた（`analysis/eval/hys_loops.py`）。

| | ループの横幅 ΔP@θ（tri0 / tri1） | 縦幅 Δθ@P |
|---|---|---|
| 実機 | 80 / 53 kPa | 9.4 / 4.0° |
| sim（流量上限モデルの滑らかな圧力） | 95 / 95 kPa | 13.8 / 13.8° |
| sim（実測圧力をそのまま入力） | 4 / 2 kPa | 0.8 / 0.8° |

1. 滑らかな圧力では、ループが実機より広い（1 周目と同程度〜2 周目の 2 倍）。実機は 1 周目と 2 周目で幅が違う（履歴の影響）が、sim は同じ
2. 圧力に数 kPa のノイズが乗ると、向きが毎ステップ反転してヒステリシスが消える
   （`replay_open_loop.py` の measured モードに平滑化と遊びを入れていたのはこのため。ただし遊びを圧力そのものにかけると 30 ms ほど遅れる）
3. 圧力のわずかな差で枝が切り替わるので、関節角が圧力の小さな違いに敏感になる
   （正弦で DF 2 kPa の差が手首角 15°、ステップの動作でしきい値の有無により圧力誤差は同じ 2.2 % なのに手首角 4.8° と 8.5°）

文献では、PAM のヒステリシスは**速さにほとんど依存せず**（Festo DMSP-5 で 0〜0.2 Hz のループ幅がほぼ同じ）、
**枝は圧力がある程度戻ってから連続に切り替わる**（Preisach / Prandtl–Ishlinskii などの幅を持つ作用素で表す）。
relay は「幅 0・変化率で切り替え」なので、この 2 点と合わない。

---

## 3. 修正案：遊び型（play、`pam_hys_mode="play"`）

向きの入力は内圧のまま（§1 の判断は変えない）。決め方だけを、**変化率の符号**から**遊び作用素**に変える。

$$
z_{j,k} = \min\!\Big(\max\big(z_{j,k-1},\; P_k - w_j\big),\; P_k + w_j\Big),
\qquad z_{j,0} = P_0
$$

$$
d_k = \sum_{j=1}^{J} a_j\,\frac{P_k - z_{j,k}}{w_j},
\qquad \sum_j a_j = 1,\quad d_k \in [-1, 1]
$$

摩擦力は §1 の式に、この $d_k$ を入れる（大きさ・非対称・粘性はそのまま）。

- $w_j$：遊びの片側の幅 [MPa]（`pam_hys_play_widths`）。$a_j$：重み（`pam_hys_play_weights`、既定は等分）
- $J = 1$ のとき：圧力が上がり続けると $d = +1$、反転して $2w$ 戻ると $d = -1$。その間は**連続に**移る
- $J > 1$（幅の違う要素を並べる）：Prandtl–Ishlinskii 型。反転の大きさに応じて枝が段階的に移り、内側のループ（小さな往復）も表せる

### 性質

| | relay（今） | play（案） |
|---|---|---|
| 切り替えのきっかけ | $\lvert\dot P\rvert > 2.5$ kPa/s（速さ） | 圧力が $2w$ 戻る（量） |
| 切り替わり方 | 一瞬で $\pm$ 全量 | $2w$ の間で連続 |
| 速さへの依存 | あり（しきい値が速さ） | なし（文献と整合） |
| $w$ より小さいノイズ・揺れ | 毎回反転 | 向きはほぼ変わらない |
| 実測圧力をそのまま入力 | ヒステリシスが消える | 残る（平滑化・遊びを圧力にかけなくてよい） |

ループの横幅（同じ角度での上げ・下げの圧力差）は、おおよそ $2w$ と $2\,\bar F_f / (\partial F_s/\partial P)$ の和で決まる
（$\bar F_f$：摩擦の大きさ）。実機の 53〜80 kPa に合わせるには、$w$ と $c_P$ を一緒に決める必要がある。

### 残る限界

- 内圧が一定のまま長さが変わる場合（拮抗筋に引かれる）の摩擦は表せない（入力が内圧だけのため）。
  修論で機構側を作り直すときは、入力を静的な力 $F_s(P, h)$ にする（圧力と長さの両方の変化を拾う）案がある
- $J = 1$ では 1 周目と 2 周目の違い（実機 80 → 53 kPa）は表せない。$J = 2$〜3 で確かめる

---

## 4. 10/11 の実験計画（研究室 PC・Isaac Lab のみ。実機は使わない）

### 準備（10 分）
1. porcaro_2026：PR（hysteresis-play）をマージして `git pull`。jetson_project：`git pull`（v4 の配置）
2. `python scripts/replay_open_loop.py --signal x --out x -h` が通ること（Isaac Lab の環境 env_isaaclab）
3. `python analysis/eval/run_hys_sweep.py --stage 1 --dry_run` で、信号と実測ログのパスが解決されること

### Stage 1：同定（tm_A、約 30 分）
- 入力圧力は**実測圧力そのまま**（`measured_raw`）。圧力モデルの誤差を除き、ヒステリシスだけを比べる
- 条件：relay（今）＋ play の $w \in \{10, 20, 40\}$ kPa × $c_P \in \{15, 7.5\}$ N/MPa、参照に relay＋measured_v2 → 計 8 本
- `python analysis/eval/run_hys_sweep.py --stage 1` → `python analysis/eval/hys_loops.py`
- **選び方**：ループの横幅 ΔP@θ と縦幅 Δθ@P が実機（80/53 kPa、9.4/4.0°）に近く、tm_A の手首角 RMSE が小さいもの
- 境界（$w$ = 10 か 40、$c_P$ = 7.5）が選ばれたら、外側に 1 段広げて追加（例 `--stage 1 --play_ws 0.005 0.08 --play_cps 15 7.5 3.75`。既にある条件は飛ばす）

### Stage 2：検証（他の動作、約 50 分）
- `python analysis/eval/run_hys_sweep.py --stage 2 --play_w <選んだ w> --play_cp <選んだ cP>`（2要素も試すなら `--play_ws2 0.01 0.04`）
- 動作：tm_B〜E（gmd138 は元圧低下で除外）。圧力：`measured_raw` と `orificeV2`。ヒステリシス：relay と play → 計 20 本
- `python analysis/eval/hys_loops.py --fig play_w<w>_cp<cP>`
- **見ること**
  1. 実測圧力そのままで、play はヒステリシスを保つか（relay では消える）
  2. orificeV2 の手首角 RMSE が relay より下がるか（特に tm_C 正弦、tm_E 打撃）
  3. tm_B（ステップ）で、圧力のわずかな違いに角度が振られる現象が減るか
  4. tm_D の共収縮区間で、sim だけが揺れる現象が変わるか

### 時間が余ったら
- $J = 2$（例 $w$ = 10・40 kPa、等分）で tm_A の 1 周目と 2 周目の違いが出るか
- shaped（しきい値モデル）＋ play で tm_B を流し、しきい値の有無で角度が変わる現象が消えるか

### 持ち帰るもの
- `data/user0/jfps2026/replay/hys/summary/`（summary.md、loops.csv、angles.csv、loops_tm_A.pdf）をコミットして PR
  （再生結果の csv は git 管理外のまま。summary の csv と pdf は小さいので `git add -f`）
- loops_tm_A.pdf は前刷（Tsukasa-B/JFPS2026）の figs/ に入れる

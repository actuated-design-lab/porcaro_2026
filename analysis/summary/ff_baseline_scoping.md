# FF(model-basedフィードフォワード)ベースライン — 実装コスト見積もり

対象: RA-L R3 Weakness-3a「同定済みの遅延・ヒステリシスモデルを使ったmodel-based
feedforward制御器との比較が無い」への対応。本ドキュメントは**調査と見積もりのみ**。
実装・学習・評価は一切実行していない(read-only)。既存ファイルは変更していない。

---

## 0. 結論だけ先に

**見積もり: 人手換算 約3.5〜6人日(Claude Code併用で人の実作業は約1.5〜3日に圧縮可能)。
GPU新規学習は不要(eval実行のみ、既存play_sim_*.pyと同オーダーの軽量ジョブ)。
最大の不確実性は「Phase 2: FFゲイン調整」の所要時間と、
「複雑リズムで崩れる」という結果が科学的に意味のある崩れ方かどうかの見極め。**
go/no-goの推奨は末尾(§5)。

---

## 1. 再利用できるもの / 新規に作るもの

### 1-1. 順モデル(逆算の材料) — 完全に再利用可能、変更不要

- `source/porcaro_2026/porcaro_2026/tasks/direct/porcaro_2026/common/actions/pneumatic.py`
  - `P_TAB` / `TAU_TAB` / `L_TAB`(1D, pneumatic.py:14-16)、
    `TAU_TABLE_2D_DATA` / `DEAD_TABLE_2D_DATA` / `get_2d_tables()`(19-44、状態依存2Dテーブル)。
  - `tau_L_from_pressure()`(161-168)、`first_order_lag()`(170-182)、
    `FractionalDelay`(184-240、可変むだ時間の分数遅延バッファ)。
  - **isaaclab/isaacsim非依存の純torch実装**(import は `torch`/`numpy`/`scipy.interpolate`のみ)。
    GPU/シム起動なしでCPU上で単体呼び出し・単体テスト可能 — これがPhase 0の検証を
    軽くしている最大の理由。
- `source/porcaro_2026/.../common/actions/pam.py`
  - `PAMChannel`(163-277): `step(P_cmd)`が実際に呼ばれている経路そのもの。
    `use_2d_dynamics=True`固定(pam.py:179)なので、実運用では**常に2Dテーブル
    (P_cmd, P_start_latch)引き**でtau/Lが決まる(258行目以降の`use_table_i`分岐は
    実質未使用)。`P_start_latch`は「直近の方向反転時の圧力」を保持するラッチ
    (232-250行目)— ヒステリシスの状態変数はこれ1個。
  - `calculate_effective_contraction()`(108-139、幾何→有効収縮率h)、
    `PamForceMap`/`H0Map`(45-104、2D力マップF(P,h)とh0カットオフ)、
    `calculate_simple_latched_friction()`(149-161、摩擦/ヒステリシス力)。
  - **これらは全て「自分で自分の状態を追跡できるPAMChannelインスタンスを
    もう1組(3ch分)持てば、順方向シミュレーションがそのまま使える」ことを意味する
    — FFは「シャドウ(自己複製)PAMChannel」を持つのが最小実装。**
- `source/porcaro_2026/.../common/cfg/actuator_cfg.py` / `user0/cfg/controller_cfg.py`
  - `Pmax=0.6`, `tau=0.09`, `dead_time=0.03`, `theta_t_*_deg`, `r=0.014`,
    `L0=0.150`, `force_scale=0.2`など、全て定数として既知(controller_cfg.py:13-26)。
    新規同定は不要 — 既存テーブル・定数をそのまま逆算に使う。

### 1-2. 目標表現(FFの入力) — 完全に再利用可能

- `user0/rhythm_generator.py`の`RhythmGenerator`が、エピソード開始時(`reset()`,
  72-158行目)に**4小節分の目標打撃力トレース全体を一括生成**して
  `self.target_trajectories`に保持している(スーパーガウシアンカーネルの畳み込み、
  146-158行目)。`get_lookahead()`(160-166)がこれの窓出しを行う。
- **重要な発見**: `user0/porcaro_2026_env.py`の`_get_observations()`(538-569行目)は、
  この`get_lookahead()`の出力を`target_hit_force`で正規化した上で、
  そのまま観測ベクトルの末尾`lookahead_steps`次元に載せている(552-559行目、
  `obs = cat(q(2), qd(2), prev_actions(3), sin(1), cos(1), bpm(1), lookahead(N))`)。
  **つまりFFが必要とする「目標打撃力の先読み列」も、幾何(q, qd)も、
  RLポリシーに渡るのと全く同じ`obs`テンソルの中に既に入っている。**
  → FFは`env.unwrapped.rhythm_generator`や`env.unwrapped.robot`に一切触れず、
  `policy(obs)`と全く同じ入出力契約の関数として実装できる
  (obsの列レイアウトは`scripts/rsl_rl/obs_mask.py:23`の`compute_mask_range()`が
  既に「先読み列の場所」を計算するロジックとして存在しており、これをそのまま
  流用してlookahead列を切り出せる)。
  - 設計上の含意: FFに与える先読み長は、比較対象にしたいRLモデル
    (例: B, lh=0.5s)と**同じ`lookahead_horizon`にそろえる**のが公平。
    `rhythm_generator.target_trajectories`を直接読めば「全未来を透視した」
    FFも作れてしまうが、それは別物(上限としての参考値)であり、
    本命は「obs経由・同じ先読み窓」版。

### 1-3. 行動インタフェース — 想定より小さい変更で済む

- `scripts/rsl_rl/play_sim_rhythm.py`(191-195行目)/
  `scripts/rsl_rl/play_sim_midi.py`(269-270行目)は、どちらも
  `runner = OnPolicyRunner(...); runner.load(resume_path); policy = runner.get_inference_policy(...)`
  という1箇所でしか`policy`を作っておらず、メインループは両ファイルとも
  `actions = policy(obs)`(rhythm:264行目 / midi:316行目)という1行のみ。
  `obs`は`RslRlVecEnvWrapper`適用後の**プレーンなtorch.Tensor**
  (dict化されていない)。
  → **新規に作るもの**: 両ファイルに`--ff_baseline`フラグを追加し、
  そのフラグが立っている時だけ`OnPolicyRunner`/`--checkpoint`解決をスキップして
  `policy = FFPolicy(env_cfg, ...)`を代入する分岐を1箇所ずつ挿入(各ファイル
  15〜25行程度のdiff、`--mask_mode`分岐(63-66, 138-149行目)と同種の
  「既存if分岐に1本足す」パターンなので構造的リスクは低い)。
  - `--checkpoint`前提の`resume_path`(rhythm:97-102行目)から作っている
    `ckpt_name`/`run_tag`(→`eval_out_dir`)もFF用にダミー値
    (例: `run_tag="ff_baseline"`, `ckpt_name="analytic_v1"`)に差し替えが必要。

### 1-4. 評価・集計への接続 — 既存の集計コードは無改造で再利用、ドライバのみ新規

- `analysis/eval/run_eval_matrix.py`の`build_eval_plan()`/`build_priority_plan()`は
  いずれも`discover_all_runs()`(学習済みrun_dirの一覧)を起点にしており、
  FFには対応する学習run自体が存在しない。
  → **既存のbuild_eval_plan系には手を入れず**、`run_eval_tau_sweep.py`と同型の
  **新規**の小さいドライバ(`analysis/eval/run_eval_ff_baseline.py`)を作るのが
  最も自然(このリポジトリは「新しい評価軸には専用の小さいドライバを足す」
  パターンを既に3回踏襲している: tau-sweep, mask, trials5)。
  - `MODEL_ENV_OVERRIDES`(run_eval_matrix.py:78-84)相当として、FF用に1エントリ
    (`lookahead_horizon`をBと同じ0.5sにそろえる、など)を新規ドライバ内に定義。
  - checkpointがないため`build_rhythm_command`/`build_midi_command`
    (同ファイル121-166行目)は使わず、`--ff_baseline`フラグ付きの新規コマンド
    ビルダを書く。
- `analysis/eval/aggregate_offline.py`の`extract_strikes_tree()`(55-80行目)は
  `{root}/*/*/*/simulation_log.csv`という**ディレクトリ構造だけ**を見ており、
  FFの出力(1-3節のダミーrun_tag/ckpt_name)も1-3節の`eval_out_dir`命名を
  変えなければ**無改造でそのまま歩ける**。
  → **新規に作るもの**: `analysis/eval/aggregate_ff_baseline.py`
  (`extract_strikes_tree(Path("eval_logs_ff"))` → `strikes["model"]="FF"` →
  `analysis.harness.stats.seed_level()`/`across_seed()`を**無改造で**呼ぶだけ、
  60行未満で書ける)。出力は`eval_summary.csv`と同じ列形状の別ファイル
  (例: `eval_summary_with_ff.csv`)にして既存ファイルを上書きしない。
  - `analysis/harness/tb_curves.py`の`MODEL_COLORS`/`MODEL_LABELS`
    (28-41行目)に"FF"を1エントリ追加すると、既存の図生成コードにそのまま
    載せられる(任意、2行差分)。
- FFの「seed」概念: 学習seedが存在しないため、代わりに評価seed
  (`--seed`、DR課題ならτ±20%等のランダム抽選が乗る)を複数回変えて走らせた
  ものを「seed」として`seed_level()`/`across_seed()`に渡す
  — 既存コードの列契約(`model, seed, task, eval_trial, success, timing_err_ms`)を
  満たしてさえいれば統計処理は無改造で動く。

### 1-5. 新規に作るファイルのまとめ

| ファイル | 内容 | 既存ファイルへの依存 |
|---|---|---|
| `common/actions/ff_controller.py`(新規) | シャドウ`PAMChannel`×3、目標force→目標P逆算、delay/tau逆算の固定点反復ソルバ | `pam.py`, `pneumatic.py`を**無改造**import |
| `scripts/rsl_rl/play_sim_rhythm.py`(**編集**) | `--ff_baseline`分岐 | 15-25行diff |
| `scripts/rsl_rl/play_sim_midi.py`(**編集**) | 同上 | 15-25行diff |
| `analysis/eval/run_eval_ff_baseline.py`(新規) | FF用eval実行プラン(discover_all_runs非依存) | `run_eval_matrix.py`のGMD/TASK_ID等を**import**して再利用 |
| `analysis/eval/aggregate_ff_baseline.py`(新規) | strike抽出→seed_level→across_seed | `aggregate_offline.py`/`analysis.harness.stats`を**無改造**import |
| `analysis/harness/tb_curves.py`(**任意の軽微編集**) | "FF"の色・ラベル追加 | 2行 |

---

## 2. 逆算・スケジューリングの実装アプローチ(1案)

**方針: 「obsの先読み窓 + シャドウPAMChannelによる自己状態追跡」を使った
因果的(causal)・逐次(per-step)な解析的フィードフォワード。**
全軌道を一括最適化する方式(trajectory optimization)は採らない
— 理由は§3で述べる`torch.no_grad()`制約と、実装・検証コストの両方。

各制御ステップ`t`、各チャンネル(DF/F/G)ごとに:

1. **目標力→目標圧力**: obsの先読み列(1-2節)から`target_force(t)`を取り出し、
   現在の測定`q(t), qd(t)`(obsの先頭4次元、既に`torque.py`と同じ符号規約)から
   `calculate_effective_contraction()`を**そのまま呼んで**現在の有効収縮率`h(t)`を計算。
   `PamForceMap`は`P`について単調なので、`F(P, h(t)) = target_force(t)`を満たす`P`を
   1次元二分探索(数反復、`force_map`テーブルの値域内)で求める → `P*(t)`。
   *(簡略化A: 未来の`q`を予測せず「今の`q`のまま」で力マップを引く。
   幾何と圧力動特性の結合を厳密には解かない — §3参照)*
2. **1次遅れ+むだ時間の解析的逆算**: シャドウ`PAMChannel`が自分の過去の
   コマンド履歴から`P_state_hat(t)`(現在の内部状態推定)と`P_start_latch_hat(t)`
   (ヒステリシス反転点)を保持している。
   - むだ時間`L`は状態依存(2Dテーブル)だが、Smith predictor式に**代表運用点での
     固定値**とみなして「`L`秒分だけ早く目標を送る」方式で近似補償
     *(簡略化B: 状態依存のLを毎ステップ厳密逆算しない)*。
   - 1次遅れは解析的に反転可能: `P_cmd_needed = P_state_hat + (target - P_state_hat) / alpha`
     (`alpha = dt/(tau+dt)`)。ただし`tau`自体が`(P_cmd, P_start_latch_hat)`の
     2Dテーブル引きで決まり、`P_cmd`は今まさに解いている未知数 → **2-3回の
     固定点反復**(前回推定値でtauを引き直す)で収束させる。
3. `P_cmd`を`[0, Pmax]`にクランプし、`a = 2*P_cmd/Pmax - 1`で
   `torque.py:_compute_command_pressure()`の`control_mode="pressure"`の
   逆変換に合わせてaction化。
4. シャドウ`PAMChannel.step()`に実際に採用した`P_cmd`を通し、内部状態
   (`P_state_hat`, `P_start_latch_hat`, 方向ラッチ)を進める
   (=本物の`TorqueActionController`内の`ch_DF/F/G`と**同じ更新式**を
   自分の分身に対しても回す、というだけ — 新規アルゴリズムではなく
   既存`PAMChannel`の使い回し)。

**この設計の要点**: 新規に書く数学は「2Dテーブル逆引きの固定点反復」と
「1次遅れの解析的逆算」の2つだけで、他は既存関数の呼び出しの組み合わせ。

---

## 3. 主要リスク・未知数

1. **★最大の懸念(ユーザー指摘と同じ): 「それらしく動くが科学的に無意味」になるリスク。**
   `torque.py:apply()`を見ると、実際の力は「圧力(delay+hysteresis付き)」**と**
   「その瞬間の関節角`q`(=腕の実際の運動)」の**両方**に依存する結合系
   (`h = calculate_effective_contraction(q, ...)`)。今回の設計(§2簡略化A)は
   「未来の`q`を予測せず今の`q`で力マップを引く」ため、**打撃タイミングの成否は
   腕自体の力学(慣性・反発・接触)にも左右され、FFは圧力側しか直接制御していない**。
   → 単発打で崩れず連打で崩れた場合、それが「記憶・先読みの必要性」を示すのか、
   「幾何結合を無視した近似の限界」を示しているだけなのかは、**崩れ方の質的検証
   (どのタイミング誤差がいつ発生するか、力の立ち上がり波形)なしには区別できない**。
   本文に載せる前に、この切り分け自体に追加の考察・検証時間が要る。
2. **ヒステリシス2Dテーブルの逆引きは単純な反転ではない**
   (`P_cmd`が入力軸自身でもある暗黙方程式、§2参照)— 固定点反復は収束するはずだが、
   反復回数・収束判定・発散時のフォールバックの実装/検証が要る
   (「単純にテーブルをひっくり返せば終わり」ではない、ユーザー懸念通り)。
3. **むだ時間`L`の状態依存性を厳密に逆算していない**(Smith predictor近似)。
   `L`はP軸で0.023〜0.131s程度まで変動しうる(pneumatic.py:15-16, 20-27)ため、
   固定値近似の誤差が高BPM/複雑リズムで無視できなくなる可能性がある。
4. **FFゲイン調整の要否・所要時間が読めない。**
   解析的逆算に理論上は自由なゲインはないが、(a)固定点反復の打ち切り誤差、
   (b)簡略化A/Bの近似誤差、(c)方向反転検出のチャタリング、を吸収するための
   安全マージン(オーバーシュート防止のスケールダウン等)は実機・実装後に
   「打ってみて見た目で」調整することになりやすく、これは本質的に
   見積もりにくい(§0・§4の最大のブレ要因)。
5. **DR課題(`Template-Porcaro-2026-ModelB-DR-user0`)で評価する場合、
   FFのシャドウモデルは実際にサンプリングされたτスケール(±20%)を知らない**
   (`pam.py:225-229`の`reset_idx()`でランダムに決まり、`env.unwrapped`に
   アクセスしない設計なら原理的に不可知)。これはRLモデルが学習中にDRへ
   適応できるのと対照的で、「FFはDRに弱い」という結果が出てもそれ自体は
   想定通り・興味深い側面(DR比較実験の文脈で語れる)であり欠陥ではないが、
   本文での位置づけを誤ると「不公平な比較」と読まれる可能性がある。
6. **num_envs=1評価で足りるか**: 足りる。既存の全play_sim_*.py評価ジョブが
   `--num_envs 1`前提であり(run_eval_matrix.py各所)、FFもこれに合わせれば
   既存のstrike抽出パイプラインとの整合が取れる。ここは新規リスクではない。

---

## 4. 工数見積もり

前提: 「学習は不要、eval実行のみ」(§4末尾で確認)。人手見積もりは
このコードベースに既に精通した人(このセッションの調査結果を前提にできる人)の場合。

| フェーズ | 内容 | 人手 | Claude Code併用時の人の実作業 |
|---|---|---|---|
| 0. 逆算モジュール単体実装+単体検証(CPU、IsaacLab起動なし) | `ff_controller.py`、pneumatic.py/pam.py経由の順方向再生でのオフライン検算 | 1.5〜2.5日 | 0.5〜1日 |
| 1. ポリシー差し込み(2ファイルへの分岐追加) | `--ff_baseline`分岐、ダミーrun_tag/ckpt_name | 0.5〜1日 | 2〜4時間 |
| 2. 単発/基本パターンでの妥当性検証・ゲイン調整 | num_envs=1実eval、simulation_log.csv目視、反復チューニング | **0.5〜1.5日(最大のブレ)** | 1〜2時間(スクリプト面のみ、判断は人手) |
| 3. eval/集計パイプライン統合 | `run_eval_ff_baseline.py`, `aggregate_ff_baseline.py` | 0.5日 | 1〜2時間 |
| 4. 本eval実行+結果の質的検証・考察 | double_160/gmd_03/gmd_04 × 評価seed数本、GPU eval(軽量) | <0.5日(大半は待ち) | — |
| **合計** | | **約3.5〜6人日** | **約1.5〜3日相当** |

**GPU/計算の要否**: 新規学習(train.py実行)は不要 — 確認済み
(FFは非学習コントローラで、既存の学習済みチェックポイントに相当するものを
必要としない)。必要なのは既存`play_sim_rhythm.py`/`play_sim_midi.py`と
同オーダーの**eval実行のみ**(num_envs=1、既存A-Eモデルのevalと同程度の
軽量ジョブ、docs/monday_run.mdの既存eval実測値から数分/ジョブ程度と推測)。
Phase 0の逆算モジュール単体検証は**GPU/IsaacLab起動すら不要**
(pneumatic.py/pam.pyがisaaclab非依存の純torchのため)。

---

## 5. go / no-go の推奨と理由

**推奨: 今回のRA-L改訂サイクルではno-go(見送り、"今後の課題"として本文に明記)。**

理由:

1. 最短見積もり(Claude Code併用)でも人の実作業約1.5〜3日、素の人手なら
   3.5〜6日規模で、締切7月末までの残り作業(本文修正、他指摘対応)と
   比較して軽くない。
2. §3-1の「それらしく動くが科学的に無意味」リスクが、単なる実装バグの心配
   ではなく**設計上の簡略化(現在q前提、Smith predictor近似)に起因する
   構造的なもの**であり、これを切り分ける追加の質的検証が工数表(Phase 2/4)の
   外側にさらに必要になりうる。締切直前にこの切り分けが間に合わなかった場合、
   「結果を載せたが解釈が弱い」図をRA-Lに出すことになり、むしろ査読リスクを
   増やしかねない。
3. Phase 2(ゲイン調整)の所要時間が本質的に見積もり不能で、締切直前に
   工数超過するリスクの大部分がここに集中している。
4. 一方で、R3 Weakness-3aの片側(「LSTM-free MLPでhistoryを見る比較が無い」)は
   既にModel E(framestack MLP)で対応済み(`analysis/summary/report.html`参照)
   — 本baselineは残るもう片側(「model-basedフィードフォワードとの比較」)
   のみに対応するものであり、今回見送っても指摘への対応が全くのゼロにはならない。
5. §1で確認した通り、再利用可能な部品(pneumatic.py/pam.pyの無改造import、
   obs経由の目標・幾何情報、既存ドライバ/集計パターンの流用)により
   **将来やる場合の初期コストは既にかなり下がっている** — 今回見送っても、
   次回(次の投稿サイクルやカメラレディ、または追加rebuttal)に着手する際は
   本ドキュメントがそのまま設計書として使える。

本文/rebuttalでの書き方の候補: 「a model-based feedforward baseline using the
identified delay/hysteresis tables is a natural comparison and is left as future
work; our ablations (framestack-MLP memory isolation, far-future masking,
tau-sweep) address the remaining aspects of this critique within the current
scope」といった形で、今回のE/masking/tauスイープでの部分対応と、
FFが未対応である点を明示的に切り分けて書くのが誠実。

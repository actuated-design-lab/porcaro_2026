# マジックナンバー洗い出しレポート

対象論文: *Temporal Anticipation and Memory Enable High-Speed Sim-to-Real Drumming in a
Delay-Dominant Pneumatic Musculoskeletal Percussion Robot*

対象コード: `source/porcaro_2026/porcaro_2026/tasks/direct/porcaro_2026/` 以下

> `user0` と `user1` はほぼ同一構造（実質差分は `lookahead_horizon` と `observation_space` の扱いのみ）。
> 共通ロジックは1回にまとめ、差分がある箇所のみ明記した。

---

## A. PPO/RSL-RLハイパーパラメータ

`rsl_rl_ppo_lstm_cfg.py` / `rsl_rl_ppo_mlp_cfg.py` / `rsl_rl_ppo_cfg.py`（3ファイルとも同一値）

| パラメータ | 値 | 論文記載 | 備考 |
|---|---|---|---|
| `learning_rate` | 1.0e-4 | なし | PPOハイパラの数値は論文非記載（PPO[19]使用としか言及なし） |
| `gamma` | 0.99 | なし | |
| `lam` (GAE λ) | 0.95 | なし | |
| `clip_param` | 0.2 | なし | |
| `entropy_coef` | 0.002 | なし | |
| `num_learning_epochs` | 5 | なし | |
| `num_mini_batches` | 4 | なし | |
| `desired_kl` | 0.02 | なし | |
| `max_grad_norm` | 1.0 | なし | |
| `value_loss_coef` | 1.0 | なし | |
| `init_noise_std` | 0.5 | なし | |
| `actor_hidden_dims` / `critic_hidden_dims` | [256, 128, 64] | なし | |
| `rnn_hidden_dim` | 128 | **あり** | 論文 "LSTM policy (128 hidden units)" と一致 |
| `rnn_num_layers` | 1 | なし | |
| `num_steps_per_env` | 120 | なし | ロールアウト長。論文に記載なし |
| `max_iterations` | 1500 | **あり** | 論文 "1,500 iterations" と一致 |
| `save_interval` | 50 | なし | |

## B. カリキュラム学習・訓練スケール

`porcaro_2026_env.py`, `rhythm_generator.py`

| 場所 | 値 | 論文記載 | 備考 |
|---|---|---|---|
| `curriculum_thresholds` (env.py:60) | `[25_000_000, 100_000_000]` | **あり** | 論文 "Level 0 up to 25M steps, Level 1 up to 100M steps" と一致 |
| `max_bpm_idx_per_level` (rhythm_generator.py:24) | `[1, 3, 5]`（BPM 80/120/160が上限） | **あり** | 論文 "Level0: 60–80BPM / Level1: up to120BPM / Level2: up to160BPM" と一致 |
| `bpm_options` | `[60,80,100,120,140,160]` | **あり** | 論文と一致 |
| curriculum確率分布 (Lv0/1/2) | `[0.4,0.45,0.05,0.1]` / `[0.2,0.4,0.3,0.1]` / `[0.2,0.2,0.5,0.1]` | なし | パターン別サンプリング確率は論文非記載 |
| `scene.num_envs` (env_cfg.py既定値) | 32 | **不一致注意** | 実訓練は2048（`run_experiment_matrix.py --num_envs 2048`, 論文"2,048 parallel environments"）で上書き。cfgのデフォルト値自体は論文値と異なる |
| env.py内コメントの総ステップ数 (line 56-58) | "総ステップ数737280000 / 1iterationあたり246043steps" | **論文の≈3×10⁹と不一致** | `1500 iter × 2048env × 120steps ≈ 3.78×10⁸` となり、コード内コメントの数字とも、論文記載の"≈3×10⁹"とも合わない。要確認 |

## C. 報酬関数

`user*/cfg/rewards_cfg.py` + `user*/rewards/reward.py`

| パラメータ | 値 | 論文記載 | 備考 |
|---|---|---|---|
| `weight_match` (w_match) | 1.0 | **あり** | 重み一致 |
| `weight_rest` (w_rest) | 0.005 | **あり** | 一致 |
| `weight_contact_continuous` (w_push) | -0.2 | **あり** | 一致 |
| `w_miss` (reward.py L243, getattr default) | -1.0 | **あり** | 一致（RewardsCfgには`weight_miss`未定義でgetattrのdefaultのみ使用） |
| `w_double` (weight_double_hit) | -0.5 | **あり** | 一致 |
| `weight_grip_penalty` (w_grip) | -0.5 | **あり** | 一致 |
| Eq.(7) `(c1,c2,c3)` (reward.py:179 `0.2 + 0.4*mag + 0.4*acc`) | (0.2, 0.4, 0.4) | **あり** | 論文式と完全一致 |
| `target_force_fd` (F_ref) | 20.0 | **あり** | 一致 |
| `sigma_force` (σ_f) | 15.0 | なし | 論文はσ_fの記号のみで数値未記載 |
| `rest_threshold = target_ref_val * 0.10` (reward.py:21) | 10% | **あり** | 本文 "target force is less than 10% of Fref" と一致 |
| `dyn_max_contact = t_16th * 0.40` | 0.4×T16th | **あり** | "exceeds 0.4 T16th" と一致 |
| `dyn_cooltime = t_16th * 0.25` | 0.25×T16th | **あり** | "within 0.25 T16th" と一致 |
| `T16th = 15.0 / safe_bpm` | 15/BPM | **あり** | 論文式と完全一致 |
| `dyn_impact_window = t_16th * 0.40` | 0.4×T16th | なし | 論文に対応する定義なし（独自追加） |
| `dyn_miss_thresh = t_16th * 2.00` | 2.0×T16th | なし | 論文非記載 |
| `weight_rest_penalty` | -0.01 | なし | 論文はw_restのみでrest penalty別項は非記載 |
| `weight_wrist_co_contract` | 0.0（無効化） | なし | 論文の列挙報酬項に無し（実験的追加、現在オフ） |
| `weight_joint_limits` | 0.0（無効化） | なし | 未使用 |
| `swing_amplitude_threshold_deg` | 0.0（コメントに"10度"の記述あり） | なし | 論文非記載。コード内コメントと実値も不一致（0.0 vs 想定10°） |
| `hit_threshold_force` (getattr default) | 1.0 | なし | |
| force_excess deadband (reward.py:216) | `(force_z - 1.0).clamp(min=0)` / `10.0` | なし | 休符中の許容量。論文非記載 |

## D. PAM物理定数

`controller_cfg.py`, `common/actions/pam.py`, `pneumatic.py`, `torque.py`

| 場所 | 値 | 論文記載 | 備考 |
|---|---|---|---|
| `Pmax` (controller_cfg.py) | 0.6 MPa | **あり** | "[0, 0.6] MPa" と一致 |
| **`force_scale`** (controller_cfg.py:24) | **0.2** | **論文と不一致** | 論文Eq.(4-5)は "$k_F = 0.3$ was determined via hysteresis identification" と明記。コードは0.2。要確認（意図的変更か記載ミスか） |
| `force_scale_sim_to_real` (env.py:63) | 3.0 | **あり** | 論文 "impact force scaling factor of $k_t = 3.0$" と完全一致 |
| `pam_contract_gain` (torque.py:38, C(1)) | 1.5 | **あり** | 論文 "C(1) = 1.5 for inflation" と一致 |
| `pam_extend_gain` (torque.py:39, C(-1)) | 1.0 | **あり** | 論文 "C(−1) = 1.0 for deflation" と一致 |
| `pam_hys_const`, `pam_hys_coef_p` (torque.py:28-29) | 0.5, 15 | なし | 論文の$H_{const}, H_{coef}$は数値未公開 |
| `r` (プーリー半径) | 0.014 m | なし | 記号のみで数値は論文非記載 |
| `L` / `natural_length` (L0) | 0.150 m | なし | 同上 |
| `theta_t_DF/F/G_deg` | 7.0 / 70.0 / 45.0 | なし | 論文はθ_{t,i}を記号のみ、数値なし |
| `tau` (基準時定数, controller_cfg) | 0.09 s | 部分的 | 論文本文 "dominant pneumatic time constant (≈100 ms)" と近似だが完全一致ではない |
| `dead_time` | 0.03 s | なし | |
| `N` (簡易式係数, 未使用寄り) | 630.0 | なし | CSVがあれば不要と明記されており、論文でも言及なし |
| `TAU_TAB` (P=0.1〜0.6MPa) | [0.043,0.045,0.060,0.066,0.094,0.131] | 概念のみ | Eq.(1)の$\tau(P_{cmd},P_{start})$の実装だが、テーブル値自体は論文非公開 |
| `L_TAB` | [0.038,0.035,0.032,0.030,0.023,0.023] | 概念のみ | 同上（dead time $L$） |
| `TAU_TABLE_2D_DATA` / `DEAD_TABLE_2D_DATA` (7×7) | 多数の数値 | 概念のみ | "state-dependent... determined online via interpolation" に対応する実装だが数値非公開 |
| `pam_p_dot_scale` | 100 | なし | Eq.(3)の$\dot P$正規化。論文非記載 |
| `FractionalDelay.L_max` | 0.20 s | なし | 実装上の遅延バッファ上限のみ |
| `PAMChannel.deadband` | 1.0e-4 | なし | チャタリング防止用の実装的しきい値 |
| `pam_tau_scale_range` (Model_DR) | (0.8, 1.2) | **あり** | Table I "±20%" および本文 "0.8–1.2 times its nominal value" と一致 |

## E. リズム生成器

`rhythm_generator.py`

| パラメータ | 値 | 論文記載 | 備考 |
|---|---|---|---|
| カーネル `width_sec` | 0.035 s → `sigma = width_sec/2 = 0.0175` | **あり** | 論文 "σ = 17.5 ms" と一致 |
| カーネル指数 | 4乗 (`** 4`) | **あり** | "order = 4" と一致（super-Gaussian） |
| ルーディメント数 (single_4/8/double) | 6/12/24 strikes相当のオフセット配列 | **あり** | 論文の "6, 12, 24 strikes" に対応するオフセット定義と一致 |
| バー数 | 4小節 | **あり** | "four measures" と一致 |
| 1小節目=count-in (rest) | 固定 | **あり** | "first measure as a full rest" と一致 |
| `rudiments` オフセット配列そのもの (例: double=[0,1,4,5,8,9,12,13]) | — | 概念のみ | 具体的な16分音符グリッド実装値は論文非記載（自然な帰結だが数値までは書かれていない） |

## F. 環境/シム設定

`porcaro_2026_env_cfg.py`, `assets.py`, `sensors.py`

| パラメータ | 値 | 論文記載 | 備考 |
|---|---|---|---|
| `decimation=4`, `sim.dt=1/200` | 制御周期50Hz | **あり** | 論文 "50 Hz high-level" / "200 Hz low-level" と一致 |
| `episode_length_s` | 100.0 | なし | 論文に総エピソード長の数値記載なし（実際はBPM依存で16拍分に動的設定） |
| `physics_material` (static/dynamic friction=0.0, restitution=0.8) | — | なし | 論文非記載 |
| `lookahead_horizon` (user1既定=0.1 / user0既定=0.5) | 0.1s / 0.5s | **あり** | 論文の0.1s(Model D)・0.5s(Model A/C) と対応する値。ただし0.1sと1.0s(Model B)用の明示的なcfgクラスはコード中に見当たらず、`__init__.py`にModel B(0.5s相当)のみgym登録 |
| `observation_space` (user1固定値=15) | 15 | 式はあり | 論文式 $o_t\in\mathbb R^{10+k}$、k=5(0.1s@50Hz)で15次元、一致 |
| `bpm_range` | (60.0, 160.0) | **あり** | 論文と一致 |
| `apply_domain_randomization`: mass `(0.95,1.05)` | ±5% | **あり** | Table I "Mass & Inertia ±5%" 一致 |
| 同 friction `(0.95,1.05)` | ±5% | **あり** | Table I "Joint Friction ±5%" 一致 |
| 同 `restitution_range=(0.0,0.0)` | — | なし | Table Iに対応行なし（反発係数は固定運用らしいが論文未記載） |
| `reset_joints position_range=(0.95,1.05)` | ±5% | 部分的 | 本文 "small offsets in initial joint positions" と定性的一致だが数値(±5%)は論文非記載 |
| `contact_offset=0.02` (stick/drum両方) | 20mm | なし | 実装コメント "打撃感がフワフワします" のみ、論文非記載 |
| `DRUM_CFG mass=1.0e2` | 100kg | なし | 論文非記載 |
| `WRIST_J0=0°`, `GRIP_J0=-8.1°` | — | なし | 初期姿勢。論文非記載 |
| ロボット設置 `yaw_deg=30` | — | なし | シーン配置のみ |
| `ImplicitActuatorCfg` damping/effort_limit/friction (wrist: 0.1/500/0.001, grip: 0.001/500/0.00001) | — | なし | Isaac Lab側の数値ダンパ/摩擦。論文非記載 |
| `ContactSensorCfg.history_length=32` | — | なし | 実装上のバッファ長 |

## G. 特に確認を推奨する不一致

1. **`force_scale=0.2`（controller_cfg.py:24) vs 論文 $k_F=0.3$** — Eq.(4)(5)で明示された値と食い違っている。実験的にチューニングし直したのか、論文記載が古いのか要確認。
2. **総学習ステップ数の不一致** — env.py内コメント「737,280,000」/「246,043 steps/iteration」と、`num_steps_per_env=120 × num_envs=2048 × max_iterations=1500 ≈ 3.78×10⁸`、および論文本文の「≈3×10⁹ total steps」の三者が互いに一致しない。
3. **`swing_amplitude_threshold_deg=0.0`** — コード内コメントは「10度をギリギリの閾値とする」と書いてあるが実際値は0.0。コメントと実装が乖離。
4. **user0/user1のlookahead既定値の食い違い** — user0=0.5s、user1=0.1sとハードコードされており、どちらがどのModel(A/B/C/D)に対応するのか`__init__.py`のgym登録からは判別しづらい（両方とも"ModelB"としてのみ登録、lookahead差分を明示するgym IDが存在しない）。

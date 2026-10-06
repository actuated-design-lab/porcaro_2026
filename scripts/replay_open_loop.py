"""
replay_open_loop.py — 実機と同じ圧力指令をシミュレータに開ループで流し、圧力と関節角を記録する

JFPS 柱③「作り直した圧力モデルをシミュレータに入れて、実機の動作を再現できるか」用。
RA-L で使った user0 環境（Porcaro2026EnvCfg_ModelB, DR なし）をそのまま使い、
方策の代わりに CSV の圧力指令を行動に変換して入れる。

置き場所: porcaro_2026/scripts/replay_open_loop.py

圧力モデル（--pmodel）:
  table    : 今のシミュレータ（始点圧×指令圧の 2 次元テーブル, pneumatic.py）
  lag      : 対称な一次遅れ＋むだ時間（9/29–30 のエコー付きデータで同定）
  orifice  : lag に、給気・排気のオリフィス流量の上限を足したもの（同上）
  measured : 実機で測った圧力をそのまま入れる（--real_log 必須）。
             「圧力→角度」だけを比べ、機構側（力マップ・摩擦・たるみ）の誤差を切り分ける
             実測圧力は移動平均（--meas_lpf_ms, 既定 50 ms）と遊び（--meas_play_kpa, 既定 ±10 kPa）を
             かけてから入れる。生のままだと圧力ノイズでヒステリシス力の向きが毎ステップ反転し、
             sim 側のヒステリシスが実質消える

機械側の上書き（--ctrl）:
  cfg.controller の項目を JSON で上書きする（force_scale は (DF, F, G) の組も可）。
  例: --ctrl '{"force_scale":[0.2,0.15,0.2],"theta_t_F_deg":60}'

入力:
  --signal  test_signals/tm_*.csv（time, cmd_pressure_DF, cmd_pressure_F, cmd_pressure_G, 50 Hz）
  --real_log  measured モードのときの実機ログ（run_signal_playback.py の出力）
              実機ログは DF/F の圧力列が入れ替わっている（2026/9/29 判明）ので、ここで戻す。
              時刻は flag 列（MicroLabBox が受け取った DF 指令のエコー）で指令に合わせる。

出力（--out, 200 Hz, 物理ステップごと）:
  time, P_cmd_DF/F/G, P_out_DF/F/G, wrist_angle_deg, grip_angle_deg, force_N
  角度は実機と同じ向き（Up+）に直したもの。力はスティック接触力の大きさ（×3.0 は環境と同じ）。

使い方（Isaac Lab の python で）:
  python scripts/replay_open_loop.py --signal ../jetson_project/test_signals/tm_C_sine.csv \
      --pmodel table --no_drum --headless --out out/sim_tm_C_sine_table.csv
  python scripts/replay_open_loop.py --signal ... --pmodel measured \
      --real_log ../jetson_project/test_signals/data_tm_C_sine_XXXX.csv --no_drum --headless --out ...
"""
from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description="開ループ再生（圧力指令 → シミュレータ）")
p.add_argument("--signal", required=True)
p.add_argument("--pmodel", choices=["table", "lag", "orifice", "measured"], default="table")
p.add_argument("--real_log", default=None)
p.add_argument("--params", default=None,
               help="lag/orifice のパラメータを JSON で上書き（例: '{\"tau\":0.09,\"L\":0.045}'）")
p.add_argument("--no_drum", action="store_true", help="打面を横へ 2 m 退避（打面なしの実機試験と揃える）")
p.add_argument("--ctrl", default=None,
               help="機械側（張力・関節）の設定を JSON で上書き。cfg.controller の項目名で指定。"
                    "例: '{\"force_scale\":[0.2,0.15,0.2],\"theta_t_F_deg\":60,\"pam_hys_const\":1.0}'")
p.add_argument("--meas_lpf_ms", type=float, default=50.0,
               help="measured モードで実測圧力にかける移動平均の幅 [ms]（中心合わせ・遅れなし）。0 で無効")
p.add_argument("--meas_play_kpa", type=float, default=10.0,
               help="移動平均のあとにかける遊び（バックラッシュ）の半幅 [kPa]。0 で無効。"
                    "生のままだと圧力ノイズでヒステリシス力の向き d が毎ステップの56%%で反転する（2026/10/6 判明）。"
                    "既定の 50 ms + ±10 kPa で反転 0.4%%/step（tm_B で確認）")
p.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

# ---- ここから Isaac Lab のモジュールを import できる ----
import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
import torch                  # noqa: E402

from porcaro_2026.tasks.direct.porcaro_2026.user0.porcaro_2026_env import Porcaro2026Env        # noqa: E402
from porcaro_2026.tasks.direct.porcaro_2026.user0.porcaro_2026_env_cfg import Porcaro2026EnvCfg_ModelB  # noqa: E402

PA = 0.1013  # 大気圧 [MPa]

# 9/29–30 のエコー付きデータ（通信を含まない）で同定した値。analysis/compare_pressure_models.py
DEFAULT_PARAMS = {
    "lag": {"tau": 0.0836, "L": 0.0446},
    "orifice": {"tau": 0.0856, "L": 0.0442, "c_in": 5.17, "c_out": 11.33, "ps": 0.578, "b": 0.053},
}


# =============================================================================
# 圧力チャネル（PAMChannel と同じインタフェース: reset / reset_idx / step）
# =============================================================================
class _DelayBuf:
    def __init__(self, dt, L):
        self.dt, self.L = dt, float(L)
        self.D = self.L / dt
        self.M = int(math.floor(self.D)); self.mu = self.D - self.M
        self.K = self.M + 3
        self.buf = None; self.wp = 0

    def reset(self, shape, device):
        self.buf = torch.zeros((self.K, *shape), device=device); self.wp = 0
        self._primed = False

    def step(self, x):
        if not self._primed:           # 最初の値で埋める（開始時の段差を作らない）
            self.buf[:] = x; self._primed = True
        self.buf[self.wp] = x
        a = self.buf[(self.wp - self.M) % self.K]
        b = self.buf[(self.wp - self.M - 1) % self.K]
        self.wp = (self.wp + 1) % self.K
        return (1 - self.mu) * a + self.mu * b


def _g(pu, pd, b):
    """ISO 6358 型の流量関数（pu 上流・pd 下流, 絶対圧）"""
    r = torch.clamp(pd / pu, max=1.0)
    x = torch.clamp((r - b) / (1 - b), min=0.0)
    return torch.where(r <= b, pu, pu * torch.sqrt(torch.clamp(1 - x * x, min=0.0)))


class LagChannel:
    def __init__(self, dt, tau, L, Pmax=0.6):
        self.dt, self.tau, self.Pmax = dt, tau, Pmax
        self.delay = _DelayBuf(dt, L); self.p = None

    def reset(self, n_envs, device):
        self.delay.reset((n_envs,), device); self.p = torch.zeros(n_envs, device=device)

    def reset_idx(self, env_ids):
        if self.p is not None:
            self.p[env_ids] = 0.0

    def step(self, P_cmd):
        u = self.delay.step(torch.clamp(P_cmd, 0.0, self.Pmax))
        self.p = self.p + self.dt / (self.tau + self.dt) * (u - self.p)
        return self.p


class OrificeChannel(LagChannel):
    def __init__(self, dt, tau, L, c_in, c_out, ps, b, Pmax=0.6):
        super().__init__(dt, tau, L, Pmax)
        self.c_in, self.c_out, self.ps, self.b = c_in, c_out, ps, b

    def step(self, P_cmd):
        u = self.delay.step(torch.clamp(P_cmd, 0.0, self.Pmax))
        q = (u - self.p) / self.tau
        pab = self.p + PA
        lim_in = self.c_in * _g(torch.full_like(pab, self.ps + PA), pab, self.b)
        lim_in = torch.where(self.ps + PA > pab, lim_in, torch.zeros_like(lim_in))
        lim_out = self.c_out * _g(pab, torch.full_like(pab, PA), self.b)
        lim_out = torch.where(pab > PA, lim_out, torch.zeros_like(lim_out))
        q = torch.where(q > 0, torch.minimum(q, lim_in), torch.maximum(q, -lim_out))
        self.p = self.p + q * self.dt
        return self.p


class MeasuredChannel:
    """実機で測った圧力を 1 物理ステップ（5 ms）ずつ返す"""
    def __init__(self, seq):
        self.seq = np.asarray(seq, np.float32); self.i = 0; self.dev = "cpu"

    def reset(self, n_envs, device):
        self.i = 0; self.n = n_envs; self.dev = device

    def reset_idx(self, env_ids):
        pass

    def step(self, P_cmd):
        v = float(self.seq[min(self.i, len(self.seq) - 1)]); self.i += 1
        return torch.full((self.n,), v, device=self.dev)


# =============================================================================
# 実機ログの読み込み（measured モード）
# =============================================================================
def smooth_centered(x, width_ms, dt):
    """中心合わせの移動平均（遅れなし）。端は端の値で延長"""
    k = int(round(width_ms / 1000.0 / dt))
    if k <= 1:
        return x
    k += (k + 1) % 2                                  # 奇数に
    pad = k // 2
    xp = np.r_[np.full(pad, x[0]), x, np.full(pad, x[-1])]
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def play_operator(x, half_width):
    """遊び（バックラッシュ）：入力が ±half_width の帯の中で揺れている間は出力を保つ"""
    if half_width <= 0:
        return x
    y = np.empty_like(x); y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = min(max(y[i - 1], x[i] - half_width), x[i] + half_width)
    return y


def load_real_aligned(path, cmd_50hz, dt):
    """実機ログを、エコー（flag）で指令の時間軸に合わせ、dt 刻みの DF/F/G 圧力列にして返す"""
    d = pd.read_csv(path)
    P = np.stack([d["meas_pres_F"].values, d["meas_pres_DF"].values, d["meas_pres_G"].values], 1)  # 入れ替わりを戻す
    flag = np.nan_to_num(d["flag"].values.astype(float))
    t_real = np.arange(len(d)) * 0.005
    t_cmd = np.arange(len(cmd_50hz)) * 0.02
    # 指令DF（50 Hz, ZOH）を 200 Hz に展開
    cmd_df = cmd_50hz[np.clip((t_real / 0.02).astype(int), 0, len(cmd_50hz) - 1), 0]
    best = (1e9, 0)
    for lag in range(-100, 400):                    # 実機ログ上で flag が指令より何行ずれているか
        if lag >= 0:
            a = cmd_df[: len(cmd_df) - lag]; b = flag[lag: lag + len(a)]
        else:
            b = flag[: len(flag) + lag]; a = cmd_df[-lag: -lag + len(b)]
        n = min(len(a), len(b))
        if n < 100:
            continue
        e = np.mean((a[:n] - b[:n]) ** 2)
        if e < best[0]:
            best = (e, lag)
    lag = best[1]
    print(f"[real] エコーで合わせたずれ: {lag * 5} ms（flag が指令に一致する位置）")
    t_sim = np.arange(int(round(t_cmd[-1] / dt)) + 1) * dt
    out = np.stack([np.interp(t_sim + lag * 0.005, t_real, P[:, k]) for k in range(3)], 1)
    return out


# =============================================================================
def main():
    sig = pd.read_csv(args.signal)
    cmd = sig[["cmd_pressure_DF", "cmd_pressure_F", "cmd_pressure_G"]].values.astype(np.float32)
    n_steps = len(cmd)

    cfg = Porcaro2026EnvCfg_ModelB()
    cfg.scene.num_envs = 1
    cfg.events = None                       # DR なし
    cfg.pam_tau_scale_range = (1.0, 1.0)
    cfg.logging.enabled = False
    cfg.episode_length_s = n_steps * 0.02 + 10.0   # 途中でエピソードが切れないように
    ctrl_over = json.loads(args.ctrl) if args.ctrl else {}
    for k, v in ctrl_over.items():
        if not hasattr(cfg.controller, k):
            raise SystemExit(f"--ctrl: cfg.controller に '{k}' はありません")
        setattr(cfg.controller, k, tuple(v) if isinstance(v, list) else v)
    if ctrl_over:
        print(f"[replay] 機械側の上書き: {ctrl_over}")
    if args.no_drum:
        x, y, z = cfg.drum_cfg.init_state.pos
        cfg.drum_cfg.init_state.pos = (x + 2.0, y, z)   # 下へ動かすと床に埋まるので横へ

    env = Porcaro2026Env(cfg)
    dt_phys = cfg.sim.dt
    Pmax = env.action_controller.Pmax
    ctrl = env.action_controller

    # ---- 圧力モデルの差し替え ----
    prm = dict(DEFAULT_PARAMS.get(args.pmodel, {}))
    if args.params:
        prm.update(json.loads(args.params))
    if args.pmodel == "lag":
        ctrl.ch_DF, ctrl.ch_F, ctrl.ch_G = (LagChannel(dt_phys, prm["tau"], prm["L"], Pmax) for _ in range(3))
    elif args.pmodel == "orifice":
        ctrl.ch_DF, ctrl.ch_F, ctrl.ch_G = (OrificeChannel(dt_phys, prm["tau"], prm["L"], prm["c_in"], prm["c_out"],
                                                           prm["ps"], prm["b"], Pmax) for _ in range(3))
    elif args.pmodel == "measured":
        if not args.real_log:
            raise SystemExit("--pmodel measured には --real_log が必要です")
        Pm = load_real_aligned(args.real_log, cmd, dt_phys)
        Pm = np.stack([play_operator(smooth_centered(Pm[:, k], args.meas_lpf_ms, dt_phys), args.meas_play_kpa / 1000.0)
                       for k in range(3)], 1)
        ctrl.ch_DF, ctrl.ch_F, ctrl.ch_G = (MeasuredChannel(Pm[:, k]) for k in range(3))
    ctrl.reset(env.num_envs, env.device)
    print(f"[replay] pmodel={args.pmodel} params={prm} steps={n_steps} ({n_steps * 0.02:.1f} s) no_drum={args.no_drum}")

    # ---- 物理ステップごとの記録（_apply_action の直後 = そのステップの圧力と、その時点の関節状態） ----
    rec = []
    orig_apply = env._apply_action

    def patched_apply():
        orig_apply()
        tel = ctrl.get_last_telemetry()
        q, _ = env._get_corrected_joint_state()
        fv = env.stick_sensor.data.net_forces_w
        fv = fv[:, 0, :] if fv.dim() == 3 else fv
        f = torch.norm(fv * env.force_scale_sim_to_real, dim=-1)
        rec.append(np.r_[tel["P_cmd"][0].cpu().numpy(), tel["P_out"][0].cpu().numpy(),
                         np.degrees(q[0].cpu().numpy()), f[0].item()])

    env._apply_action = patched_apply

    env.reset()
    ctrl.reset(env.num_envs, env.device)                 # reset 後にチャネルを初期化し直す
    big = torch.full_like(env.episode_duration_steps, 10 ** 9)
    for k in range(n_steps):
        env.episode_duration_steps[:] = big              # BPM 由来のタイムアウトを無効化
        a = torch.as_tensor(cmd[k] / Pmax * 2.0 - 1.0, device=env.device).view(1, 3)
        env.step(a)

    r = np.array(rec)
    out = pd.DataFrame(r, columns=["P_cmd_DF", "P_cmd_F", "P_cmd_G", "P_out_DF", "P_out_F", "P_out_G",
                                   "wrist_angle_deg", "grip_angle_deg", "force_N"])
    out.insert(0, "time", np.arange(len(out)) * dt_phys)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    meta = dict(signal=os.path.abspath(args.signal), pmodel=args.pmodel, params=prm, no_drum=args.no_drum,
                real_log=args.real_log, dt=dt_phys, env_cfg="Porcaro2026EnvCfg_ModelB (DRなし)",
                ctrl=ctrl_over,
                meas_filter=(dict(lpf_ms=args.meas_lpf_ms, play_kpa=args.meas_play_kpa) if args.pmodel == "measured" else None))
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"[replay] 保存: {args.out}  ({len(out)} 行)")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()

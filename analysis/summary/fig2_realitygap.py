"""fig2_realitygap.py — 本文 Fig.2（1段幅）: シミュレータと実機のリアリティギャップ。

  (a) 準静的ヒステリシスループ（角度 vs 内圧）
  (b) 非対称ステップ応答（角度・内圧の時系列）
  (c) 掃引周波数（チャープ）入力に対する時間応答（角度・内圧の時系列）

データは IROS 期（2026-02）のモデル検証実験そのもの。同一の指令信号を実機と
シミュレータに与え、それぞれ独立に記録したログを重ねている。

  paper_data/validation/real/data_exp{1,2,3}_*.csv   実機（200 Hz）
  paper_data/validation/sim/sim_log_ModelB_exp{1,2,3}_*.csv  sim（50 Hz）

★配色は論文全体の規約に従う: **暖色 = 実機 / 寒色 = シミュレーション**。
  Fig.4/5 の「青 = Simulation、橙 = Hardware」と同じ意味付けにしてある。
  旧版（PowerPoint 由来）は角度が赤/青、内圧が緑/サーモンで、しかも橙系が
  sim を指しており Fig.4/5 と逆転していた。ここで解消する。

★旧版は (b) だけ左右二軸（角度と内圧の重ね描き）、(c) は上下2段と、同じ2量の
  見せ方が図の中で揺れていた。全パネルで「角度が上・内圧が下」に統一し、
  二軸を使わない。二軸は軸スケールの取り方で印象を操作できるという批判を
  受けやすく、査読上も避けたい。

Usage:
  python analysis/summary/fig2_realitygap.py --outdir ../RALpaper/figures
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from ral_figstyle import (  # noqa: E402
    INK, INK_MUTED, OKABE_ITO, apply_style, base_argparser, save, tidy,
)

DATA = REPO_ROOT / "paper_data" / "validation"

C_REAL = OKABE_ITO["vermillion"]   # 実機 = 暖色（Fig.4/5 の Hardware と同じ）
C_SIM = OKABE_ITO["blue"]          # sim  = 寒色（Fig.4/5 の Simulation と同じ）
C_CMD = INK_MUTED                  # 指令値は中立インク

STEP_T = 6.0     # (b) の表示窓 [s]。立ち上がりと解放の非対称が両方入る長さ
SWEEP_T = 15.0   # (c) の表示窓 [s]


def load(exp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """同一の指令信号に対する実機ログと sim ログを、時刻を 0 起点に揃えて返す。"""
    r = pd.read_csv(sorted(glob.glob(str(DATA / "real" / f"data_{exp}_*.csv")))[-1])
    s = pd.read_csv(sorted(glob.glob(str(DATA / "sim" / f"sim_log_ModelB_{exp}_*.csv")))[-1])
    r["t"] = r["time"] - r["time"].iloc[0]
    s["t"] = s["time"] - s["time"].iloc[0]
    return r, s


def _series(ax, r, s, col_r, col_s, tmax, real_label=None, sim_label=None):
    mr, ms = r.t <= tmax, s.t <= tmax
    ax.plot(r.loc[mr, "t"], r.loc[mr, col_r], color=C_REAL, lw=0.9, zorder=4,
            label=real_label)
    ax.plot(s.loc[ms, "t"], s.loc[ms, col_s], color=C_SIM, lw=0.9, ls="--", zorder=3,
            label=sim_label)
    ax.set_xlim(0, tmax)


def panel_hysteresis(ax, r, s) -> dict:
    """準静的なゆっくりした加減圧での、内圧と関節角の関係。ループ幅がヒステリシス。"""
    ax.plot(r.meas_pres_DF, r.angle_deg, color=C_REAL, lw=0.9, zorder=4, label="Hardware")
    ax.plot(s.sim_pres_DF, s.sim_angle_deg, color=C_SIM, lw=0.9, ls="--", zorder=3,
            label="Simulation")
    ax.set_xlabel("Pressure [MPa]", labelpad=1)
    ax.set_ylabel("Angle [deg]")
    ax.set_xlim(-0.03, 0.63)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    # 図全体の凡例をここに置く。(a) の右下だけが確実に空くので、他パネルの
    # 凡例をデータに重ねずに済む。指令値はプロキシで足す。
    ax.plot([], [], color=C_CMD, lw=0.7, ls=":", label="Command")
    ax.legend(loc="lower right", handlelength=1.6, borderpad=0.25,
              labelspacing=0.3, borderaxespad=0.3,
              fontsize=plt.rcParams["font.size"] - 1.8)
    tidy(ax)
    return dict(real_max=float(r.angle_deg.max()), sim_max=float(s.sim_angle_deg.max()))


def main() -> int:
    ap = base_argparser("Fig.2: リアリティギャップ")
    ap.add_argument("--width", type=float, default=3.4)
    ap.add_argument("--height", type=float, default=3.05)
    args = ap.parse_args()
    apply_style(args.fontsize)

    r1, s1 = load("exp1_static_hysteresis")
    r2, s2 = load("exp2_step_response")
    r3, s3 = load("exp3_frequency_sweep")

    fig = plt.figure(figsize=(args.width, args.height))
    # 上段((a)(b))と下段((c))は別々の GridSpec にする。1つの GridSpec だと
    # (a) の軸ラベルと (c) のパネル記号がぶつかる。
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.46,
                  left=0.135, right=0.995, top=0.955, bottom=0.575)
    gs_c = GridSpec(2, 1, figure=fig, hspace=0.18,
                    left=0.135, right=0.995, top=0.415, bottom=0.115)

    res = {}
    ax_a = fig.add_subplot(gs[0:2, 0])
    res["a"] = panel_hysteresis(ax_a, r1, s1)

    # (b) ステップ応答。角度と内圧を上下に分け、二軸を使わない。
    ax_b1 = fig.add_subplot(gs[0, 1])
    _series(ax_b1, r2, s2, "angle_deg", "sim_angle_deg", STEP_T)
    ax_b1.set_ylabel("Angle [deg]", labelpad=2)
    ax_b1.tick_params(labelbottom=False)
    tidy(ax_b1)

    ax_b2 = fig.add_subplot(gs[1, 1], sharex=ax_b1)
    ax_b2.plot(r2.loc[r2.t <= STEP_T, "t"], r2.loc[r2.t <= STEP_T, "cmd_DF"],
               color=C_CMD, lw=0.7, ls=":", zorder=2, label="Command")
    _series(ax_b2, r2, s2, "meas_pres_DF", "sim_pres_DF", STEP_T)
    ax_b2.set_ylabel("$P$ [MPa]", labelpad=2)
    ax_b2.set_xlabel("Time [s]", labelpad=1)
    tidy(ax_b2)

    # (c) 掃引周波数入力。時間領域の追従を見る（ボード線図ではない）。
    ax_c1 = fig.add_subplot(gs_c[0])
    _series(ax_c1, r3, s3, "angle_deg", "sim_angle_deg", SWEEP_T)
    ax_c1.set_ylabel("Angle [deg]", labelpad=2)
    ax_c1.tick_params(labelbottom=False)
    tidy(ax_c1)

    ax_c2 = fig.add_subplot(gs_c[1], sharex=ax_c1)
    ax_c2.plot(r3.loc[r3.t <= SWEEP_T, "t"], r3.loc[r3.t <= SWEEP_T, "cmd_DF"],
               color=C_CMD, lw=0.7, ls=":", zorder=2, label="Command")
    _series(ax_c2, r3, s3, "meas_pres_DF", "sim_pres_DF", SWEEP_T)
    ax_c2.set_ylabel("$P$ [MPa]", labelpad=2)
    ax_c2.set_xlabel("Time [s]", labelpad=1)
    tidy(ax_c2)

    for tag, ax, x, y in [("(a)", ax_a, -0.30, 1.03), ("(b)", ax_b1, -0.26, 1.08),
                          ("(c)", ax_c1, -0.135, 1.14)]:
        ax.text(x, y, tag, transform=ax.transAxes, fontweight="bold", color=INK,
                fontsize=plt.rcParams["font.size"], ha="left", va="bottom")

    save(fig, args.outdir, "fig2_realitygap", args.format)
    import json
    print(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())

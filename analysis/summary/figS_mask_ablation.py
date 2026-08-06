"""figS_mask_ablation.py — 補足図: マスクアブレーション単体（1段幅）。

本文では Fig.5(d) に統合してあるので、通常は不要。
スライドや査読回答レターで単独に見せたいとき用。脚注付き。

Usage:
  python analysis/summary/figS_mask_ablation.py --outdir out --format both
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402

from fig5_quanteval import panel_mask  # noqa: E402
from ral_figstyle import COL_W, INK_MUTED, apply_style, base_argparser, save  # noqa: E402

FOOTNOTE = ("$\\dagger$ uniform noise also moves the input off-distribution: the true\n"
            "   lookahead signal is 83 % exact zeros (mean 0.10).")


def main() -> int:
    ap = base_argparser(__doc__.splitlines()[0])
    ap.add_argument("--height", type=float, default=2.05)
    args = ap.parse_args()
    apply_style(args.fontsize)
    fig, ax = plt.subplots(figsize=(COL_W, args.height))
    res = panel_mask(ax)
    ax.text(0.0, -0.30, FOOTNOTE, transform=ax.transAxes,
            fontsize=plt.rcParams["font.size"] - 2.2, color=INK_MUTED,
            ha="left", va="top", linespacing=1.3)
    save(fig, args.outdir, "figS_mask_ablation", args.format)
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())

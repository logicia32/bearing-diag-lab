#!/usr/bin/env python3
"""記事用の図を figures/ に生成する。

  fig1_envelope_spectra.png : エンベロープスペクトル 4 種比較 (幾何の予言線入り)
  fig2_score_matrix.png     : ハーモニックスコア行列 (10 グループ x 3 スコア)
  fig3_leak_bars.png        : ランダム分割 vs サイズまたぎ (特徴量 3 種)

数字は全部この場で計算する (ハードコードしない)。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from classify_sizeleak import build, eval_random, eval_sizeout
from cwru import load, load_all
from diagnose import ORDERS, diagnose, envelope_spectrum

plt.rcParams["font.family"] = "IPAGothic"
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

LINE_STYLE = {
    "OR": dict(color="tab:blue", ls="--"),
    "IR": dict(color="tab:red", ls="--"),
    "B": dict(color="tab:green", ls="--"),
}
LINE_NAME = {"OR": "BPFO (外輪)", "IR": "BPFI (内輪)", "B": "2xBSF (転動体)"}
FTF_ORDER = 0.39828


def fig1_envelope() -> None:
    panels = [
        ("Normal_0", "正常"),
        ("IR007_0", '内輪キズ 0.007"'),
        ("B007_0", '転動体キズ 0.007"'),
        ("OR007@6_0", '外輪キズ 0.007"'),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 9), sharex=True)
    for ax, (label, title) in zip(axes, panels):
        rec = load(label)
        f_shaft = (rec.rpm if rec.rpm is not None else rec.rpm_nominal) / 60.0
        f, pxx = envelope_spectrum(rec.de)
        m = (f >= 2) & (f <= 200)
        db = 10 * np.log10(pxx[m] / np.median(pxx[m]))
        ax.plot(f[m], db, color="0.25", lw=0.8)
        for key, order in ORDERS.items():
            ax.axvline(order * f_shaft, lw=1.4, label=LINE_NAME[key], **LINE_STYLE[key])
        ax.axvline(
            FTF_ORDER * f_shaft, color="0.5", ls=":", lw=1.4, label="FTF (保持器)"
        )
        ax.set_ylabel("dB")
        ax.set_title(f"{title}  ({label}, 軸回転 {f_shaft:.1f} Hz)", fontsize=10)
        ax.set_xlim(0, 200)
        ax.set_ylim(bottom=-5)
    axes[0].legend(loc="upper right", fontsize=8, ncol=4)
    axes[-1].set_xlabel("エンベロープスペクトル周波数 [Hz]")
    fig.suptitle("エンベロープスペクトルと幾何の予言 (破線 = 幾何から計算した欠陥周波数の 1 次)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_envelope_spectra.png", dpi=150)
    plt.close(fig)
    print("fig1 done")


def fig2_scores() -> None:
    groups = [
        "Normal", "IR007", "IR014", "IR021",
        "B007", "B014", "B021",
        "OR007@6", "OR014@6", "OR021@6",
    ]
    score_keys = ["OR", "IR", "B"]
    acc: dict[str, list[list[float]]] = {g: [] for g in groups}
    for rec in load_all():
        g = "Normal" if rec.label.startswith("Normal") else rec.label.split("_")[0]
        s = diagnose(rec)["scores"]
        acc[g].append([s[k] for k in score_keys])
    mat = np.array([np.mean(acc[g], axis=0) for g in groups]).T  # 3 x 10

    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(groups)), groups, rotation=30, ha="right")
    ax.set_yticks(
        range(3), ["OR スコア\n(BPFO)", "IR スコア\n(BPFI)", "B スコア\n(2xBSF)"]
    )
    for j, g in enumerate(groups):
        top = int(np.argmax(mat[:, j]))
        for i in range(3):
            ax.text(
                j, i, f"{mat[i, j]:.0f}",
                ha="center", va="center", fontsize=9,
                color="white" if mat[i, j] < mat.max() * 0.6 else "black",
                fontweight="bold" if i == top else "normal",
            )
        # argmax の枠。正解なら白、不正解なら赤
        truth = {"N": None, "I": 1, "B": 2, "O": 0}[g[0]]
        ec = "white" if truth is None or top == truth else "red"
        ax.add_patch(
            plt.Rectangle(
                (j - 0.5, top - 0.5), 1, 1, fill=False, ec=ec, lw=2, clip_on=False
            )
        )
    ax.set_title(
        "ハーモニックスコア (負荷 0-3HP の平均, dB)。枠 = argmax 判定 (赤枠 = 誤り)",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, label="dB", pad=0.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_score_matrix.png", dpi=150)
    plt.close(fig)
    print("fig2 done")


def fig3_leak() -> None:
    Xa, Xb, Xc, y, sizes = build()
    sets = [
        ("A: 時間統計\n(4 個)", Xa),
        ("B: 物理特徴\n(4 個)", Xb),
        ("C: 帯域エネルギー\n(16 個)", Xc),
    ]
    rand_acc, size_acc = [], []
    for _, X in sets:
        rand_acc.append(eval_random(X, y))
        size_acc.append(eval_sizeout(X, y, sizes)[0])

    x = np.arange(len(sets))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(x - w / 2, rand_acc, w, label="ランダム分割 (リークあり)", color="tab:orange")
    b2 = ax.bar(x + w / 2, size_acc, w, label="キズサイズまたぎ (正直)", color="tab:blue")
    for bars in (b1, b2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.1%}", ha="center", fontsize=9,
            )
    ax.axhline(1 / 3, color="0.4", ls=":", lw=1)
    ax.text(2.42, 1 / 3 + 0.01, "あてずっぽう (3 クラス)", fontsize=8, ha="right", color="0.3")
    ax.set_xticks(x, [n for n, _ in sets])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("正解率")
    ax.set_title("同じ特徴量・同じ分類器でも、評価の切り方で数字は別物になる", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_leak_bars.png", dpi=150)
    plt.close(fig)
    print("fig3 done")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    fig1_envelope()
    fig2_scores()
    fig3_leak()


if __name__ == "__main__":
    main()

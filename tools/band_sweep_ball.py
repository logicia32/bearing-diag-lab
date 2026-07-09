#!/usr/bin/env python3
"""転動体レコードの帯域掃引。

診断失敗が「手法の限界」なのか「私の帯域の決め打ちが悪い」のかを
切り分ける。複数の共振帯候補でエンベロープスペクトルを取り直し、
2xBSF (と、その FTF 側帯波) がどこかの帯域で立つかを見る。
ついでにエンベロープスペクトルの上位ピークが実際どこに立っているかも
表示する (次数に換算して「正体」を推測する)。
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch

from cwru import load
from diagnose import harmonic_score

FS = 12_000
BANDS = [
    (500, 2000),
    (1000, 3000),
    (2000, 4000),
    (2500, 5500),
    (3000, 5800),
    (4000, 5800),
]
ORDERS = {"OR": 3.5848, "IR": 5.4152, "2xBSF": 4.7135, "BSF": 2.3567, "FTF": 0.39828}


def env_spec(x, band):
    b, a = butter(4, band, btype="bandpass", fs=FS)
    env = np.abs(hilbert(filtfilt(b, a, x)))
    env -= env.mean()
    return welch(env, fs=FS, nperseg=1 << 15)


def top_peaks(f, pxx, fmax=250.0, n=6):
    m = (f > 2) & (f < fmax)
    fm, pm = f[m], pxx[m]
    # 素朴な極大抽出
    idx = [i for i in range(1, len(pm) - 1) if pm[i] > pm[i - 1] and pm[i] > pm[i + 1]]
    idx.sort(key=lambda i: pm[i], reverse=True)
    return [(fm[i], pm[i]) for i in idx[:n]]


def main() -> None:
    for label in ["B007_0", "B007_2"]:
        rec = load(label)
        f_shaft = (rec.rpm or rec.rpm_nominal) / 60.0
        print(f"=== {label} (軸回転 {f_shaft:.2f} Hz) ===")
        print(f"    期待値: 2xBSF={4.7135*f_shaft:.1f}Hz BSF={2.3567*f_shaft:.1f}Hz "
              f"FTF={0.39828*f_shaft:.1f}Hz OR={3.5848*f_shaft:.1f}Hz IR={5.4152*f_shaft:.1f}Hz")
        for band in BANDS:
            f, pxx = env_spec(rec.de, band)
            sc = {k: harmonic_score(f, pxx, o * f_shaft) for k, o in ORDERS.items()}
            peaks = top_peaks(f, pxx)
            peak_str = " ".join(
                f"{fp:.1f}Hz({fp/f_shaft:.2f}x)" for fp, _ in peaks[:4]
            )
            print(
                f"  band {band[0]:>4}-{band[1]:>4}:"
                f" 2xBSF {sc['2xBSF']:>5.1f} | BSF {sc['BSF']:>5.1f}"
                f" | OR {sc['OR']:>5.1f} | IR {sc['IR']:>5.1f} dB"
                f"   上位ピーク: {peak_str}"
            )
        print()


if __name__ == "__main__":
    main()

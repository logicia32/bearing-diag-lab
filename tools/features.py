#!/usr/bin/env python3
"""窓単位の特徴量と分類器 — リーク実験 (classify_*.py) の共通部品。

特徴量は 3 系統。すべて 1 秒窓 (12,000 点) を入力にとる。
  feats_naive   : 素朴な時間統計 4 個 (RMS / 尖度 / クレストファクタ / p2p)
  feats_bands   : 帯域エネルギー 16 個 (収録の「指紋」を最も拾いやすい系統)
  feats_physics : 幾何の予言位置のハーモニックスコア 4 個 (物理特徴)

分類器はロジスティック回帰 (標準化つき) に固定。実験ごとに変えない。
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, periodogram, welch
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from diagnose import BAND, FS, harmonic_score

WIN = FS  # 1 秒窓 (12,000 点)
N_WIN = 10  # 全レコード先頭 10 窓に揃える (正常だけ収録が長いので)
ORDERS = {"OR": 3.5848, "IR": 5.4152, "2xBSF": 4.7135, "FTF": 0.39828}
SEED = 20260709


def feats_naive(x: np.ndarray) -> list[float]:
    rms = float(np.sqrt(np.mean(x**2)))
    return [
        np.log10(rms),
        float(kurtosis(x)),
        float(np.max(np.abs(x)) / rms),  # クレストファクタ
        np.log10(float(np.ptp(x))),
    ]


def feats_bands(x: np.ndarray, n_bands: int = 16) -> list[float]:
    """帯域エネルギー。0-6kHz を等分し、各帯域の log パワーを並べる。

    「収録の指紋」を最も拾いやすい種類の特徴。リーク評価を 99% に
    見せかける再現用で、物理の意味づけはしていない。
    """
    f, pxx = welch(x, fs=FS, nperseg=4096)
    edges = np.linspace(0, FS / 2, n_bands + 1)
    return [
        float(np.log10(pxx[(f >= lo) & (f < hi)].sum() + 1e-30))
        for lo, hi in zip(edges[:-1], edges[1:])
    ]


def feats_physics(x: np.ndarray, f_shaft: float) -> list[float]:
    """diagnose.py と同じ「予言位置のスコア」だが PSD 推定だけ違う:
    あちらは全長 (~10s) を welch、こちらは 1 秒窓なので periodogram
    (分解能 1Hz)。welch で平均化すると分解能が ±2% の探索窓 (107-162Hz
    で ±2-3Hz) より粗くなるため、分散は受け入れて分解能を取る。"""
    b, a = butter(4, BAND, btype="bandpass", fs=FS)
    env = np.abs(hilbert(filtfilt(b, a, x)))
    env -= env.mean()
    f, pxx = periodogram(env, fs=FS)
    return [harmonic_score(f, pxx, o * f_shaft) for o in ORDERS.values()]


def make_clf():
    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)
    )

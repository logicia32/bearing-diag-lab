#!/usr/bin/env python3
"""エンベロープ解析で故障種別を「物理の予言」だけで言い当てる。

手順 (教科書どおりの型):
  1. 帯域通過 — 軸受のキズの衝撃は構造共振 (数 kHz) を励振する。
     回転 1 次や噛み合いなどの低周波の大物をどけて、共振帯だけ残す。
  2. Hilbert エンベロープ — 衝撃の「繰り返し」を取り出す。
  3. エンベロープスペクトル — 繰り返し周波数が BPFO/BPFI/2xBSF の
     どれに立つかで、外輪/内輪/転動体を見分ける。

判定は機械学習なし。幾何から計算した欠陥周波数 (回転次数) に
エネルギーが立っているかのスコアだけで argmax する。
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch

from cwru import Record, load_all

FS = 12_000  # 全ファイル 12kHz (正常も検証済み: verify_normal_fs.py)

# SKF 6205-2RS JEM の欠陥次数 (公式値・幾何から逆算一致済み)
ORDERS = {
    "OR": 3.5848,   # BPFO: 外輪キズ
    "IR": 5.4152,   # BPFI: 内輪キズ
    "B": 4.7135,    # 2xBSF: 転動体キズ (1 自転で両レースに当たる)
}

BAND = (2500.0, 5500.0)  # 共振帯の素朴な決め打ち (Nyquist 6k の内側)
N_HARMONICS = 3


def envelope_spectrum(x: np.ndarray, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
    b, a = butter(4, BAND, btype="bandpass", fs=fs)
    xf = filtfilt(b, a, x)
    env = np.abs(hilbert(xf))
    env -= env.mean()
    return welch(env, fs=fs, nperseg=1 << 15)


def harmonic_score(
    f: np.ndarray, pxx: np.ndarray, f0: float, n_harm: int = N_HARMONICS
) -> float:
    """f0 の 1..n 次高調波での (ピーク / 近傍床) を dB で平均する。

    床は「±15Hz の中央値」という素朴な固定幅。CWRU の回転レンジ
    (軸 28.7-30Hz、対象次数 0.4-5.4) では欠陥周波数どうしが 15Hz より
    離れているので成立するが、回転数や幾何が大きく違う機械に
    そのまま持ち出せる定義ではない。"""
    scores = []
    for k in range(1, n_harm + 1):
        target = k * f0
        half = max(1.0, 0.02 * target)  # 探索窓: ±2% (最低 ±1 Bin 分)
        m_peak = (f >= target - half) & (f <= target + half)
        m_floor = (f >= target - 15) & (f <= target + 15) & ~m_peak
        if not m_peak.any() or not m_floor.any():
            continue
        peak = pxx[m_peak].max()
        floor = np.median(pxx[m_floor])
        scores.append(10 * np.log10(peak / floor))
    return float(np.mean(scores)) if scores else float("nan")


def diagnose(rec: Record) -> dict:
    rpm = rec.rpm if rec.rpm is not None else rec.rpm_nominal
    f_shaft = rpm / 60.0
    f, pxx = envelope_spectrum(rec.de)
    scores = {k: harmonic_score(f, pxx, order * f_shaft) for k, order in ORDERS.items()}
    return {"label": rec.label, "f_shaft": f_shaft, "scores": scores}


def main() -> None:
    print(f"{'label':<10} {'軸回転':>6}   {'OR score':>8} {'IR score':>8} {'B score':>8}   判定")
    hits = 0
    total = 0
    for rec in load_all():
        d = diagnose(rec)
        s = d["scores"]
        best = max(s, key=s.get)
        truth = (
            "Normal" if rec.label.startswith("Normal")
            else "IR" if rec.label.startswith("IR")
            else "B" if rec.label.startswith("B0")
            else "OR"
        )
        mark = ""
        if truth != "Normal":
            total += 1
            ok = best == truth
            hits += ok
            mark = "✓" if ok else f"✗ (正解 {truth})"
        print(
            f"{d['label']:<10} {d['f_shaft']:>5.2f}Hz"
            f"   {s['OR']:>7.1f}dB {s['IR']:>7.1f}dB {s['B']:>7.1f}dB"
            f"   -> {best} {mark}"
        )
    print(f"\n故障レコード {total} 本中 {hits} 本正解 (argmax、しきい値なし)")


if __name__ == "__main__":
    main()

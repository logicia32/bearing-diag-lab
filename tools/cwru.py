#!/usr/bin/env python3
"""CWRU .mat の正しいローダ。

素朴に「_DE_time にマッチした最初の変数」を拾うと 99.mat で事故る:
99.mat には前のファイルの変数 (X098_DE_time) と ans が混入している。
ファイル番号から変数名を組み立てて明示的に選ぶ。

正常ベースラインのサンプリング周波数は公式ページに明記がないが、
検証 (verify_normal_fs.py) の結果 48kHz。故障 (12k Drive End) と
揃えるため、既定では正常レコードを 12kHz に間引いて返す。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
from scipy.signal import decimate

from download_cwru import DATA_DIR, MANIFEST

FS_12K = 12_000
FS_NORMAL_RAW = 48_000  # 正常ベースラインの生 fs (検証済み)


@dataclass
class Record:
    label: str
    num: int
    load_hp: int
    rpm_nominal: int
    rpm: float | None  # 実測 (RPM 変数が無いファイルは None)
    fs: int  # de/fe/ba のサンプリング周波数 (間引き後)
    de: np.ndarray
    fe: np.ndarray | None
    ba: np.ndarray | None


def load(label: str, data_dir: Path = DATA_DIR, to_12k: bool = True) -> Record:
    """to_12k=True (既定) なら正常レコードを 48kHz -> 12kHz に間引く。"""
    num, load_hp, rpm_nominal = MANIFEST[label]
    mat = scipy.io.loadmat(data_dir / f"{num}.mat")
    prefix = f"X{num:03d}"

    def pick(suffix: str) -> np.ndarray | None:
        arr = mat.get(f"{prefix}{suffix}")
        return None if arr is None else np.asarray(arr).squeeze().astype(float)

    de = pick("_DE_time")
    if de is None:
        raise KeyError(f"{num}.mat: {prefix}_DE_time が見つからない")
    rpm_arr = pick("RPM")

    fs_raw = FS_NORMAL_RAW if label.startswith("Normal") else FS_12K
    fs = fs_raw
    fe, ba = pick("_FE_time"), pick("_BA_time")
    if to_12k and fs_raw != FS_12K:
        q = fs_raw // FS_12K
        de = decimate(de, q, ftype="fir", zero_phase=True)
        fe = None if fe is None else decimate(fe, q, ftype="fir", zero_phase=True)
        ba = None if ba is None else decimate(ba, q, ftype="fir", zero_phase=True)
        fs = FS_12K

    return Record(
        label=label,
        num=num,
        load_hp=load_hp,
        rpm_nominal=rpm_nominal,
        rpm=None if rpm_arr is None else float(rpm_arr),
        fs=fs,
        de=de,
        fe=fe,
        ba=ba,
    )


def load_all() -> list[Record]:
    return [load(label) for label in MANIFEST]


if __name__ == "__main__":
    print(f"{'label':<10} {'file':>4} {'DE 長':>8} {'FE 長':>8} {'BA':>3} {'実測rpm':>8}")
    for r in load_all():
        print(
            f"{r.label:<10} {r.num:>4} {len(r.de):>8}"
            f" {len(r.fe) if r.fe is not None else -1:>8}"
            f" {'有' if r.ba is not None else '無':>3}"
            f" {r.rpm if r.rpm is not None else float('nan'):>8.0f}"
        )

#!/usr/bin/env python3
"""ダウンロードした .mat の中身を検証する。

変数名の慣例 (公式サイトの記述):
  X{num}_DE_time : drive end 加速度 [g]
  X{num}_FE_time : fan end 加速度 [g]
  X{num}_BA_time : base 加速度 [g] (無いファイルもある)
  X{num}RPM      : 収録時の実回転数
"""

from pathlib import Path

import numpy as np
import scipy.io

from download_cwru import DATA_DIR, MANIFEST

FS_FAULT = 12_000  # 12k Drive End データ


def main() -> None:
    print(f"{'label':<10} {'file':>4}  {'keys (X 系のみ)':<42} {'DE 長':>8} {'秒@12k':>7} {'rpm':>6}")
    for label, (num, _load, rpm_nominal) in MANIFEST.items():
        path = DATA_DIR / f"{num}.mat"
        mat = scipy.io.loadmat(path)
        xkeys = [k for k in mat.keys() if not k.startswith("__")]
        de = next((mat[k] for k in xkeys if k.endswith("_DE_time")), None)
        rpm = next((mat[k] for k in xkeys if k.endswith("RPM")), None)
        n = len(de) if de is not None else -1
        rpm_v = float(np.squeeze(rpm)) if rpm is not None else float("nan")
        print(
            f"{label:<10} {num:>4}  {','.join(sorted(xkeys)):<42}"
            f" {n:>8} {n / FS_FAULT:>7.1f} {rpm_v:>6.0f}"
            f"  (公称 {rpm_nominal})"
        )


if __name__ == "__main__":
    main()

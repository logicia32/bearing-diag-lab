#!/usr/bin/env python3
"""正常ベースラインのサンプリング周波数を、データ自身に判定させる。

公式ページは正常データの fs を明記していない。結論: 48kHz。

## 一度まちがえた話 (このスクリプト自体の教訓)

最初は「fs=12kHz と仮定して軸回転 1 次 (rpm/60) の位置にピークが
立つか」を見て、立ったので 12kHz と結論した。これは罠だった。
このモータは 4 極・同期 1800rpm 近くで回るので、

    軸 1 次 = 29.9Hz 前後   と   電源 2 倍 120Hz / 4 = 30.0Hz

が、48kHz のデータを 12kHz と誤読したときにほぼ同じ表示位置に
落ちる。29.9 と 30.0 は分解能以下で区別できない。2 次・4 次と
思った線も、実は 240Hz・480Hz (電磁加振の高調波) だった。

## 正しい切り分け: rpm に追従するか、30.00Hz に固定か

負荷 3HP のレコードは回転が 1725rpm (軸 28.75Hz) まで下がるので、
機械由来なら 28.75Hz、電気由来 (120Hz/4) なら 30.0Hz と、
1.25Hz 離れて分解能で切り分けられる。

  - 正常レコード: ピークは rpm に依らず表示 30.03Hz に固定
                  → 実 120Hz。すなわち fs=48kHz
  - 故障レコード (fs=12kHz は公式明記): ピークは rpm/60 に追従
                  → 対照実験として整合

補強証拠: 48kHz と読むと (1) 収録長が 5〜10 秒と故障レコード並みに
なる (2) 包絡スペクトルの「1/4 次の謎のコム」がただの軸次数
(1x, 2x, 3x...) になる。
"""

import numpy as np
from scipy.signal import welch

from cwru import load

CHECKS = [
    # (label, 対照か)  正常 4 本 + 同負荷の故障 2 本 (対照)
    ("Normal_0", False),
    ("Normal_1", False),
    ("Normal_2", False),
    ("Normal_3", False),
    ("OR007@6_3", True),
    ("IR007_3", True),
]


def peak_near(f, pxx, target, tol=0.7):
    m = (f >= target - tol) & (f <= target + tol)
    i = np.argmax(pxx[m])
    return float(f[m][i]), float(pxx[m][i])


def main() -> None:
    print(f"{'label':<10} {'rpm':>5} {'軸1次(機械説)':>14} {'30.00Hz(電気説)':>16}  判定")
    for label, is_control in CHECKS:
        r = load(label, to_12k=False)  # 生のまま。表示軸は fs=12k と誤読した状態を再現
        rpm = r.rpm if r.rpm is not None else r.rpm_nominal
        f, pxx = welch(r.de, fs=12_000, nperseg=1 << 16)
        med = np.median(pxx[(f > 2) & (f < 200)])
        f_mech, p_mech = peak_near(f, pxx, rpm / 60)
        f_elec, p_elec = peak_near(f, pxx, 30.00)
        mech, elec = p_mech / med, p_elec / med
        verdict = "rpm追従=機械(12kHz)" if mech > 3 * elec else (
            "30Hz固定=電気(48kHz)" if elec > 3 * mech else "判定不能(軸≒30Hzの負荷)"
        )
        tag = " (対照)" if is_control else ""
        print(
            f"{label:<10} {rpm:>5.0f} {f_mech:>7.2f}Hz {mech:>5.0f}x"
            f" {f_elec:>9.2f}Hz {elec:>5.0f}x  {verdict}{tag}"
        )
    print()
    print("決め手は Normal_3 (軸 28.75Hz と 30.0Hz が分解能で分離できる唯一の正常")
    print("レコード) と、対照の故障レコード 2 本。負荷 0-2 は軸 1 次と電源 2 倍の")
    print("窓が重なり原理的に判定できない。傍証: 48kHz と読むと収録長が 5〜10 秒と")
    print("故障レコード並みになり、包絡の「1/4 次コム」もただの軸次数になる。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""定番の「10 クラス分類 99%」を最小の道具で再現する。

CWRU を使う論文・記事でよく見るタスク設定:
  10 クラス = 正常 + {内輪, 転動体, 外輪@6時} x {0.007, 0.014, 0.021 inch}
  評価 = 窓単位のランダム分割 (クラス名はラベル先頭部そのまま。外輪は OR..@6)

各クラスは物理的に 1 個のベアリング個体なので、このタスクは
「故障の種類を見分ける」ではなく「どの収録か言い当てる」に等しい。
それを確かめるため、深層学習どころか特徴量 4 個 (RMS/尖度/CF/p2p) +
ロジスティック回帰でどこまで出るかを見る。
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold

from cwru import load_all
from features import N_WIN, SEED, WIN, feats_bands, feats_naive, feats_physics, make_clf


def tenclass(label: str) -> str:
    """'IR014_2' -> 'IR014', 'Normal_1' -> 'Normal'。"""
    return "Normal" if label.startswith("Normal") else label.split("_")[0]


def main() -> None:
    Xa, Xb, Xc, y = [], [], [], []
    for rec in load_all():
        f_shaft = (rec.rpm if rec.rpm is not None else rec.rpm_nominal) / 60.0
        for i in range(N_WIN):
            seg = rec.de[i * WIN : (i + 1) * WIN]
            if len(seg) < WIN:
                break
            Xa.append(feats_naive(seg))
            Xb.append(feats_physics(seg, f_shaft))
            Xc.append(feats_bands(seg))
            y.append(tenclass(rec.label))
    Xa, Xb, Xc, y = np.array(Xa), np.array(Xb), np.array(Xc), np.array(y)
    n_cls = len(set(y))
    print(f"{len(y)} 窓, {n_cls} クラス: {sorted(set(y))}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, X in [
        ("A: 時間統計 4 個", Xa),
        ("B: 物理特徴 4 個", Xb),
        ("C: 帯域エネルギー 16 個", Xc),
    ]:
        hits = 0
        for tr, te in skf.split(X, y):
            clf = make_clf().fit(X[tr], y[tr])
            hits += int((clf.predict(X[te]) == y[te]).sum())
        print(f"  {name}: ランダム 5-fold 精度 {hits / len(y):.1%}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""「評価のリーク」を数字で見せる実験 その 1: 負荷またぎ。

ベアリング診断の論文・ブログでよく見る「精度 99%」は、
同じ収録データの切れ端が訓練とテストの両方に入っている
(= リーク) ことが多い。CWRU は 1 条件 1 収録なので、
ランダム分割すると必ずこのリークが起きる。

実験:
  特徴量 A: 素朴な時間統計 / B: 物理特徴 (features.py)
  評価 1: ランダム 5-fold (窓単位でシャッフル) — リークあり
  評価 2: 負荷またぎ leave-one-load-out — 未見の運転条件でテスト

結果メモ: 負荷またぎは時間統計ですら 100% になる。CWRU は
1 クラス = 物理的に 1 個のベアリングを負荷だけ変えて録っている
ので、負荷をまたいでも「同じキズ個体」が両側にいる (個体リーク)。
本当に正直な評価はキズサイズまたぎ (classify_sizeleak.py)。
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from cwru import load_all
from features import N_WIN, SEED, WIN, feats_naive, feats_physics, make_clf

CLASSES = ["Normal", "IR", "B", "OR"]


def truth_of(label: str) -> str:
    if label.startswith("Normal"):
        return "Normal"
    if label.startswith("IR"):
        return "IR"
    if label.startswith("B0"):
        return "B"
    return "OR"


def build_dataset():
    """窓ごとの特徴量とメタ情報 (真値クラス, 負荷) を作る。

    対象は正常 + 0.007" の 16 レコードに限定する。この実験の主張は
    「1 クラス = 1 個体なら、負荷をまたいでも同じ個体が両側にいる」
    なので、サイズを混ぜて 1 クラス複数個体にすると成立しない。
    """
    Xa, Xb, y, loads = [], [], [], []
    for rec in load_all():
        if not (rec.label.startswith("Normal") or "007" in rec.label):
            continue
        f_shaft = (rec.rpm if rec.rpm is not None else rec.rpm_nominal) / 60.0
        t = truth_of(rec.label)
        for i in range(N_WIN):
            seg = rec.de[i * WIN : (i + 1) * WIN]
            if len(seg) < WIN:
                break
            Xa.append(feats_naive(seg))
            Xb.append(feats_physics(seg, f_shaft))
            y.append(t)
            loads.append(rec.load_hp)
    return (np.array(Xa), np.array(Xb), np.array(y), np.array(loads))


def eval_random(X, y) -> float:
    """ランダム 5-fold。同じ収録の窓が訓練とテストに混ざる (リーク)。"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    hits = 0
    for tr, te in skf.split(X, y):
        clf = make_clf().fit(X[tr], y[tr])
        hits += int((clf.predict(X[te]) == y[te]).sum())
    return hits / len(y)


def eval_loadout(X, y, loads) -> tuple[float, np.ndarray]:
    """leave-one-load-out。テスト負荷の収録は訓練に一切入らない。"""
    y_true, y_pred = [], []
    for held in sorted(set(loads)):
        tr, te = loads != held, loads == held
        clf = make_clf().fit(X[tr], y[tr])
        y_true.extend(y[te])
        y_pred.extend(clf.predict(X[te]))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    return acc, cm


def main() -> None:
    Xa, Xb, y, loads = build_dataset()
    n = len(y)
    counts = {c: int((y == c).sum()) for c in CLASSES}
    print(f"窓データセット: {n} 窓 (1 秒窓 x 最大 {N_WIN}/レコード, 内訳 {counts})")
    print()
    print(f"{'':<28} {'ランダム5-fold':>14} {'負荷またぎ':>10}")
    for name, X in [("A: 時間統計 (RMS/尖度/CF/p2p)", Xa), ("B: 物理特徴 (欠陥次数スコア)", Xb)]:
        r = eval_random(X, y)
        lo, cm = eval_loadout(X, y, loads)
        print(f"{name:<28} {r:>13.1%} {lo:>10.1%}")
    print()
    for name, X in [("A: 時間統計", Xa), ("B: 物理特徴", Xb)]:
        _, cm = eval_loadout(X, y, loads)
        print(f"--- 負荷またぎの混同行列 ({name}) 行=真値 列=予測 {CLASSES}")
        for cls, row in zip(CLASSES, cm):
            print(f"  {cls:<7} {row}")
        print()


if __name__ == "__main__":
    main()

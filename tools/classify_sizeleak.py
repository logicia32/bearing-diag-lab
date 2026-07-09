#!/usr/bin/env python3
"""キズサイズまたぎ = 「別の物理個体」で試す、本当に正直な評価。

負荷またぎ (classify_loadleak.py) では時間統計ですら 100% だった。
CWRU は 1 クラス 1 個のベアリングを負荷だけ変えて録っているので、
負荷をまたいでも「同じキズ個体」が訓練とテストの両方にいるからだ。

そこでキズサイズ (0.007/0.014/0.021 inch) でまたぐ。サイズが違えば
物理的に別の個体・別の加工なので、「そのキズの音色の暗記」は通用せず、
故障の種類そのものを掴んだ特徴だけが生き残るはず。

実験: 故障レコードのみ 3 クラス (IR/B/OR) 分類。
  評価 1: ランダム 5-fold (リークあり: 同じ収録の窓が両側に入る)
  評価 2: leave-one-size-out (訓練 2 サイズ -> 未見サイズでテスト)
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from cwru import load_all
from features import N_WIN, SEED, WIN, feats_bands, feats_naive, feats_physics, make_clf

CLASSES = ["IR", "B", "OR"]
SIZES = ["007", "014", "021"]


def parse(label: str) -> tuple[str, str] | None:
    """'IR014_2' -> ('IR', '014')。Normal は None。"""
    if label.startswith("Normal"):
        return None
    for cls in CLASSES:
        if label.startswith(cls):
            rest = label[len(cls):]
            return cls, rest[:3]
    raise ValueError(label)


def build():
    Xa, Xb, Xc, y, sizes = [], [], [], [], []
    for rec in load_all():
        meta = parse(rec.label)
        if meta is None:
            continue
        cls, size = meta
        f_shaft = (rec.rpm if rec.rpm is not None else rec.rpm_nominal) / 60.0
        for i in range(N_WIN):
            seg = rec.de[i * WIN : (i + 1) * WIN]
            if len(seg) < WIN:
                break
            Xa.append(feats_naive(seg))
            Xb.append(feats_physics(seg, f_shaft))
            Xc.append(feats_bands(seg))
            y.append(cls)
            sizes.append(size)
    return np.array(Xa), np.array(Xb), np.array(Xc), np.array(y), np.array(sizes)


def eval_random(X, y) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    hits = 0
    for tr, te in skf.split(X, y):
        clf = make_clf().fit(X[tr], y[tr])
        hits += int((clf.predict(X[te]) == y[te]).sum())
    return hits / len(y)


def eval_sizeout(X, y, sizes):
    """テストサイズの個体は訓練に一切入らない。"""
    y_true, y_pred, fold_acc = [], [], {}
    for held in SIZES:
        tr, te = sizes != held, sizes == held
        clf = make_clf().fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        fold_acc[held] = float((pred == y[te]).mean())
        y_true.extend(y[te])
        y_pred.extend(pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    return acc, fold_acc, cm


def main() -> None:
    Xa, Xb, Xc, y, sizes = build()
    print(f"窓データセット: {len(y)} 窓 (故障のみ 3 クラス, サイズ 007/014/021)")
    print()
    print(f"{'':<28} {'ランダム5-fold':>14} {'サイズまたぎ':>12}")
    results = {}
    for name, X in [
        ("A: 時間統計 (RMS/尖度/CF/p2p)", Xa),
        ("B: 物理特徴 (欠陥次数スコア)", Xb),
        ("C: 帯域エネルギー 16 個", Xc),
    ]:
        r = eval_random(X, y)
        so, fold_acc, cm = eval_sizeout(X, y, sizes)
        results[name] = (fold_acc, cm)
        print(f"{name:<28} {r:>13.1%} {so:>12.1%}")
    print()
    for name, (fold_acc, cm) in results.items():
        folds = "  ".join(f"テスト{k}: {v:.1%}" for k, v in fold_acc.items())
        print(f"--- {name}")
        print(f"    サイズ別: {folds}")
        print(f"    混同行列 (行=真値 列=予測 {CLASSES})")
        for cls, row in zip(CLASSES, cm):
            print(f"      {cls:<3} {row}")
        print()


if __name__ == "__main__":
    main()

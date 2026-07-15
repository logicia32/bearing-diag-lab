# bearing-diag-lab

**English** | [日本語](#日本語)

A workbench for bearing-fault diagnosis on the public CWRU Bearing Data Center data: identifying the fault location by envelope analysis, and a companion evaluation experiment that checks where the familiar "99% diagnostic accuracy" figure actually comes from.

Write-ups:
**English** → https://logicia32.hashnode.dev/bearing-diagnosis-99-percent-accuracy-cwru ·
**日本語** → https://zenn.dev/logicia32/articles/2026-07-09-bearing-envelope-leak

![Harmonic score matrix. Inner race is correct at every size, outer race is wrong only at 0.014, ball faults miss almost everywhere.](figures/fig2_score_matrix_en.png)

> Note: the source-code comments are in Japanese. The code itself is plain Python, so it reads fine either way; only the prose comments are untranslated.

## Usage

```bash
pip install -r requirements.txt
cd tools
python download_cwru.py      # fetch the 40 files (~134 MB) from the official site
python cwru.py               # load-check (prints the record list)
python diagnose.py           # envelope analysis + argmax verdict from the geometry prediction
python classify_tenclass.py  # reproduce the classic 10-class × random-split task
python classify_loadleak.py  # load-holdout evaluation (specimen-leak demo)
python classify_sizeleak.py  # fault-size-holdout evaluation (test on a different specimen)
python make_figs.py          # generate the article's 3 figures into figures/
```

Helper scripts:

| File | What it does |
|---|---|
| `inspect_cwru.py` | Naively peeks inside the .mat files — reproduces how the stray variable in `99.mat` was noticed |
| `verify_normal_fs.py` | Determines the healthy data's fs (conclusion: 48 kHz). The comments also keep the record of once misjudging it as 12 kHz |
| `band_sweep_ball.py` | Sweeps the ball-fault resonance band six ways — confirms 2×BSF stands out in none of them |
| `features.py` | The three feature families (time statistics / band energy / defect-order scores) and the shared classifier parts |

## Key results

- Argmax verdict, no machine learning: inner race 12/12, outer race 8/12 (0.014 alone 0/4), ball 2/12
- 10-class × random split: 99.7% from just 16 band-energy features + logistic regression
- Same features under fault-size holdout: 68.1% (with 0.014 held out, 33.3% = chance for 3 classes)
- Physics features (defect-order scores): 68.6%. The errors concentrate on ball faults and outer race 0.014 — matching the limit of envelope analysis

The meaning of the numbers, and the pitfalls, are covered in the write-up.

## About the data

The measurement data is not redistributed. `download_cwru.py` fetches it directly from the official site (`data/raw/` is git-ignored).

Source: [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter)

Reference: W. A. Smith and R. B. Randall, "Rolling element bearing diagnostics using the Case Western Reserve University data: a benchmark study," *Mechanical Systems and Signal Processing*, Vol. 64-65, 2015.

## Requirements

Python 3.10+. Dependencies are numpy / scipy / scikit-learn / matplotlib only. The Japanese labels in `make_figs.py` need a Japanese font such as IPAGothic (`make_figure_en.py`-style English-label figures are also in `figures/`, suffixed `_en`).

## License

MIT

---

## 日本語

# bearing-diag-lab

CWRU Bearing Data Center の公開データを使って、ベアリングの故障種別をエンベロープ解析で特定する実験と、「診断精度 99%」という定番の数字がどこから来るのかを確かめる評価実験の置き場です。

解説記事: [ベアリング診断の「精度 99%」はどこから来るのか ―― 定番データセットで確かめる](https://zenn.dev/logicia32/articles/2026-07-09-bearing-envelope-leak)

![ハーモニックスコア行列。内輪は全サイズで正解、外輪は 0.014 のみ誤り、転動体はほぼ全滅](figures/fig2_score_matrix.png)

## 使い方

```bash
pip install -r requirements.txt
cd tools
python download_cwru.py      # 公式サイトから 40 ファイル (約 134MB) を取得
python cwru.py               # 読み込み検証 (レコード一覧を表示)
python diagnose.py           # エンベロープ解析 + 幾何予言の argmax 判定
python classify_tenclass.py  # 定番の 10 クラス x ランダム分割を再現
python classify_loadleak.py  # 負荷またぎ評価 (個体リークのデモ)
python classify_sizeleak.py  # キズサイズまたぎ評価 (別個体でのテスト)
python make_figs.py          # 記事の図 3 枚を figures/ に生成
```

補助スクリプト:

| ファイル | 中身 |
|---|---|
| `inspect_cwru.py` | .mat の中身を素朴に覗く。99.mat の変数混入に気づいた経緯の再現 |
| `verify_normal_fs.py` | 正常データの fs 判定 (結論 48kHz)。一度 12kHz と誤判定した顛末もコメントに残してある |
| `band_sweep_ball.py` | 転動体レコードの共振帯を 6 通り掃引。2xBSF がどの帯域でも立たないことの確認 |
| `features.py` | 窓特徴量 3 系統 (時間統計 / 帯域エネルギー / 欠陥次数スコア) と分類器の共通部品 |

## 主な結果

- 機械学習なしの argmax 判定: 内輪 12/12、外輪 8/12 (0.014 のみ 0/4)、転動体 2/12
- 10 クラス x ランダム分割: 帯域エネルギー 16 個 + ロジスティック回帰だけで 99.7%
- 同じ特徴量でキズサイズまたぎ: 68.1% (テストサイズ 0.014 では 33.3% = 3 クラスのあてずっぽう)
- 物理特徴 (欠陥次数スコア): 68.6%。誤りは転動体と外輪 0.014 に集中し、エンベロープ解析の限界と一致

数字の意味と落とし穴は記事のほうに書いています。

## データについて

計測データは再配布していません。`download_cwru.py` が公式サイトから直接取得します (`data/raw/` は git 管理外)。

出典: [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter)

参考文献: W. A. Smith and R. B. Randall, "Rolling element bearing diagnostics using the Case Western Reserve University data: a benchmark study," Mechanical Systems and Signal Processing, Vol. 64-65, 2015.

## 動作環境

Python 3.10 以降。依存は numpy / scipy / scikit-learn / matplotlib のみです。`make_figs.py` の日本語ラベルには IPA ゴシックなどの日本語フォントが必要です。

## License

MIT

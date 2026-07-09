#!/usr/bin/env python3
"""CWRU Bearing Data Center からデータを取得する。

データの再配布はしない方針なので、このスクリプトで各自公式サイトから
ダウンロードする。出典:
  Case Western Reserve University Bearing Data Center
  https://engineering.case.edu/bearingdatacenter

ファイル番号とラベルの対応は公式サイトの表から転記した。
  - 正常ベースライン (Normal Baseline Data)
  - 12k Drive End Bearing Fault Data (12,000 samples/sec)
  - 欠陥は放電加工による人工欠陥。直径 0.007 / 0.014 / 0.021 inch
    (外輪はいずれも 6 時方向 @6:00)
  - 負荷 0-3 HP (公称回転数 1797/1772/1750/1730 rpm)
"""

import os
import sys
import urllib.request
from pathlib import Path

MAT_MAGIC = b"MATLAB"  # .mat v5 ヘッダ先頭。HTML エラーページの取り違えを検出する

BASE_URL = "https://engineering.case.edu/sites/default/files/{num}.mat"
# 置き場所は環境変数 CWRU_DATA_DIR で上書き可 (既定はリポジトリ内 data/raw)
DATA_DIR = Path(
    os.environ.get("CWRU_DATA_DIR", Path(__file__).resolve().parent.parent / "data" / "raw")
)

# label: (file_number, motor_load_hp, rpm)
MANIFEST = {
    # 正常ベースライン
    "Normal_0": (97, 0, 1797),
    "Normal_1": (98, 1, 1772),
    "Normal_2": (99, 2, 1750),
    "Normal_3": (100, 3, 1730),
    # 内輪 0.007" (12k Drive End)
    "IR007_0": (105, 0, 1797),
    "IR007_1": (106, 1, 1772),
    "IR007_2": (107, 2, 1750),
    "IR007_3": (108, 3, 1730),
    # 転動体 0.007" (12k Drive End)
    "B007_0": (118, 0, 1797),
    "B007_1": (119, 1, 1772),
    "B007_2": (120, 2, 1750),
    "B007_3": (121, 3, 1730),
    # 外輪 0.007" @6:00 (12k Drive End)
    "OR007@6_0": (130, 0, 1797),
    "OR007@6_1": (131, 1, 1772),
    "OR007@6_2": (132, 2, 1750),
    "OR007@6_3": (133, 3, 1730),
    # --- ここから 0.014" / 0.021" (キズサイズまたぎ評価用) ---
    # 内輪 0.014"
    "IR014_0": (169, 0, 1797),
    "IR014_1": (170, 1, 1772),
    "IR014_2": (171, 2, 1750),
    "IR014_3": (172, 3, 1730),
    # 転動体 0.014"
    "B014_0": (185, 0, 1797),
    "B014_1": (186, 1, 1772),
    "B014_2": (187, 2, 1750),
    "B014_3": (188, 3, 1730),
    # 外輪 0.014" @6:00
    "OR014@6_0": (197, 0, 1797),
    "OR014@6_1": (198, 1, 1772),
    "OR014@6_2": (199, 2, 1750),
    "OR014@6_3": (200, 3, 1730),
    # 内輪 0.021"
    "IR021_0": (209, 0, 1797),
    "IR021_1": (210, 1, 1772),
    "IR021_2": (211, 2, 1750),
    "IR021_3": (212, 3, 1730),
    # 転動体 0.021"
    "B021_0": (222, 0, 1797),
    "B021_1": (223, 1, 1772),
    "B021_2": (224, 2, 1750),
    "B021_3": (225, 3, 1730),
    # 外輪 0.021" @6:00
    "OR021@6_0": (234, 0, 1797),
    "OR021@6_1": (235, 1, 1772),
    "OR021@6_2": (236, 2, 1750),
    "OR021@6_3": (237, 3, 1730),
}


def _looks_like_mat(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(len(MAT_MAGIC)) == MAT_MAGIC


def download(label: str, num: int) -> Path:
    dest = DATA_DIR / f"{num}.mat"
    if dest.exists() and dest.stat().st_size > 0 and _looks_like_mat(dest):
        print(f"  skip {label} ({num}.mat, already exists)")
        return dest
    url = BASE_URL.format(num=num)
    print(f"  get  {label} <- {url}")
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
            expected = resp.headers.get("Content-Length")
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        if expected is not None and tmp.stat().st_size != int(expected):
            raise ValueError(
                f"{url}: サイズ不一致 (期待 {expected} / 実際 {tmp.stat().st_size}) — 途中切断?"
            )
        if not _looks_like_mat(tmp):
            raise ValueError(f"{url}: .mat ヘッダでない応答 (エラーページ?)")
        tmp.rename(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for label, (num, _load, _rpm) in MANIFEST.items():
        try:
            download(label, num)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {label}: {e}", file=sys.stderr)
            return 1
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

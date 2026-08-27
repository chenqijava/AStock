# -*- coding: utf-8 -*-
"""诊断：中市值股金叉→回踩信号 转化率 vs 非名单对照组。"""
import glob
import os
import statistics

import numpy as np
import pandas as pd

MC = set(open("codes_50_200.txt", encoding="utf-8").read().split())
FILES = [f for f in sorted(glob.glob("a_share_daily/*.csv"))
         if os.path.basename(f) != "stock_list.csv"]


def stats(path):
    df = pd.read_csv(path, usecols=["open", "high", "low", "close"])
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    ma5 = c.rolling(5).mean()
    ma24 = c.rolling(24).mean()
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    cross = ((ma5 > ma24) & (ma5.shift(1) <= ma24.shift(1))).fillna(False)
    n = ok = 0
    for i in df.index[cross.to_numpy()]:
        n += 1
        for k in range(i + 1, min(i + 1 + 6, len(df))):
            if c[k] > o[k] and abs(l[k] - ma24[k]) < 0.1 * atr[k]:
                ok += 1
                break
    return n, ok


def main():
    for name, lst in [("中市值", FILES), ("非名单", FILES)]:
        if name == "中市值":
            lst = [f for f in lst if os.path.basename(f).rsplit(".", 1)[0] in MC]
        else:
            lst = [f for f in lst if os.path.basename(f).rsplit(".", 1)[0] not in MC][:1500]
        ns, oks = [], []
        for f in lst:
            n, ok = stats(f)
            if n > 0:
                ns.append(n)
                oks.append(ok)
        print(f"{name}: 有金叉{len(ns)}只 | 金叉中位{statistics.median(ns):.0f} "
              f"| 有合格回踩的金叉中位{statistics.median(oks):.0f} "
              f"| 转化率 {sum(oks) / sum(ns):.1%}")


if __name__ == "__main__":
    main()

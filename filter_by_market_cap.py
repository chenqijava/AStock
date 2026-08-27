# -*- coding: utf-8 -*-
"""
filter_by_market_cap.py — 按流通市值筛选股票名单
=====================================================

用 流通市值 = 成交额(amount) / 换手率(turn%) × 100 从日线数据估算流通市值
(该指标来自真实成交数据，与复权无关，后复权/不复权数据通用)。取每只股票
最近 20 个有效交易日的流通市值中位数，筛出 [min, max] 亿元区间的股票，
输出代码名单(每行一个)，供策略脚本 --codes-file 使用。

用法
----
    python filter_by_market_cap.py --data a_share_daily --min 50 --max 200 --out codes_50_200.txt
    python filter_by_market_cap.py --min 50 --max 200 --workers 8     # 默认读 a_share_daily
"""

import argparse
import glob
import multiprocessing
import os
import sys

import numpy as np
import pandas as pd


def mc_of_one(path):
    """读单只股票，估算流通市值(亿元)。失败/数据不足返回 (code, None)。"""
    code = os.path.basename(path).rsplit(".", 1)[0]
    try:
        df = pd.read_csv(path, usecols=["amount", "turn"])
    except Exception:                                   # noqa: BLE001
        return code, None
    # turn 单位是 %；流通市值(元) = amount / (turn/100) = amount * 100 / turn
    valid = (df["turn"] > 0) & (df["amount"] > 0)
    if int(valid.sum()) < 5:
        return code, None
    mc = (df.loc[valid, "amount"] * 100.0 / df.loc[valid, "turn"]).tail(20)
    return code, float(np.median(mc)) / 1e8             # 亿元


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按流通市值筛选股票名单(流通市值 = 成交额/换手率)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    parser.add_argument("--data", default="a_share_daily", help="日线数据目录(默认 a_share_daily)")
    parser.add_argument("--min", type=float, default=50.0, help="最小流通市值(亿元，默认50)")
    parser.add_argument("--max", type=float, default=200.0, help="最大流通市值(亿元，默认200)")
    parser.add_argument("--out", default="codes_mc.txt", help="输出名单文件(默认 codes_mc.txt)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="并发进程数(默认 min(8,CPU核数))")
    args = parser.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.data, "*.csv"))
                   if os.path.basename(f) != "stock_list.csv")
    print(f"计算 {len(files)} 只股票流通市值...")

    if args.workers > 1:
        with multiprocessing.Pool(args.workers) as pool:
            rows = list(pool.imap_unordered(mc_of_one, files, chunksize=64))
    else:
        rows = [mc_of_one(f) for f in files]

    rows = [r for r in rows if r[1] is not None]
    picked = sorted((code, mc) for code, mc in rows
                    if args.min <= mc <= args.max)
    print(f"有效 {len(rows)} 只 | 区间 [{args.min:.0f}, {args.max:.0f}] 亿元命中 {len(picked)} 只")
    if not picked:
        print("区间内无股票，未生成名单")
        sys.exit(0)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(code for code, _ in picked) + "\n")
    mcs = np.array([mc for _, mc in picked])
    print(f"命中流通市值: {mcs.min():.0f} ~ {mcs.max():.0f} 亿元 | 中位数 {np.median(mcs):.0f} 亿元")
    print(f"名单已保存: {args.out}")


if __name__ == "__main__":
    main()

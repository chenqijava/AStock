# -*- coding: utf-8 -*-
"""
sweep_ma_dual.py — 双均线策略参数扫描(金叉做多/死叉卖出)
========================================================

对 快线(fast_ma) × 慢线(slow_ma) 做网格扫描：数据一次性载入内存、多个
worker 共享，逐组合回测并扣 A 股交易成本，按净盈亏比(PF)排序，标记 PF>1。
每个轴支持“逗号列表”或“区间 [start,stop,step]”；值为单数的轴固定不扫。

用法
----
    python sweep_ma_dual.py --fast "5,20,5" --slow "30,150,10"          # 快5~20步5 × 慢30~150步10
    python sweep_ma_dual.py --fast "5,10,20" --slow "60" --codes-file codes_50_200.txt
"""

import argparse
import glob
import itertools
import multiprocessing
import os
import sys
import time
from statistics import mean

import numpy as np
import pandas as pd

from strategy_ma_dual import backtest_stock, USECOLS   # noqa: E402 复用回测逻辑

_WORKER_DATA = {}


def worker_init(data):
    global _WORKER_DATA
    _WORKER_DATA = data


def load_one(path):
    code = os.path.basename(path).rsplit(".", 1)[0]
    try:
        df = pd.read_csv(path, usecols=USECOLS)
    except Exception:                                  # noqa: BLE001
        return code, None
    df["date"] = df["date"].str.replace("-", "", regex=False).astype(np.int32)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(np.float32)
    df["isST"] = df["isST"].astype(np.int8)
    df = df.sort_values("date").reset_index(drop=True)
    return code, df


def run_stock(args):
    cfg, code = args
    return backtest_stock(cfg, code, _WORKER_DATA[code])


def combo_metrics(trades):
    if not trades:
        return None
    rets = [float(t["ret"]) for t in trades]
    gross = [float(t["gross_ret"]) for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    gw, gl = sum(wins), -sum(losses)
    return {
        "n": len(trades),
        "pf": gw / gl if gl > 0 else float("inf"),
        "avg_ret": mean(rets),
        "win_rate": len(wins) / len(trades),
        "avg_cost": mean(gross) - mean(rets),
        "sum_ret": sum(rets),
    }


def parse_axis(s, dtype):
    vals = [dtype(x.strip()) for x in s.split(",")]
    if (len(vals) == 3 and vals[0] < vals[1]
            and 0 < vals[2] <= vals[1] - vals[0]):
        vals = [vals[0] + k * vals[2]
                for k in range(int((vals[1] - vals[0]) / vals[2]) + 1)]
        if dtype is int:
            vals = [int(round(v)) for v in vals]
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="双均线参数扫描(扣成本，按净PF排序)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    parser.add_argument("--data", default="a_share_daily")
    parser.add_argument("--fast", default="5,20,5", help="快线周期(列表或 start,stop,step)")
    parser.add_argument("--slow", default="30,150,10", help="慢线周期(列表或 start,stop,step)")
    parser.add_argument("--sample", type=int, default=0, help="抽样只数(0=全部)")
    parser.add_argument("--codes-file", default=None,
                        help="只扫该文件内列出的股票代码(每行一个)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--commission", type=float, default=0.00025)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--stamp", type=float, default=0.0005)
    parser.add_argument("--slip", type=float, default=0.001)
    args = parser.parse_args()

    axes = [
        ("ma_f", parse_axis(args.fast, int)),
        ("ma_s", parse_axis(args.slow, int)),
    ]
    active = [(k, v) for k, v in axes if len(v) > 1]
    fixed = {k: v[0] for k, v in axes if len(v) == 1}
    grid = list(itertools.product(*(v for _, v in active))) if active else [()]
    print(f"扫描组合数: {len(grid)}")
    if fixed:
        print("固定参数: " + ", ".join(f"{k}={v}" for k, v in fixed.items()))

    files = sorted(f for f in glob.glob(os.path.join(args.data, "*.csv"))
                   if os.path.basename(f) != "stock_list.csv")
    if args.codes_file:
        keep = {ln.strip() for ln in open(args.codes_file, encoding="utf-8") if ln.strip()}
        files = [f for f in files if os.path.basename(f).rsplit(".", 1)[0] in keep]
    if args.sample and args.sample < len(files):
        files = files[:args.sample]
    print(f"载入 {len(files)} 只股票数据...")

    data = {}
    for f in files:
        code, df = load_one(f)
        if df is not None:
            data[code] = df
    print(f"可用 {len(data)} 只")

    base = {
        "include_st": False, "usecols": USECOLS,
        "capital": args.capital, "commission": args.commission,
        "min_commission": args.min_commission, "stamp": args.stamp,
        "slip": args.slip, "lot": 100,
    }

    t0 = time.time()
    rows = []
    with multiprocessing.Pool(processes=args.workers,
                              initializer=worker_init, initargs=(data,)) as pool:
        codes = sorted(data)
        for gi, combo in enumerate(grid, 1):
            p = dict(zip((k for k, _ in active), combo))
            p.update(fixed)
            cfg = dict(base, fast_ma=p["ma_f"], slow_ma=p["ma_s"])
            trades = []
            tasks = [(cfg, c) for c in codes]
            for res in pool.imap_unordered(run_stock, tasks, chunksize=64):
                trades.extend(res)
            m = combo_metrics(trades)
            rows.append((p, m))
            if gi % 5 == 0 or gi == len(grid):
                print(f"  [{gi}/{len(grid)}] 完成 | 用时{time.time()-t0:.0f}s")

    print(f"\n扫描完成，总用时 {time.time()-t0:.0f}s\n")

    ranked = sorted((r for r in rows if r[1] is not None),
                    key=lambda r: r[1]["pf"], reverse=True)
    print("=" * 88)
    print(f"{'排名':<4}{'快线':<6}{'慢线':<6}"
          f"{'笔数':>7}{'净PF':>7}{'平均净':>9}{'胜率':>8}{'成本':>8}{'净累计':>10}")
    print("-" * 88)
    for i, (p, m) in enumerate(ranked, 1):
        flag = " *" if m["pf"] > 1 else ""
        print(f"{i:<4}{p['ma_f']:<6}{p['ma_s']:<6}"
              f"{m['n']:>7}{m['pf']:>7.2f}{m['avg_ret']:>9.2%}"
              f"{m['win_rate']:>8.1%}{m['avg_cost']:>8.2%}{m['sum_ret']:>10.0%}{flag}")
    print("-" * 80)
    ok = sum(1 for _, m in ranked if m["pf"] > 1)
    print(f"扣成本后净PF>1 的组合: {ok}/{len(ranked)}  (* 标记)")

    out = pd.DataFrame([
        dict(**p, **{k: v for k, v in m.items()}) for p, m in ranked
    ])
    out.to_csv("sweep_ma_dual.csv", index=False, encoding="utf-8-sig")
    print(f"\n已保存: sweep_ma_dual.csv")


if __name__ == "__main__":
    main()
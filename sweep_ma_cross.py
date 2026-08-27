# -*- coding: utf-8 -*-
"""
sweep_ma_cross.py — 参数扫描：寻找扣成本后仍为正收益的参数组合
=============================================================

对 MA快/慢线+贴MA 策略做网格扫描：数据一次性载入内存、多个 worker 共享，
逐参数组合回测并扣 A 股交易成本，按净盈亏比(PF)排序，标记 PF>1 的组合。
每个轴支持“逗号列表”或“区间 [start,stop,step]”；值为单数的轴固定不扫。

用法
----
    python sweep_ma_cross.py --ma-slow "20,90,10" --window "3,10,1"   # 慢线20~90步10 × 窗3~10
    python sweep_ma_cross.py --tp-atr "4,5,6,8" --sl-atr "1.5,2,2.5,3" --touch-div "5,10,15"
    python sweep_ma_cross.py --sample 1500 --workers 8           # 快速抽样扫描
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

from strategy_ma_cross import backtest_stock, USECOLS   # noqa: E402 复用回测逻辑

_WORKER_DATA = {}


def worker_init(data):
    """Pool 初始化：把内存中的全量数据共享给本 worker。"""
    global _WORKER_DATA
    _WORKER_DATA = data


def load_one(path):
    """读单只 CSV 并压缩成紧凑格式(float32/int32)，省内存。"""
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
    """worker 任务：对共享数据里的单只股票回测。"""
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
    """解析扫描轴。逗号列表 "4,5,6,8"；3 个数且符合作区间时按 [start,stop,step] 展开。
    例: "20,90,10" → [20,30,...,90]；"3,10,1" → [3,4,...,10]。"""
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
        description="MA快/慢线+贴MA 参数扫描(扣成本，按净PF排序)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    parser.add_argument("--data", default="a_share_daily")
    parser.add_argument("--ma-fast", default="5", help="快均线(列表或 start,stop,step)")
    parser.add_argument("--ma-slow", default="20,90,10", help="慢均线(列表或 start,stop,step)")
    parser.add_argument("--window", default="3,10,1", help="金叉后找信号K线数(列表或区间)")
    parser.add_argument("--tp-atr", default="4", help="止盈倍数×ATR(列表或区间)")
    parser.add_argument("--sl-atr", default="2", help="止损倍数×ATR(列表或区间)")
    parser.add_argument("--touch-div", default="10",
                        help="贴MA容差: abs(低-MA)<(高-低)/N(列表或区间)")
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--time-stop", type=int, default=10)
    parser.add_argument("--sample", type=int, default=0, help="抽样只数(0=全部)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    # 成本参数(与回测脚本一致)
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--commission", type=float, default=0.00025)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--stamp", type=float, default=0.0005)
    parser.add_argument("--slip", type=float, default=0.001)
    args = parser.parse_args()

    axes = [
        ("ma_f", parse_axis(args.ma_fast, int)),
        ("ma_s", parse_axis(args.ma_slow, int)),
        ("win", parse_axis(args.window, int)),
        ("tp", parse_axis(args.tp_atr, float)),
        ("sl", parse_axis(args.sl_atr, float)),
        ("td", parse_axis(args.touch_div, int)),
    ]
    active = [(k, v) for k, v in axes if len(v) > 1]     # 多值 → 参与网格
    fixed = {k: v[0] for k, v in axes if len(v) == 1}    # 单值 → 固定
    grid = list(itertools.product(*(v for _, v in active))) if active else [()]
    print(f"扫描组合数: {len(grid)}")
    if fixed:
        print("固定参数: " + ", ".join(f"{k}={v}" for k, v in fixed.items()))

    files = sorted(f for f in glob.glob(os.path.join(args.data, "*.csv"))
                   if os.path.basename(f) != "stock_list.csv")
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
        "open_frac": 2.0 / 3.0,
        "atr_period": args.atr_period, "time_stop": args.time_stop,
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
            cfg = dict(base,
                       ma_fast=p["ma_f"], ma_slow=p["ma_s"], window=p["win"],
                       tp_atr=p["tp"], sl_atr=p["sl"], touch_div=p["td"])
            trades = []
            tasks = [(cfg, c) for c in codes]
            for res in pool.imap_unordered(run_stock, tasks, chunksize=32):
                trades.extend(res)
            m = combo_metrics(trades)
            rows.append((p, m))
            if gi % 5 == 0 or gi == len(grid):
                print(f"  [{gi}/{len(grid)}] 完成 | 用时{time.time()-t0:.0f}s")

    print(f"\n扫描完成，总用时 {time.time()-t0:.0f}s\n")

    # 排序并输出
    ranked = sorted((r for r in rows if r[1] is not None),
                    key=lambda r: r[1]["pf"], reverse=True)
    print("=" * 92)
    print(f"{'排名':<4}{'MA慢':<5}{'窗':<4}{'止盈':<5}{'止损':<5}{'贴N':<4}"
          f"{'笔数':>7}{'净PF':>7}{'平均净':>9}{'胜率':>8}{'成本':>8}{'净累计':>10}")
    print("-" * 92)
    for i, (p, m) in enumerate(ranked, 1):
        flag = " *" if m["pf"] > 1 else ""
        print(f"{i:<4}{p['ma_s']:<5}{p['win']:<4}{p['tp']:<5.0f}{p['sl']:<5.1f}{p['td']:<4.0f}"
              f"{m['n']:>7}{m['pf']:>7.2f}{m['avg_ret']:>9.2%}"
              f"{m['win_rate']:>8.1%}{m['avg_cost']:>8.2%}{m['sum_ret']:>10.0%}{flag}")
    print("-" * 80)
    ok = sum(1 for _, m in ranked if m["pf"] > 1)
    print(f"扣成本后净PF>1 的组合: {ok}/{len(ranked)}  (* 标记)")

    # 保存明细
    out = pd.DataFrame([
        dict(**p, **{k: v for k, v in m.items()}) for p, m in ranked
    ])
    out.to_csv("sweep_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n已保存: sweep_results.csv")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
sweep_panic_params.py — 恐慌策略参数敏感性扫描
================================================
对 panic_threshold × min_mc × max_mc 三维网格做扫描。
逐股回测只跑一次，在结果上切片过滤，避免重复计算。

用法:
    python sweep_panic_params.py
    python sweep_panic_params.py --data a_share_daily_hfq --capital 1000000 --notional 20000
"""
import argparse
import glob
import multiprocessing
import os
import re
import sys
import time
import itertools

import numpy as np
import pandas as pd

from strategy_ma_cross import net_return, USECOLS
from strategy_bounce import process_one, USECOLS_VOL
from sim_portfolio import simulate
from strategy_panic import build_files, make_mc_fetcher


def main():
    ap = argparse.ArgumentParser(description="恐慌策略参数敏感性扫描")
    ap.add_argument("--data", default="a_share_daily_hfq")
    ap.add_argument("--universe", default="main")
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--notional", type=float, default=20_000)
    ap.add_argument("--commission", type=float, default=0.00025)
    ap.add_argument("--min-commission", type=float, default=5.0)
    ap.add_argument("--stamp", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.001)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--down-days", type=int, default=15)
    ap.add_argument("--down-thresh", type=float, default=-0.20)
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--max-atr-pct", type=float, default=0.08)
    ap.add_argument("--ma-period", type=int, default=20)
    ap.add_argument("--stop-pct", type=float, default=0.08)
    ap.add_argument("--time-stop", type=int, default=25)
    args = ap.parse_args()

    # 扫描网格
    panic_thresholds = [40, 50, 60, 70, 80]
    min_mcs = [50, 80, 100, 120, 150]
    max_mcs = [300, 500, 800]

    # 个股回测配置(固定)
    cfg = {
        "down_days": args.down_days, "down_thresh": args.down_thresh,
        "ma_period": args.ma_period, "stop_pct": args.stop_pct,
        "time_stop": args.time_stop, "include_st": False,
        "atr_period": args.atr_period,
        "min_atr_drop": 0.0, "max_atr_pct": args.max_atr_pct,
        "vol_n": 0,
        "usecols": USECOLS,
        "capital": args.capital, "commission": args.commission,
        "min_commission": args.min_commission, "stamp": args.stamp,
        "slip": args.slip, "lot": 100,
    }

    files = build_files(args.data, args.universe, None)
    if not files:
        sys.stderr.write("未找到数据文件\n")
        sys.exit(1)

    # 1) 逐股回测(只跑一次)
    t0 = time.time()
    print(f"扫描组合数: {len(panic_thresholds) * len(min_mcs) * len(max_mcs)}")
    print(f"载入 {len(files)} 只股票数据...")
    all_trades = []
    tasks = [(f, cfg) for f in files]
    if args.workers > 1:
        with multiprocessing.Pool(processes=args.workers) as pool:
            for _code, trades in pool.imap_unordered(process_one, tasks, chunksize=16):
                all_trades.extend(trades)
    else:
        for f in files:
            _code, trades = process_one((f, cfg))
            all_trades.extend(trades)
    print(f"可用 {len(files)} 只")
    print(f"个股回测: {len(all_trades)}笔 信号, 用时{time.time()-t0:.0f}s")

    if not all_trades:
        sys.stderr.write("无任何信号\n")
        sys.exit(1)

    df = pd.DataFrame(all_trades)
    df = df.sort_values(["code", "entry_date"]).reset_index(drop=True)

    # 预计算市值(只算一次)
    print("计算流通市值(成交额/换手率×100, 亿元)...")
    mc_of = make_mc_fetcher(args.data)
    all_codes = df["code"].unique()
    mc_map = {}
    for c in all_codes:
        mc_map[c] = mc_of(c)
    df["mc"] = df["code"].map(mc_map)

    # 预计算 density(每个 entry_date 的信号数, 在全板块信号上计算)
    cnt = df["entry_date"].value_counts()
    df["density"] = df["entry_date"].map(cnt)

    # 2) 网格扫描
    results = []
    total = len(panic_thresholds) * len(min_mcs) * len(max_mcs)
    idx = 0
    t1 = time.time()

    for pt, min_mc, max_mc in itertools.product(panic_thresholds, min_mcs, max_mcs):
        idx += 1
        # 恐慌日过滤
        panic = df[df["density"] >= pt].copy()
        if panic.empty:
            continue
        # 市值过滤
        panic = panic[(panic["mc"] >= min_mc) & (panic["mc"] <= max_mc)]
        if panic.empty:
            continue

        panic_days = int(panic["entry_date"].nunique())

        # 组合模拟
        panic["priority"] = panic["mc"]
        res = simulate(panic, args.data, args.capital, args.notional,
                       args.commission, args.min_commission, args.stamp, args.slip,
                       priority_asc=True)

        # 交易层统计
        rets = panic["ret"].to_numpy()
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        pf = (wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")

        results.append({
            "panic_threshold": pt,
            "min_mc": min_mc,
            "max_mc": max_mc,
            "trades": len(panic),
            "panic_days": panic_days,
            "win_rate": len(wins) / len(rets) if len(rets) > 0 else 0,
            "pf": pf,
            "avg_ret": rets.mean() if len(rets) > 0 else 0,
            "total_ret": res["total_ret"],
            "cagr": res["cagr"],
            "mdd": res["mdd"],
            "sharpe": res["sharpe"],
            "fund_rate": res["fund_rate"],
            "max_open": res["max_open"],
            "final_nav": res["nav"][-1],
        })

        if idx % 10 == 0 or idx == total:
            elapsed = time.time() - t1
            print(f"  [{idx}/{total}] 完成 | 用时{elapsed:.0f}s")

    elapsed_total = time.time() - t1
    print(f"\n扫描完成，总用时 {elapsed_total:.0f}s")

    # 3) 输出排名
    rdf = pd.DataFrame(results)
    rdf = rdf.sort_values("cagr", ascending=False)

    print()
    print("=" * 120)
    print(f"{'排名':>4}  {'恐慌阈值':>6}  {'最小市值':>6}  {'最大市值':>6}  "
          f"{'笔数':>6}  {'恐慌日':>4}  {'胜率':>6}  {'PF':>6}  "
          f"{'CAGR':>7}  {'回撤':>7}  {'夏普':>6}  {'命中率':>6}  {'总收益':>8}")
    print("-" * 120)
    for i, (_, r) in enumerate(rdf.iterrows(), 1):
        marker = " *" if r["cagr"] > 0 else ""
        print(f"{i:4d}  {int(r['panic_threshold']):6d}  {int(r['min_mc']):6d}  "
              f"{int(r['max_mc']):6d}  {int(r['trades']):6d}  {int(r['panic_days']):4d}  "
              f"{r['win_rate']:5.1%}  {r['pf']:6.2f}  "
              f"{r['cagr']:+6.1%}  {r['mdd']:6.1%}  {r['sharpe']:6.2f}  "
              f"{r['fund_rate']:5.1%}  {r['total_ret']:+7.1%}{marker}")
    print("-" * 120)
    profitable = (rdf["cagr"] > 0).sum()
    print(f"扣成本后 CAGR>0 的组合: {profitable}/{total}")
    print()

    # 4) 热力图数据(按 panic_threshold × min_mc, max_mc=500 固定)
    print("=" * 80)
    print("热力图: CAGR(%) — panic_threshold(行) × min_mc(列), max_mc=500")
    print("=" * 80)
    heat = rdf[rdf["max_mc"] == 500].pivot_table(
        index="panic_threshold", columns="min_mc", values="cagr"
    )
    print(heat.to_string(float_format=lambda x: f"{x:+.1%}"))
    print()

    print("=" * 80)
    print("热力图: 最大回撤(%) — panic_threshold(行) × min_mc(列), max_mc=500")
    print("=" * 80)
    heat_mdd = rdf[rdf["max_mc"] == 500].pivot_table(
        index="panic_threshold", columns="min_mc", values="mdd"
    )
    print(heat_mdd.to_string(float_format=lambda x: f"{x:.1%}"))
    print()

    print("=" * 80)
    print("热力图: 夏普 — panic_threshold(行) × min_mc(列), max_mc=500")
    print("=" * 80)
    heat_sharpe = rdf[rdf["max_mc"] == 500].pivot_table(
        index="panic_threshold", columns="min_mc", values="sharpe"
    )
    print(heat_sharpe.to_string(float_format=lambda x: f"{x:.2f}"))
    print()

    # 5) 默认参数附近稳定性
    print("=" * 80)
    print("默认参数附近稳定性(panic_threshold=60, min_mc=100, max_mc=500 为基准)")
    print("=" * 80)
    base = rdf[(rdf["panic_threshold"] == 60) & (rdf["min_mc"] == 100) & (rdf["max_mc"] == 500)]
    if not base.empty:
        base_cagr = base.iloc[0]["cagr"]
        base_mdd = base.iloc[0]["mdd"]
        base_sharpe = base.iloc[0]["sharpe"]
        print(f"基准: CAGR {base_cagr:+.1%}  回撤 {base_mdd:.1%}  夏普 {base_sharpe:.2f}")
        print()
        print(f"{'参数变动':>20}  {'CAGR':>7}  {'ΔCAGR':>7}  {'回撤':>7}  {'夏普':>6}")
        for _, r in rdf.iterrows():
            if (r["panic_threshold"] == 60 and r["min_mc"] == 100 and r["max_mc"] == 500):
                continue
            # 只显示离默认参数一步以内的组合
            dt = abs(r["panic_threshold"] - 60)
            dm = abs(r["min_mc"] - 100)
            dx = abs(r["max_mc"] - 500)
            if dt <= 10 and dm <= 20 and dx <= 200:
                delta = r["cagr"] - base_cagr
                label = f"pt={int(r['panic_threshold'])} min={int(r['min_mc'])} max={int(r['max_mc'])}"
                print(f"{label:>20}  {r['cagr']:+6.1%}  {delta:+6.1%}  {r['mdd']:6.1%}  {r['sharpe']:6.2f}")
    print()

    # 保存结果
    out_file = "sweep_panic_params_results.csv"
    rdf.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"已保存: {out_file}")


if __name__ == "__main__":
    main()

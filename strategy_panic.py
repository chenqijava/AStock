# -*- coding: utf-8 -*-
"""
strategy_panic.py — 恐慌日超跌反弹(正式化组合策略)
====================================================

一体式流水线：逐股回测 → 恐慌日过滤 → 组合净值模拟 → 逐年报告。

个股层入场
----------
近 down_days(15) 日跌幅 <= down_thresh(-20%)，当日收阳，且 ATR% <= max_atr_pct(8%)
→ 次日开盘买入。同一股票同时只持一个仓位。

组合层入场(恐慌日)
------------------
当日全市场(限定板块内)触发上述个股信号的股票数 >= panic_threshold(默认60) 才执行。
孤立超跌(信号少的日子)是接飞刀，实测为净亏损，全部放弃——本策略只买"市场恐慌潮"。

恐慌选股(市值偏好)
------------------
在恐慌日的全部信号里，只选流通市值在 [min_mc, max_mc] 区间(默认 100-500亿)的
信号——"出现恐慌选大盘，但别太大"。实测：100亿下界把建仓命中率从28%提到65%
(修复资金不足反选择)、最大回撤19%→9%、夏普0.95→1.26，且 2019 假恐慌年由亏转平
(11年全非负)；500亿上界去掉反弹最弱的大盘股(>500亿均+3.8~4.1% PF2.6~2.9)。
同日建仓默认小市值优先(--mc-order small，100-500亿带内小市值反弹更强)；
--mc-order large 大市值优先，random 随机。

出场
----
- 反弹：收盘 > MA(ma_period=20) → 次日开盘卖出；
- 硬止损：-stop_pct(8%)，防飞刀续杀；
- 时停：time_stop(25) 根未到位 → 收盘平仓。

组合模拟
--------
固定资金 capital、每笔名义本金 notional(A股整手)，逐日盯市 NAV=现金+持仓市值。
资金不足则跳过信号(报告命中率)；同日信号按固定随机种子打乱，避免代码排序偏差。
成本按实际持仓量重算。

板块
----
默认沪深主板(排除 科创688/创业300/北交所bj/ST)。可用 --universe all 或 --codes-file 覆盖；
换板块时 panic_threshold 需按板块信号量级重调。

用法
----
    python strategy_panic.py                          # 主板 + 恐慌60 + 100万资金
    python strategy_panic.py --panic-threshold 50 --capital 2000000
    python strategy_panic.py --universe all --panic-threshold 100
"""

import argparse
import glob
import logging
import multiprocessing
import os
import re
import sys
import time
from statistics import mean

import numpy as np
import pandas as pd

from strategy_ma_cross import net_return, USECOLS      # noqa: E402
from strategy_bounce import process_one, USECOLS_VOL   # noqa: E402 复用个股回测
from sim_portfolio import simulate                       # noqa: E402 复用组合模拟


def build_files(data_dir: str, universe: str, codes_file: str) -> list:
    """按板块/名单构建回测文件列表。universe: main(默认) | all | codes-file优先。"""
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    files = [f for f in files if os.path.basename(f) != "stock_list.csv"]
    if codes_file:
        keep = {ln.strip() for ln in open(codes_file, encoding="utf-8") if ln.strip()}
        files = [f for f in files
                 if os.path.basename(f).rsplit(".", 1)[0] in keep]
    elif universe == "main":
        files = [f for f in files
                 if re.match(r"^(sh\.60|sz\.00)", os.path.basename(f))]
    return files


def make_mc_fetcher(data_dir: str):
    """按 code 取流通市值(亿元) = 成交额/换手率×100 的近期中位数。复权无关，惰性缓存。"""
    cache: dict = {}

    def mc_of(code: str):
        if code in cache:
            return cache[code]
        try:
            df = pd.read_csv(os.path.join(data_dir, code + ".csv"),
                             usecols=["amount", "turn"])
            # 流通市值(元) = amount / (turn/100) = amount*100/turn；取最近20个有效日中位数 → 亿元
            valid = (df["turn"] > 0) & (df["amount"] > 0)
            if int(valid.sum()) < 5:
                cache[code] = np.nan
            else:
                mc = (df.loc[valid, "amount"] * 100.0 / df.loc[valid, "turn"]).tail(20)
                cache[code] = float(np.median(mc)) / 1e8
        except Exception:                                    # noqa: BLE001
            cache[code] = np.nan
        return cache[code]

    return mc_of


def summarize_trades(trades: pd.DataFrame) -> None:
    """交易层汇总：总体 + 逐年。"""
    if trades.empty:
        print("\n无交易(该板块+阈值下无恐慌日信号)。")
        return
    rets = trades["ret"].to_numpy()
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    pf = (wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    print("=" * 62)
    print(f"个股层(已扣成本): {len(trades)}笔  胜率{len(wins)/len(trades):.1%}  "
          f"平均净{rets.mean():+.2%}  PF{pf:.2f}")
    print("=" * 62)
    t = trades.copy()
    t["y"] = t["entry_date"].astype(str).str[:4]
    g = t.groupby("y").agg(笔数=("ret", "size"), 平均净=("ret", "mean"),
                           胜率=("ret", lambda s: (s > 0).mean() * 100))
    print("逐年(交易层):")
    for y, row in g.iterrows():
        print(f"  {y}: {int(row['笔数']):5d}笔  平均净{row['平均净']:+.2%}  胜率{row['胜率']:.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="恐慌日超跌反弹——一体式组合策略(回测+恐慌过滤+组合模拟+逐年)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    ap.add_argument("--data", default="a_share_daily_hfq", help="价格数据目录(hfq)")
    ap.add_argument("--universe", default="main", choices=["main", "all"],
                    help="板块: main=沪深主板(默认, 排除科创/创业/北交) | all=全部")
    ap.add_argument("--codes-file", default=None, help="自定义名单(优先于 --universe)")
    # 个股入场参数
    ap.add_argument("--down-days", type=int, default=15)
    ap.add_argument("--down-thresh", type=float, default=-0.20)
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--max-atr-pct", type=float, default=0.08)
    # 出场参数
    ap.add_argument("--ma-period", type=int, default=20)
    ap.add_argument("--stop-pct", type=float, default=0.08)
    ap.add_argument("--time-stop", type=int, default=25)
    # 恐慌日过滤
    ap.add_argument("--panic-threshold", type=int, default=60,
                    help="当日全市场个股信号数达到该值才执行(默认60)")
    # 盘后量比过滤(标签扫描结论: 恐慌日内放量信号更优; 缩量是毒药)
    ap.add_argument("--vol-min", type=float, default=0.0,
                    help="仅选信号日量比>=该值的信号(信号日成交量/前8日均量; 默认0=不启用)")
    ap.add_argument("--vol-max", type=float, default=0.0,
                    help="仅选信号日量比<=该值的信号(默认0=不启用)")
    # 恐慌选股：市值偏好
    ap.add_argument("--min-mc", type=float, default=100.0,
                    help="仅选流通市值>=该值的信号(亿元, 默认100；0=不过滤)")
    ap.add_argument("--max-mc", type=float, default=500.0,
                    help="仅选流通市值<=该值的信号(亿元, 默认500；0=无上界)")
    ap.add_argument("--mc-order", default="small", choices=["random", "small", "large"],
                    help="恐慌日同日建仓顺序: small=小市值优先(默认) | large=大市值优先 | random=随机")
    # 组合模拟
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--notional", type=float, default=20_000)
    # 成本
    ap.add_argument("--commission", type=float, default=0.00025)
    ap.add_argument("--min-commission", type=float, default=5.0)
    ap.add_argument("--stamp", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.001)
    ap.add_argument("--limit", type=int, default=0, help="调试: 只回测前N只")
    ap.add_argument("--nav-out", default=None, help="净值曲线输出文件(默认 trades_panic_nav_{资金}w.csv)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

    cfg = {
        "down_days": args.down_days, "down_thresh": args.down_thresh,
        "ma_period": args.ma_period, "stop_pct": args.stop_pct,
        "time_stop": args.time_stop, "include_st": False,
        "atr_period": args.atr_period,
        "min_atr_drop": 0.0, "max_atr_pct": args.max_atr_pct,
        "vol_n": 8 if (args.vol_min > 0 or args.vol_max > 0) else 0,
        "usecols": USECOLS_VOL if (args.vol_min > 0 or args.vol_max > 0) else USECOLS,
        "capital": args.capital, "commission": args.commission,
        "min_commission": args.min_commission, "stamp": args.stamp,
        "slip": args.slip, "lot": 100,
    }

    files = build_files(args.data, args.universe, args.codes_file)
    if args.limit and args.limit < len(files):
        files = files[:args.limit]
    if not files:
        sys.stderr.write("未找到数据文件\n")
        sys.exit(1)

    # 1) 逐股回测
    t0 = time.time()
    print(f"板块: {'沪深主板' if args.universe == 'main' else '全部'}({len(files)}只) | "
          f"恐慌阈值 {args.panic_threshold} | 资金 {args.capital/1e4:.0f}万 每笔{args.notional/1e4:.1f}万")
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
    print(f"个股回测: {len(all_trades)}笔 信号, 用时{time.time()-t0:.0f}s")

    if not all_trades:
        sys.stderr.write("无任何信号\n")
        sys.exit(1)

    # 2) 恐慌日过滤(密度在全板块信号上计算，市值过滤只作用于被选中的信号)
    df = pd.DataFrame(all_trades)
    df = df.sort_values(["code", "entry_date"]).reset_index(drop=True)   # 可复现
    cnt = df["entry_date"].value_counts()
    df["density"] = df["entry_date"].map(cnt)
    panic = df[df["density"] >= args.panic_threshold].copy()
    panic_days = int(panic["entry_date"].nunique())

    # 盘后量比过滤(在恐慌日筛选之后, 密度不受影响; 无量比标签的信号保留)
    if args.vol_min > 0 or args.vol_max > 0:
        vr = panic["vol_ratio"]
        keep = ((vr >= args.vol_min) if args.vol_min > 0 else True)
        if args.vol_max > 0:
            keep &= (vr <= args.vol_max)
        kept = int(keep.sum())
        print(f"量比过滤(信号日量/前8日均量 ≥ {args.vol_min:g}"
              + (f" ≤ {args.vol_max:g}" if args.vol_max > 0 else "") + "): "
              f"{len(panic)} -> {kept} 笔")
        panic = panic[keep].copy()
    panic_days = int(panic["entry_date"].nunique())

    # 3) 市值层：计算被选信号流通市值(亿元)，可做区间硬过滤/建仓优先
    print("计算流通市值(成交额/换手率×100, 亿元)...")
    mc_of = make_mc_fetcher(args.data)
    panic["mc"] = [mc_of(c) for c in panic["code"]]
    if args.min_mc > 0 or args.max_mc > 0:
        before = len(panic)
        if args.min_mc > 0:
            panic = panic[panic["mc"] >= args.min_mc]
        if args.max_mc > 0:
            panic = panic[panic["mc"] <= args.max_mc]
        rng_txt = f"[{args.min_mc:.0f}, {args.max_mc:.0f}]" if args.min_mc > 0 and args.max_mc > 0 else \
                  (f">= {args.min_mc:.0f}" if args.min_mc > 0 else f"<= {args.max_mc:.0f}")
        print(f"市值过滤(流通市值 {rng_txt}亿): {before} -> {len(panic)} 笔")
    if panic.empty:
        sys.stderr.write("市值过滤后无信号\n")
        sys.exit(1)
    print(f"恐慌日过滤: 保留 {len(panic)} 笔 (丢弃 {len(df)-len(panic)} 笔孤立信号) | "
          f"恐慌日 {panic_days} 天")

    # 4) 交易层汇总
    summarize_trades(panic)
    panic_out = "trades_panic.csv"
    panic.to_csv(panic_out, index=False, encoding="utf-8-sig")
    print(f"交易明细: {panic_out}")

    # 5) 组合净值模拟
    print()
    print("=" * 62)
    print(f"组合模拟(资金 {args.capital/1e4:.0f}万, 每笔 {args.notional/1e4:.1f}万)")
    print("=" * 62)
    priority_asc = False
    if args.mc_order != "random":
        panic["priority"] = panic["mc"]
        priority_asc = args.mc_order == "small"
        print(f"建仓顺序: {'市值从小到大优先' if priority_asc else '市值从大到小优先'}"
              f"(资金不足时{'小' if priority_asc else '大'}市值先拿仓位)")
    res = simulate(panic, args.data, args.capital, args.notional,
                   args.commission, args.min_commission, args.stamp, args.slip,
                   priority_asc=priority_asc)
    print(f"期末净值        : {res['nav'][-1]/1e4:.1f}万  (初始 {args.capital/1e4:.0f}万)")
    print(f"总收益率        : {res['total_ret']:+.1%}    年化(CAGR) {res['cagr']:+.1%}")
    print(f"最大回撤(日频)  : {res['mdd']:.1%}")
    print(f"建仓 {res['funded']} 笔 / 跳过 {res['skipped']} 笔 (命中率 {res['fund_rate']:.1%})")
    print(f"并发持仓: 最大 {res['max_open']} | 平均 {res['avg_open']:.1f} | 期末 {res['final_open']}")
    print(f"日频夏普(年化)  : {res['sharpe']:.2f}")

    # 5) 逐年资金归属
    navdf = pd.DataFrame({"date": res["dates"], "nav": res["nav"]})
    navdf["y"] = navdf["date"].astype(str).str[:4]
    print()
    print("逐年资金归属(日频净值):")
    for y, g in navdf.groupby("y"):
        r = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1
        md = float(((g["nav"].cummax() - g["nav"]) / g["nav"].cummax()).max())
        print(f"  {y}: 收益{r:+.1%}  年内回撤{md:.1%}  期末{int(g['nav'].iloc[-1]/1e4)}万")

    nav_out = args.nav_out or f"trades_panic_nav_{int(args.capital//10000)}w.csv"
    navdf.drop(columns="y").to_csv(nav_out, index=False, encoding="utf-8-sig")
    print(f"净值曲线已保存: {nav_out}")


if __name__ == "__main__":
    main()

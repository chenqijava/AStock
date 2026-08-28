# -*- coding: utf-8 -*-
"""
sweep_cb_doublelow.py — 可转债双低策略参数扫描
=============================================

扫描 top_n / interval / min_price 等参数组合，汇总最优配置。

输出 CSV 对比表，可按 CAGR / 夏普 / 回撤排序。

用法
----
    python sweep_cb_doublelow.py                            # 默认扫描
    python sweep_cb_doublelow.py --top-n "5,10,15,20" --interval "5,10,20"
    python sweep_cb_doublelow.py --out sweep_cb.csv
"""

import argparse
import itertools
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy_cb_doublelow import load_all, rotation_simulate, cb_universe_on

DEFAULT_TOP_N = [5, 10, 15, 20, 25, 30]
DEFAULT_INTERVAL = [5, 10, 20]
DEFAULT_CAPITAL = 50_000


def main() -> None:
    ap = argparse.ArgumentParser(description="双低策略参数扫描")
    ap.add_argument("--data", default="cb_data", help="转债数据目录")
    ap.add_argument("--top-n", default=",".join(str(x) for x in DEFAULT_TOP_N),
                    help="持有只数列表，逗号分隔(默认 5,10,15,20,25,30)")
    ap.add_argument("--interval", default=",".join(str(x) for x in DEFAULT_INTERVAL),
                    help="调仓间隔交易日列表(默认 5,10,20)")
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL, help="初始资金")
    ap.add_argument("--out", default="sweep_cb.csv", help="输出CSV(默认 sweep_cb.csv)")
    args = ap.parse_args()

    top_ns = [int(x) for x in args.top_n.split(",")]
    intervals = [int(x) for x in args.interval.split(",")]

    print("=" * 62)
    print(f"双低参数扫描 | 数据 {args.data} | 资金 {args.capital/1e4:.0f}万")
    print(f"  top_n: {top_ns}")
    print(f"  interval: {intervals}")

    all_cb = load_all(args.data)
    cal = sorted({d for df in all_cb.values() for d in df["date"]})
    cal = [d for d in cal if d >= "2021-01-01"]
    print(f"  转债 {len(all_cb)} 只 | 日历 {len(cal)} 日")
    print()

    rows = []
    total = len(top_ns) * len(intervals)
    for i, (tn, inte) in enumerate(itertools.product(top_ns, intervals), 1):
        t0 = time.time()
        try:
            res = rotation_simulate(all_cb, cal, tn, inte, args.capital)
            rows.append({"top_n": tn, "interval": inte,
                         "total_ret": res["total_ret"],
                         "cagr": res["cagr"],
                         "mdd": res["mdd"],
                         "sharpe": res["sharpe"],
                         "avg_open": res["avg_open"],
                         "final_nav": res["nav"][-1]})
            print(f"  [{i:2d}/{total}] top_n={tn:2d} interval={inte:2d}  CAGR={res['cagr']:+.1%}  "
                  f"回撤={res['mdd']:.1%} 夏普={res['sharpe']:.2f} 期末={res['nav'][-1]/1e4:.1f}万 | "
                  f"{time.time()-t0:.0f}s")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i:2d}/{total}] top_n={tn:2d} interval={inte:2d}  ERR {exc}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("无有效扫描结果")
        return
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {args.out}")
    print("\n按 CAGR 排序 前5:")
    for _, r in df.sort_values("cagr", ascending=False).head(5).iterrows():
        print(f"  top_n={int(r['top_n']):2d} interval={int(r['interval']):2d}  CAGR={r['cagr']:+.1%}  "
              f"回撤={r['mdd']:.1%} 夏普={r['sharpe']:.2f} 期末={r['final_nav']/1e4:.1f}万")
    print("\n按夏普排序 前5:")
    for _, r in df.sort_values("sharpe", ascending=False).head(5).iterrows():
        print(f"  top_n={int(r['top_n']):2d} interval={int(r['interval']):2d}  CAGR={r['cagr']:+.1%}  "
              f"回撤={r['mdd']:.1%} 夏普={r['sharpe']:.2f} 期末={r['final_nav']/1e4:.1f}万")


if __name__ == "__main__":
    main()
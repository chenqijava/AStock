# -*- coding: utf-8 -*-
"""
strategy_ma_dual.py — 双均线交叉策略回测(金叉做多 / 死叉卖出)
============================================================

规则
----
入场：快线(close 的 fast_ma 日均线)上穿慢线(slow_ma 日均线) —— 金叉，
      下一根 K 线开盘价买入。
出场：快线下穿慢线 —— 死叉，下一根开盘价卖出；最后一天成交换按当日收盘。
      同一时间只持一个仓位；无时停、无止盈止损。
数据与成本与 strategy_ma_cross.py 一致(A股成本模型，默认剔除 ST)。

用法
----
    python strategy_ma_dual.py --data a_share_daily --fast-ma 5 --slow-ma 60 --workers 8
    python strategy_ma_dual.py --fast-ma 10 --slow-ma 30 --codes-file codes_50_200.txt
"""

import argparse
import glob
import logging
import multiprocessing
import os
import sys
import time
from statistics import mean

import numpy as np
import pandas as pd

from strategy_ma_cross import net_return, USECOLS   # noqa: E402 复用成本模型


def backtest_stock(cfg: dict, code: str, df: pd.DataFrame) -> list:
    """对单只股票跑双均线金叉/死叉，返回交易记录列表。"""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    if n < cfg["slow_ma"] + 5:
        return []

    open_, close = df["open"], df["close"]
    st = df["isST"] if "isST" in df.columns else None

    fast = close.rolling(cfg["fast_ma"]).mean()
    slow = close.rolling(cfg["slow_ma"]).mean()
    cross_up = ((fast > slow) & (fast.shift(1) <= slow.shift(1))).fillna(False).to_numpy()
    cross_dn = ((fast < slow) & (fast.shift(1) >= slow.shift(1))).fillna(False).to_numpy()

    trades = []
    last_exit = -1
    for i in np.nonzero(cross_up)[0]:
        if i <= last_exit:
            continue                     # 持仓中，跳过
        if st is not None and st[i] == 1 and not cfg["include_st"]:
            continue                     # 跳过 ST 信号
        entry_bar = i + 1                # 下一根开盘买入
        if entry_bar >= n:
            break
        entry_price = float(open_[entry_bar])

        # 死叉离场：下一根开盘卖出；最后一根触发则按当日收盘
        reason = "EXIT"
        exit_bar = exit_price = None
        for j in np.nonzero(cross_dn[entry_bar:])[0]:
            jj = entry_bar + j
            if jj + 1 >= n:
                exit_bar, exit_price = n - 1, float(close[n - 1])
            else:
                exit_bar, exit_price = jj + 1, float(open_[jj + 1])
            break
        else:                            # 到样本末尾仍未死叉，按最后收盘平仓
            reason = "END"
            exit_bar, exit_price = n - 1, float(close[n - 1])

        trades.append({
            "code": code,
            "entry_date": df["date"][entry_bar],
            "entry_price": round(entry_price, 4),
            "exit_date": df["date"][exit_bar],
            "exit_price": round(exit_price, 4),
            "reason": reason,
            "bars": exit_bar - entry_bar + 1,
            "gross_ret": round(exit_price / entry_price - 1, 6),
            "ret": round(net_return(entry_price, exit_price, cfg), 6),
        })
        last_exit = exit_bar
    return trades


def process_one(args):
    path, cfg = args
    code = os.path.basename(path).rsplit(".", 1)[0]
    try:
        df = pd.read_csv(path, usecols=cfg["usecols"])
    except Exception as exc:                   # noqa: BLE001
        logging.warning("%s 读取失败: %s", code, exc)
        return code, []
    return code, backtest_stock(cfg, code, df)


def summarize(trades: list) -> None:
    if not trades:
        print("\n未产生任何交易。")
        return
    rets = [t["ret"] for t in trades]
    gross = [t["gross_ret"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    gross_win, gross_loss = sum(wins), -sum(losses)

    print("=" * 62)
    print("总体结果(已扣交易成本)")
    print("=" * 62)
    print(f"交易笔数        : {len(trades)}")
    print(f"胜率            : {len(wins) / len(trades):.1%}  (盈利 {len(wins)} / 亏损 {len(losses)})")
    print(f"平均单笔净收益  : {mean(rets):.2%}   平均盈利 {mean(wins):.2%} / 平均亏损 {mean(losses):.2%}")
    print(f"平均每笔成本    : {mean(gross) - mean(rets):.2%}")
    print(f"累计收益(简单和): {sum(rets):.2%}")
    print(f"盈亏比(ProfitFactor): {gross_win / gross_loss if gross_loss > 0 else float('inf'):.2f}")
    print(f"平均持仓K线数   : {mean(t['bars'] for t in trades):.1f}")
    print()
    print("=" * 62)
    print("按出场方式")
    print("=" * 62)
    print(f"{'出场':<6}{'笔数':>6}{'占比':>8}{'平均收益':>10}{'胜率':>9}")
    for reason in ("EXIT", "END"):
        sub = [t for t in trades if t["reason"] == reason]
        if not sub:
            continue
        sub_ret = [t["ret"] for t in sub]
        sr_wins = sum(1 for r in sub_ret if r > 0)
        print(f"{reason:<6}{len(sub):>6}{len(sub) / len(trades):>8.1%}"
              f"{mean(sub_ret):>10.2%}{sr_wins / len(sub):>9.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="双均线交叉回测(金叉做多/死叉卖出)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    parser.add_argument("--data", default="a_share_daily")
    parser.add_argument("--fast-ma", type=int, default=5, help="快线周期(默认5)")
    parser.add_argument("--slow-ma", type=int, default=60, help="慢线周期(默认60)")
    parser.add_argument("--include-st", action="store_true", help="包含ST股票(默认剔除)")
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--commission", type=float, default=0.00025)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--stamp", type=float, default=0.0005)
    parser.add_argument("--slip", type=float, default=0.001)
    parser.add_argument("--limit", type=int, default=0, help="只回测前 N 只(0=全部)")
    parser.add_argument("--codes-file", default=None,
                        help="只回测该文件内列出的股票代码(每行一个)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

    cfg = {
        "fast_ma": args.fast_ma, "slow_ma": args.slow_ma,
        "include_st": args.include_st, "usecols": USECOLS,
        "capital": args.capital, "commission": args.commission,
        "min_commission": args.min_commission, "stamp": args.stamp,
        "slip": args.slip, "lot": 100,
    }

    files = sorted(glob.glob(os.path.join(args.data, "*.csv")))
    files = [f for f in files if os.path.basename(f) != "stock_list.csv"]
    if args.codes_file:
        keep = {ln.strip() for ln in open(args.codes_file, encoding="utf-8") if ln.strip()}
        files = [f for f in files if os.path.basename(f).rsplit(".", 1)[0] in keep]
    if args.limit and args.limit < len(files):
        files = files[:args.limit]
    if not files:
        sys.stderr.write("未找到数据文件，请检查 --data 目录\n")
        sys.exit(1)

    print(f"参数: 快线{args.fast_ma} / 慢线{args.slow_ma} | 金叉次日开盘买入/死叉次日开盘卖出 | "
          f"{'含ST' if args.include_st else '剔除ST'}")
    print(f"成本: 佣金{args.commission:.4%}/边(最低{args.min_commission:.0f}元) 印花税{args.stamp:.4%}(卖) "
          f"滑点{args.slip:.2%}/边 每笔投入{args.capital:.0f}元")
    print(f"共 {len(files)} 只股票，并发 {args.workers} 进程...")

    t0 = time.time()
    tasks = [(f, cfg) for f in files]
    all_trades = []
    if args.workers > 1:
        with multiprocessing.Pool(processes=args.workers) as pool:
            for _code, trades in pool.imap_unordered(process_one, tasks, chunksize=16):
                all_trades.extend(trades)
    else:
        for f in files:
            _code, trades = process_one((f, cfg))
            all_trades.extend(trades)
    print(f"回测完成，用时 {time.time() - t0:.0f}s")

    summarize(all_trades)
    out = "trades_ma_dual.csv"
    if all_trades:
        pd.DataFrame(all_trades).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n交易明细已保存: {out}")


if __name__ == "__main__":
    main()
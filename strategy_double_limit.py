# -*- coding: utf-8 -*-
"""
strategy_double_limit.py — 双炮涨停策略回测
============================================================

规则
----
触发(买入信号)：
  1. 当日出现涨停(最高价触及板块涨停价：主板10% / 创业板·科创板20% /
     北交所30% / ST 5%)；
  2. 往前 lookback(默认20)根 K 线内存在另一根涨停，且两根涨停 K 线的
     [最低价, 最高价] 区间重合比例 >= overlap(默认50%)，
     重合比例 = 重叠长度 / 两根中较短K线的区间长度；
  3. 同时满足则于下一根 K 线开盘价买入。
  同一时间只持一个仓位(上次平仓前的涨停信号跳过)。

出场：
  - 追踪止损：跟踪自进场起的最高价，从最高点回撤 trail(默认5%)即离场；
    跳空低开跌破止损线按开盘价成交，盘中触及按止损线成交。
  - 无时停；若到数据末尾仍未回撤到位，按最后收盘价平仓(记为 END)。

数据与成本
----------
数据默认读取 a_share_daily_hfq/*.csv(后复权，价格连续)；默认剔除 ST
(--include-st 放开)。成本模型与 strategy_ma_cross.py 一致(佣金/印花税/
滑点/最低佣金)，每笔净收益已扣成本。

用法
----
    python strategy_double_limit.py --data a_share_daily_hfq --workers 8
    python strategy_double_limit.py --lookback 20 --overlap 0.5 --trail 0.05
    python strategy_double_limit.py --codes-file codes_50_200.txt   # 只跑市值过滤名单
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


# ---------------------------------------------------------------------------
# 单只股票回测
# ---------------------------------------------------------------------------
def backtest_stock(cfg: dict, code: str, df: pd.DataFrame) -> list:
    """对单只股票跑双炮涨停策略，返回交易记录列表。"""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    if n < cfg["lookback"] + 5:
        return []

    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    st = df["isST"] if "isST" in df.columns else None
    prev_close = close.shift(1)

    # 涨停幅度(按板块)
    digits = code.split(".")[1] if "." in code else code
    if digits.startswith(("43", "83", "87", "92")):
        lim = 0.30
    elif digits.startswith(("30", "68")):
        lim = 0.20
    else:
        lim = 0.10

    # 涨停判定：最高价触及涨停价(ST 股按 5%)
    lim_up = (high >= prev_close * (1 + lim) - 1e-6).fillna(False).to_numpy()
    if st is not None:
        lim_up |= (st.to_numpy() == 1) & (
            high >= prev_close * 1.05 - 1e-6).fillna(False).to_numpy()

    lookback = cfg["lookback"]
    overlap = cfg["overlap"]
    trades = []
    last_exit = -1
    for i in range(1, n):
        if i <= last_exit:
            continue
        if not lim_up[i]:
            continue
        if st is not None and st[i] == 1 and not cfg["include_st"]:
            continue                     # 跳过 ST 信号

        # 往前找一根涨停，要求区间重合 >= overlap
        found = None
        for p in range(max(1, i - lookback), i):
            if not lim_up[p]:
                continue
            if st is not None and st[p] == 1 and not cfg["include_st"]:
                continue
            olap = min(high[i], high[p]) - max(low[i], low[p])
            if olap <= 0:
                continue
            rng1, rng2 = high[i] - low[i], high[p] - low[p]
            if rng1 > 0 and rng2 > 0 and olap / min(rng1, rng2) >= overlap:
                found = p
                break
        if found is None:
            continue

        entry_bar = i + 1                # 下一根开盘买入
        if entry_bar >= n:
            break
        entry_price = open_[entry_bar]

        # 追踪止损：从最高点回撤 trail 比例离场
        peak = high[entry_bar]
        exit_price = exit_bar = None
        reason = "TRAIL"
        for j in range(entry_bar + 1, n):
            if high[j] > peak:
                peak = high[j]
            stop = peak * (1 - cfg["trail"])
            o, l = open_[j], low[j]
            if o <= stop:                # 跳空低开跌破止损线，按开盘价
                exit_price, exit_bar = o, j
                break
            if l <= stop:                # 盘中触及止损线，按止损价
                exit_price, exit_bar = stop, j
                break
        else:                            # 到样本末尾仍未触发，按最后收盘平仓
            exit_bar = n - 1
            exit_price = close[exit_bar]
            reason = "END"

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
    """多进程 worker：读取一只股票 CSV 并回测。"""
    path, cfg = args
    code = os.path.basename(path).rsplit(".", 1)[0]
    try:
        df = pd.read_csv(path, usecols=cfg["usecols"])
    except Exception as exc:                   # noqa: BLE001
        logging.warning("%s 读取失败: %s", code, exc)
        return code, []
    return code, backtest_stock(cfg, code, df)


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize(trades: list) -> None:
    if not trades:
        print("\n未产生任何交易。可尝试 --lookback/--overlap 放宽或检查数据。")
        return

    rets = [t["ret"] for t in trades]
    gross = [t["gross_ret"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    gross_win, gross_loss = sum(wins), -sum(losses)
    eq = [1.0]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    peak, mdd = eq[0], 0.0
    for v in eq:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)

    print("=" * 62)
    print("总体结果(已扣交易成本)")
    print("=" * 62)
    print(f"交易笔数        : {len(trades)}")
    print(f"胜率            : {len(wins) / len(trades):.1%}  (盈利 {len(wins)} / 亏损 {len(losses)})")
    print(f"平均单笔净收益  : {mean(rets):.2%}   平均盈利 {mean(wins):.2%} / 平均亏损 {mean(losses):.2%}")
    print(f"平均每笔成本    : {mean(gross) - mean(rets):.2%}   (佣金+印花税+滑点，含最低佣金)")
    print(f"累计收益(简单和): {sum(rets):.2%}")
    print(f"盈亏比(ProfitFactor): {gross_win / gross_loss if gross_loss > 0 else float('inf'):.2f}")
    print(f"平均持仓K线数   : {mean(t['bars'] for t in trades):.1f}")
    print(f"顺序复利最大回撤: {mdd:.2%}")
    print()

    print("=" * 62)
    print("按出场方式")
    print("=" * 62)
    print(f"{'出场':<6}{'笔数':>6}{'占比':>8}{'平均收益':>10}{'胜率':>9}")
    for reason in ("TRAIL", "END"):
        sub = [t for t in trades if t["reason"] == reason]
        if not sub:
            continue
        sub_ret = [t["ret"] for t in sub]
        sr_wins = sum(1 for r in sub_ret if r > 0)
        print(f"{reason:<6}{len(sub):>6}{len(sub) / len(trades):>8.1%}"
              f"{mean(sub_ret):>10.2%}{sr_wins / len(sub):>9.1%}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="双炮涨停策略回测(两根区间重合涨停→次日买入→最高点回撤5%离场)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    parser.add_argument("--data", default="a_share_daily_hfq",
                        help="数据目录(默认 a_share_daily_hfq，后复权)")
    parser.add_argument("--lookback", type=int, default=20,
                        help="往前找另一根涨停的回看K线数(默认20)")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="两根涨停区间重合比例(默认0.5=50%)")
    parser.add_argument("--trail", type=float, default=0.05,
                        help="最高点回撤离场比例(默认5%)")
    parser.add_argument("--include-st", action="store_true", help="包含ST股票(默认剔除)")
    parser.add_argument("--capital", type=float, default=10000,
                        help="每笔投入金额(元，默认10000)")
    parser.add_argument("--commission", type=float, default=0.00025,
                        help="佣金率/边(默认万2.5)")
    parser.add_argument("--min-commission", type=float, default=5.0,
                        help="单笔最低佣金(元，默认5)")
    parser.add_argument("--stamp", type=float, default=0.0005,
                        help="印花税率(仅卖出，默认千0.5)")
    parser.add_argument("--slip", type=float, default=0.001,
                        help="滑点率/边(默认0.1%)")
    parser.add_argument("--limit", type=int, default=0, help="只回测前 N 只(0=全部)")
    parser.add_argument("--codes-file", default=None,
                        help="只回测该文件内列出的股票代码(每行一个，如 sh.600000)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="并发进程数(默认 min(8,CPU核数))")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

    cfg = {
        "lookback": args.lookback, "overlap": args.overlap,
        "trail": args.trail, "include_st": args.include_st,
        "usecols": USECOLS,
        "capital": args.capital, "commission": args.commission,
        "min_commission": args.min_commission, "stamp": args.stamp,
        "slip": args.slip, "lot": 100,
    }

    files = sorted(glob.glob(os.path.join(args.data, "*.csv")))
    files = [f for f in files if os.path.basename(f) != "stock_list.csv"]
    if args.codes_file:
        keep = {ln.strip() for ln in open(args.codes_file, encoding="utf-8")
                if ln.strip()}
        files = [f for f in files
                 if os.path.basename(f).rsplit(".", 1)[0] in keep]
    if args.limit and args.limit < len(files):
        files = files[:args.limit]
    if not files:
        sys.stderr.write("未找到数据文件，请检查 --data 目录\n")
        sys.exit(1)

    print(f"参数: 回看{args.lookback}根 重合>={args.overlap:.0%} 出场:最高点回撤{args.trail:.0%} | "
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

    out = "trades_double_limit.csv"
    if all_trades:
        pd.DataFrame(all_trades).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n交易明细已保存: {out}")


if __name__ == "__main__":
    main()

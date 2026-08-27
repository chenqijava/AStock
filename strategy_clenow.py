# -*- coding: utf-8 -*-
"""
strategy_clenow.py — 克列诺《Following the Trend》趋势跟踪回测(A股·逐股·只做多)
==============================================================================

规则(依据 Andreas Clenow《趋势跟踪 Following the Trend》公开规则，
按 A 股单股可执行性适配)
----------------------------------------------------------------------
入场：
  收盘价向上突破前 entry_lookback(默认55) 日最高价(不含当日) → 下一根
  开盘价买入。离场后若再次突破可再次入场；同一时间只持一个仓位。
出场：
  收盘价跌破前 exit_lookback(默认20) 日最低价(不含当日) → 下一根开盘价
  卖出。与书一致：无止损、无时停，仅靠通道离场。
  若到数据末尾仍未离场，按最后收盘价平仓(END)。

与书原版的差异(A 股适配)
------------------------
* 书为多市场期货组合、多空双向、按波动率等风险配仓；本脚本按 A 股现实改为
  逐股独立、只做多、每笔固定资金(capital)投入。
* 未建模涨停无法买入、跌停无法卖出等执行限制。
* 数据默认读取不复权日线(a_share_daily)：除权缺口可能造成少量假突破/假离场
  (趋势策略对此敏感)，建议后复权数据(hfq)下载完成后复跑核验。

用法
----
    python strategy_clenow.py --data a_share_daily --workers 8
    python strategy_clenow.py --entry-lookback 55 --exit-lookback 20
    python strategy_clenow.py --limit 300          # 调试：只看前300只
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
    """对单只股票跑克列诺趋势跟踪，返回交易记录列表(只做多)。"""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    if n < cfg["entry_lookback"] + 5:
        return []

    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    st = df["isST"] if "isST" in df.columns else None

    # 入场：收盘 > 前 entry_lookback 日最高价(不含当日)；离场：收盘 < 前 exit_lookback 日最低价
    prior_high = high.rolling(cfg["entry_lookback"]).max().shift(1)
    signal = (close > prior_high).fillna(False).to_numpy()
    prior_low = low.rolling(cfg["exit_lookback"]).min().shift(1)

    trades = []
    last_exit = -1
    for i in np.nonzero(signal)[0]:
        if i <= last_exit:
            continue                     # 持仓中，跳过
        if st is not None and st[i] == 1 and not cfg["include_st"]:
            continue                     # 跳过 ST 信号
        entry_bar = i + 1                # 下一根开盘买入
        if entry_bar >= n:
            break
        entry_price = float(open_[entry_bar])

        # 出场：收盘跌破前 exit_lookback 日最低价 → 下一根开盘卖出
        reason = "EXIT"
        exit_bar = exit_price = None
        for j in range(entry_bar + 1, n):
            if close[j] < prior_low[j]:
                if j + 1 >= n:           # 最后一根触发，按当日收盘平仓
                    exit_bar, exit_price = n - 1, float(close[n - 1])
                else:
                    exit_bar, exit_price = j + 1, float(open_[j + 1])
                break
        else:                            # 到样本末尾仍未离场，按最后收盘平仓
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
        print("\n未产生任何交易。可尝试 --entry-lookback 调小或检查数据。")
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
    for reason in ("EXIT", "END"):
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
        description="克列诺《Following the Trend》趋势跟踪回测(55日突破入场/20日离场，A股只做多)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    parser.add_argument("--data", default="a_share_daily",
                        help="数据目录(默认 a_share_daily 不复权)")
    parser.add_argument("--entry-lookback", type=int, default=55,
                        help="突破入场回看日数(默认55，书原版)")
    parser.add_argument("--exit-lookback", type=int, default=20,
                        help="通道离场回看日数(默认20，书原版)")
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
        "entry_lookback": args.entry_lookback, "exit_lookback": args.exit_lookback,
        "include_st": args.include_st, "usecols": USECOLS,
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

    print(f"参数: 突破{args.entry_lookback}日入场/跌破{args.exit_lookback}日低点离场 | 无止损无时停 | "
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

    out = "trades_clenow.csv"
    if all_trades:
        pd.DataFrame(all_trades).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n交易明细已保存: {out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
strategy_ma_cross.py — MA5 上穿 MA24 + 回踩 MA24 择时策略回测
=============================================================

策略规则
--------
买入(全部满足才买)：
  1. MA5 上穿 MA24(金叉)；
  2. 金叉后 window(默认6)根 K 线内，出现一根满足以下条件的 K 线：
       - 阳线：收盘价 > 开盘价
       - 强势开盘：开盘价 - 最低价 >= (最高价 - 最低价) × 2/3
       - 贴MA24：abs(最低价 - MA24) < (最高价 - 最低价) / 10
  3. 出现信号后，于下一根 K 线开盘价买入(不追高，次根开盘成交)。

卖出(满足任一即离场，同根多条件按止损优先)：
  - 4×ATR 止盈：价格 >= 买入价 + tp_atr×ATR，按止盈价成交；
  - 2×ATR 止损：价格 <= 买入价 - sl_atr×ATR，按止损价成交；
  - 时停：持有 time_stop(默认10)根 K 线后仍未触发前几条，按该根收盘价平仓；
  - 涨停：当日最高价触及板块涨停价(主板10%/创业板·科创板20%/北交所30%/ST 5%)，
    按当日收盘价离场，落袋为安。
  ATR 取“买入那根 K 线”的 ATR(默认 14 期，Wilder 平滑)。

数据
----
默认读取 download_a_share_daily.py 生成的 a_share_daily/*.csv(17 列)。
默认剔除 ST 股票(--include-st 可放开)；数据为当前仍在交易的股票(无退市股，
存在幸存者偏差，结果偏乐观，仅作策略研究参考)。

用法
----
    python strategy_ma_cross.py
    python strategy_ma_cross.py --data D:/量化/AStock/a_share_daily --workers 8
    python strategy_ma_cross.py --limit 300          # 只看前300只(调试)
    python strategy_ma_cross.py --window 10 --tp-atr 6 --sl-atr 3
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

USECOLS = ["date", "open", "high", "low", "close", "isST"]


# ---------------------------------------------------------------------------
# 成本模型(A股)
# ---------------------------------------------------------------------------
def net_return(entry: float, exit_: float, cfg: dict) -> float:
    """按“固定投入 × 整手(100股)”建模，扣佣金/印花税/滑点后返回净收益率。

    - 佣金: 双边, 按成交额×commission, 每笔最低 min_commission 元
    - 印花税: 仅卖出, stamp × 卖出额
    - 滑点: 双边, slip × 成交额(近似)
    """
    lot = cfg["lot"]
    shares = max(lot, int(cfg["capital"] // entry // lot) * lot)
    buy_value = entry * shares
    sell_value = exit_ * shares
    buy_fee = (max(cfg["commission"] * buy_value, cfg["min_commission"])
               + cfg["slip"] * buy_value)
    sell_fee = (max(cfg["commission"] * sell_value, cfg["min_commission"])
                + (cfg["stamp"] + cfg["slip"]) * sell_value)
    return (sell_value - sell_fee - buy_value - buy_fee) / buy_value


# ---------------------------------------------------------------------------
# 单只股票回测
# ---------------------------------------------------------------------------
def backtest_stock(cfg: dict, code: str, df: pd.DataFrame) -> list:
    """对单只股票跑策略，返回交易记录列表。"""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    if n < cfg["ma_slow"] + 5:                 # 历史太短(次新股)，指标无意义
        return []

    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    st = df["isST"] if "isST" in df.columns else None

    ma_fast = close.rolling(cfg["ma_fast"]).mean()
    ma_slow = close.rolling(cfg["ma_slow"]).mean()

    # ATR14(Wilder)：ewm(alpha=1/n, adjust=False) 即 Wilder 平滑
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / cfg["atr_period"], adjust=False).mean()

    # 金叉位置(向量化)：ma_fast[i-1]<=ma_slow[i-1] 且 ma_fast[i]>ma_slow[i]
    cross = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
    cross_idx = [int(i) for i in df.index[cross.fillna(False)].tolist()]

    trades = []
    last_exit = -1                             # 上次平仓位置，避免重复建仓
    win = cfg["window"]
    for c in cross_idx:
        if c <= last_exit:
            continue

        # 金叉后 win 根内找“阳线 + 强势开盘 + 贴MA24”信号
        entry_bar = None
        for k in range(c + 1, min(c + 1 + win, n)):
            if st is not None and st[k] == 1 and not cfg["include_st"]:
                continue                       # 跳过 ST 信号
            rng = high[k] - low[k]
            if (close[k] > open_[k]                       # 阳线
                    and (open_[k] - low[k]) >= cfg["open_frac"] * rng   # 开盘位于K线上部
                    and abs(low[k] - ma_slow[k]) < rng / cfg["touch_div"]):  # 低点紧贴MA24
                entry_bar = k + 1              # 下一根开盘买入
                break
        if entry_bar is None or entry_bar >= n:
            continue

        entry_price = open_[entry_bar]
        atr_e = atr[entry_bar]
        if not np.isfinite(atr_e) or atr_e <= 0:
            continue
        tp = entry_price + cfg["tp_atr"] * atr_e
        sl = entry_price - cfg["sl_atr"] * atr_e

        # 涨停幅度(按板块)：ST 5% / 北交所 30% / 创业板·科创板 20% / 主板 10%
        digits = code.split(".")[1] if "." in code else code
        if digits.startswith(("43", "83", "87", "92")):
            lim = 0.30
        elif digits.startswith(("30", "68")):
            lim = 0.20
        else:
            lim = 0.10

        # 出场：4×ATR止盈 / 2×ATR止损 / 时停10根 / 涨停落袋。同根多条件按止损优先。
        # 涨停判定：当日最高价触及板块涨停价，按当日收盘价成交(该日收盘通常在涨停价)。
        exit_price = exit_bar = None
        reason = ""
        hold_end = min(entry_bar + cfg["time_stop"], n)
        for j in range(entry_bar, hold_end):
            o, h, l = open_[j], high[j], low[j]
            if o <= sl:                        # 跳空低开破止损，按开盘价
                exit_price, reason, exit_bar = o, "SL", j
                break
            if o >= tp:                        # 跳空高开过止盈，按开盘价
                exit_price, reason, exit_bar = o, "TP", j
                break
            if l <= sl:                        # 盘中触止损，按止损价
                exit_price, reason, exit_bar = sl, "SL", j
                break
            if h >= tp:                        # 盘中触止盈，按止盈价
                exit_price, reason, exit_bar = tp, "TP", j
                break
            llim = 0.05 if (st is not None and st[j] == 1) else lim
            if h >= prev_close[j] * (1 + llim) - 1e-6:   # 涨停，按当日收盘落袋
                exit_price, reason, exit_bar = close[j], "LIMIT", j
                break
        else:                                  # 时停：按最后一根收盘价平仓
            exit_bar = hold_end - 1
            exit_price = close[exit_bar]
            reason = "TIME"

        trades.append({
            "code": code,
            "entry_date": df["date"][entry_bar],
            "entry_price": round(entry_price, 4),
            "exit_date": df["date"][exit_bar],
            "exit_price": round(exit_price, 4),
            "reason": reason,
            "gross_ret": round(exit_price / entry_price - 1, 6),
            "ret": round(net_return(entry_price, exit_price, cfg), 6),
            "bars": exit_bar - entry_bar + 1,
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
        print("\n未产生任何交易。可尝试放宽 window/pullback 或检查参数。")
        return

    rets = [t["ret"] for t in trades]                 # 净收益(含成本)
    gross_rets = [t["gross_ret"] for t in trades]
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

    avg_cost = mean(gross_rets) - mean(rets)          # 平均每笔交易成本
    print("=" * 62)
    print("总体结果(已扣交易成本)")
    print("=" * 62)
    print(f"交易笔数        : {len(trades)}")
    print(f"胜率            : {len(wins) / len(trades):.1%}  (盈利 {len(wins)} / 亏损 {len(losses)})")
    print(f"平均单笔净收益  : {mean(rets):.2%}   平均盈利 {mean(wins):.2%} / 平均亏损 {mean(losses):.2%}")
    print(f"平均每笔成本    : {avg_cost:.2%}   (佣金+印花税+滑点，含最低佣金)")
    print(f"累计收益(简单和): {sum(rets):.2%}   (每笔净收益直接相加，近似多仓位组合)")
    print(f"单账户顺序复利  : {np.prod([1 + r for r in rets]) - 1:.2%}   (全部交易串进一个账户，仅理论参考)")
    print(f"盈亏比(ProfitFactor): {gross_win / gross_loss if gross_loss > 0 else float('inf'):.2f}")
    print(f"平均持仓K线数   : {mean(t['bars'] for t in trades):.1f}")
    print(f"顺序复利最大回撤: {mdd:.2%}")
    print()

    print("=" * 62)
    print("按出场方式")
    print("=" * 62)
    print(f"{'出场':<6}{'笔数':>6}{'占比':>8}{'平均收益':>10}{'胜率':>9}")
    for reason in ("TP", "SL", "TIME", "LIMIT"):
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
        description="MA5上穿MA24+回踩MA24 策略回测(数据来自 a_share_daily)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    parser.add_argument("--data", default="a_share_daily", help="数据目录(默认 a_share_daily)")
    parser.add_argument("--ma-fast", type=int, default=5, help="快均线周期(默认5)")
    parser.add_argument("--ma-slow", type=int, default=24, help="慢均线周期(默认24)")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR 周期(默认14)")
    parser.add_argument("--window", type=int, default=6, help="金叉后找信号的K线数(默认6)")
    parser.add_argument("--open-frac", type=float, default=2.0 / 3.0,
                        help="开盘须位于K线区间上部比例(默认2/3)")
    parser.add_argument("--touch-div", type=int, default=10,
                        help="贴MA24容差: abs(低-MA24)<(高-低)/N(默认10)")
    parser.add_argument("--tp-atr", type=float, default=4.0, help="止盈倍数×ATR(默认4)")
    parser.add_argument("--sl-atr", type=float, default=2.0, help="止损倍数×ATR(默认2)")
    parser.add_argument("--time-stop", type=int, default=10, help="时停K线数(默认10)")
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
        "ma_fast": args.ma_fast, "ma_slow": args.ma_slow,
        "atr_period": args.atr_period, "window": args.window,
        "open_frac": args.open_frac, "touch_div": args.touch_div,
        "tp_atr": args.tp_atr,
        "sl_atr": args.sl_atr, "time_stop": args.time_stop,
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

    print(f"参数: MA{args.ma_fast}/{args.ma_slow} | 窗口{args.window} | "
          f"入场:阳线+开盘上部{args.open_frac:.2f}+贴MA24<区间/{args.touch_div} | "
          f"止盈{args.tp_atr}×ATR 止损{args.sl_atr}×ATR 时停{args.time_stop}根 +涨停 | "
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

    out = "trades_ma_cross.csv"
    if all_trades:
        pd.DataFrame(all_trades).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n交易明细已保存: {out}")


if __name__ == "__main__":
    main()

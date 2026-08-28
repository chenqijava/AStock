# -*- coding: utf-8 -*-
"""
strategy_bounce.py — 超跌反弹策略回测(均值回归 / 低吸)
======================================================

逻辑
----
入场(超跌 + 企稳)：
  1. 近 down_days 日累计跌幅 <= down_thresh(默认 10 日跌 15%) —— 超跌；
  2. 当日收阳线(close > open) —— 企稳反弹确认。
  → 下一根 K 线开盘价买入。同一时间只持一个仓位。
出场：
  - 反弹离场：收盘价 > MA(ma_period，默认10) → 下一根开盘卖出(MA)。
  - 硬止损：价格 <= 买入价×(1 - stop_pct)(默认5%) → 止损价/开盘价成交(STOP)，
    防止下跌中继继续杀。
  - 时停：持有超过 time_stop 根(默认25)仍未到位 → 当日收盘平仓(TIME)。
  - 若到数据末尾仍未触发，按最后收盘价平仓(END)。

数据与成本与 strategy_ma_cross.py 一致(A股成本模型，默认剔除 ST)。

用法
----
    python strategy_bounce.py --data a_share_daily --workers 8
    python strategy_bounce.py --down-days 5 --down-thresh -0.10 --ma-period 10
    python strategy_bounce.py --codes-file codes_50_200.txt   # 只跑市值名单
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

# 需要逐笔打量比标签(供恐慌策略 --vol-min 过滤)时用, 其余行默认无需 volume 列
USECOLS_VOL = list(USECOLS) + ["volume"]


def backtest_stock(cfg: dict, code: str, df: pd.DataFrame) -> list:
    """对单只股票跑超跌反弹，返回交易记录列表。"""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    if n < max(cfg["down_days"], cfg["ma_period"]) + 10:
        return []

    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    st = df["isST"] if "isST" in df.columns else None
    volume = df["volume"].to_numpy(dtype=float) if "volume" in df.columns else None
    vol_n = cfg.get("vol_n", 0) if volume is not None else 0   # 量比窗口(0=不打标签)

    ma = close.rolling(cfg["ma_period"]).mean()
    ret_n = close / close.shift(cfg["down_days"]) - 1
    # 超跌 + 当日收阳企稳
    signal = ((ret_n <= cfg["down_thresh"]) & (close > open_)).fillna(False).to_numpy().copy()

    # 波动率过滤(可选)：max_atr_pct 跳过波动爆表的妖股；min_atr_drop 用 ATR 归一化超跌
    if cfg["max_atr_pct"] > 0 or cfg["min_atr_drop"] > 0:
        prev_c = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_c).abs(),
                        (low - prev_c).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / cfg["atr_period"], adjust=False).mean()
        if cfg["max_atr_pct"] > 0:
            mask1 = (atr / close <= cfg["max_atr_pct"]).fillna(False).to_numpy()
            signal = signal & mask1
        if cfg["min_atr_drop"] > 0:
            mask2 = ((close.shift(cfg["down_days"]) - close) / atr
                     >= cfg["min_atr_drop"]).fillna(False).to_numpy()
            signal = signal & mask2

    stop_pct = cfg["stop_pct"]
    time_stop = cfg["time_stop"]
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

        # 盘后量比 = 信号日成交量 / 前vol_n日均量(用 ≤ 信号日收盘的信息, 无未来函数)
        if vol_n > 0:
            pf = i - vol_n
            if pf >= 0:
                v0 = float(volume[pf:i].mean())
            elif i > 0:
                v0 = float(volume[:i].mean())
            else:
                v0 = float("nan")
            vol_ratio = float(volume[i]) / v0 if (v0 == v0 and v0 > 0) else float("nan")
        else:
            vol_ratio = float("nan")

        reason = "MA"
        exit_bar = exit_price = None
        for j in range(entry_bar + 1, n):
            o, c = float(open_[j]), float(close[j])
            # 1) 止损优先(防止飞刀续杀)：止损价/跳空开盘
            if o <= entry_price * (1 - stop_pct):
                reason, exit_price, exit_bar = "STOP", o, j
                break
            if low[j] <= entry_price * (1 - stop_pct):
                reason, exit_price, exit_bar = "STOP", entry_price * (1 - stop_pct), j
                break
            # 2) 反弹到 MA 上方 → 下一根开盘卖出
            if c > ma[j]:
                if j + 1 >= n:
                    exit_bar, exit_price = n - 1, float(close[n - 1])
                else:
                    exit_bar, exit_price = j + 1, float(open_[j + 1])
                break
            # 3) 时停
            if time_stop > 0 and j - entry_bar >= time_stop:
                reason, exit_price, exit_bar = "TIME", c, j
                break
        else:                            # 到样本末尾仍未触发，按最后收盘平仓
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
            "vol_ratio": round(vol_ratio, 4) if vol_ratio == vol_ratio else None,
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
        print("\n未产生任何交易。可尝试放宽 --down-days/--down-thresh。")
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
    print(f"平均每笔成本    : {mean(gross) - mean(rets):.2%}")
    print(f"累计收益(简单和): {sum(rets):.2%}")
    print(f"盈亏比(ProfitFactor): {gross_win / gross_loss if gross_loss > 0 else float('inf'):.2f}")
    print(f"平均持仓K线数   : {mean(t['bars'] for t in trades):.1f}")
    print(f"顺序复利最大回撤: {mdd:.2%}")
    print()

    print("=" * 62)
    print("按出场方式")
    print("=" * 62)
    print(f"{'出场':<6}{'笔数':>6}{'占比':>8}{'平均收益':>10}{'胜率':>9}")
    for reason in ("MA", "STOP", "TIME", "END"):
        sub = [t for t in trades if t["reason"] == reason]
        if not sub:
            continue
        sub_ret = [t["ret"] for t in sub]
        sr_wins = sum(1 for r in sub_ret if r > 0)
        print(f"{reason:<6}{len(sub):>6}{len(sub) / len(trades):>8.1%}"
              f"{mean(sub_ret):>10.2%}{sr_wins / len(sub):>9.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="超跌反弹回测(均值回归：N日超跌+收阳→次日买入→MA反弹离场/止损/时停)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    parser.add_argument("--data", default="a_share_daily")
    parser.add_argument("--down-days", type=int, default=10, help="超跌回看天数(默认10)")
    parser.add_argument("--down-thresh", type=float, default=-0.15,
                        help="N日累计跌幅阈值(默认-15%)")
    parser.add_argument("--ma-period", type=int, default=10,
                        help="反弹离场：收盘>MA(默认10)")
    parser.add_argument("--stop-pct", type=float, default=0.05,
                        help="硬止损比例(默认5%，防飞刀)")
    parser.add_argument("--time-stop", type=int, default=25,
                        help="时停K线数(默认25根，0=关闭)")
    parser.add_argument("--atr-period", type=int, default=14,
                        help="ATR周期(用于波动率过滤，默认14)")
    parser.add_argument("--min-atr-drop", type=float, default=0.0,
                        help="波动率归一化超跌：跌幅=close[N-1]/N日CLOSE差÷ATR，要求 >= 该倍(0=关闭)")
    parser.add_argument("--max-atr-pct", type=float, default=0.0,
                        help="波动率上限：ATR/收盘价 > 该比例则跳过信号(0=关闭)")
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
        "down_days": args.down_days, "down_thresh": args.down_thresh,
        "ma_period": args.ma_period, "stop_pct": args.stop_pct,
        "time_stop": args.time_stop, "include_st": args.include_st,
        "atr_period": args.atr_period,
        "min_atr_drop": args.min_atr_drop, "max_atr_pct": args.max_atr_pct,
        "usecols": USECOLS,
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

    ts = "关" if args.time_stop == 0 else f"{args.time_stop}根"
    vol_f = []
    if args.min_atr_drop > 0:
        vol_f.append(f"跌幅>={args.min_atr_drop:g}×ATR")
    if args.max_atr_pct > 0:
        vol_f.append(f"ATR%<={args.max_atr_pct:.0%}")
    vol_s = ("；波动过滤: " + "+".join(vol_f)) if vol_f else ""
    print(f"参数: 近{args.down_days}日跌>={-args.down_thresh:.0%}且收阳→次日买入 | "
          f"MA{args.ma_period}反弹离场 止损{-args.stop_pct:.0%} 时停{ts}{vol_s} | "
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
    out = "trades_bounce.csv"
    if all_trades:
        pd.DataFrame(all_trades).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n交易明细已保存: {out}")


if __name__ == "__main__":
    main()
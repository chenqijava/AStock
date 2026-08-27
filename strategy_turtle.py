# -*- coding: utf-8 -*-
"""
strategy_turtle.py — 经典海龟交易法则(Turtle Trading)回测
============================================================

规则(依据 Curtis Faith《海龟交易法则》，System 1 为主、System 2 为备选)
----------------------------------------------------------------------
入场：
  1. System 1: 收盘价向上突破前 20 日最高价(不含当日)开多，下一根开盘买入；
  2. System 2: 突破前 55 日最高价(改 --entry-lookback 55 --exit-lookback 20)。
加仓(金字塔)：
  - 每朝有利方向移动 0.5N 加 1 个单位，最多 4 个单位(--max-units)；
    加仓价 = max(当日开盘, 上一加仓价 + 0.5N)，每根 K 线最多加 1 个。
出场：
  - 止损：全部单位共用“主止损” = 最近一个单位入场价 - 2N(--n-risk)；
    每加 1 仓主止损上移 0.5N。盘中触及按止损价成交，跳空低开按开盘价成交。
  - 离场：收盘价跌破前 10 日最低价(--exit-lookback)，下一根开盘卖出。
  - 时停(原版规则含 2N×time_stop 日，默认关闭)：入场后第 time_stop 根
    (System1=10 / System2=20)若仍未相对首仓盈利 2N，按当日收盘价平仓。
    仅在第 time_stop 根判定一次(与原始规则一致)；--time-stop 0 关闭。
  - 若到数据末尾仍未平仓，按最后收盘价平仓(END)。
N = ATR(默认 20 期，Wilder 平滑)。

数据与成本
----------
数据默认读取 a_share_daily_hfq/*.csv(后复权，价格连续，无需再手工处理分红除权)；
默认剔除 ST(--include-st 放开)。成本模型与 strategy_ma_cross.py 一致
(佣金/印花税/滑点/最低佣金)，每笔净收益已扣成本。

与经典的简化/差异
-----------------
* 单只股票独立回测、同一时间只持一个仓位(经典海龟在多个市场并行建仓)；
* 仓位以固定资金 capital 计(经典按“账户 1% 风险 / N”计算单位大小，未建模)；
* 未处理 A 股涨停无法买入、跌停无法卖出的执行细节(近似处理)；
* 出场按下一根开盘成交，未计滑点以外的涨跌停成交障碍。

用法
----
    python strategy_turtle.py --data a_share_daily_hfq --workers 8
    python strategy_turtle.py --entry-lookback 55 --exit-lookback 20   # System 2
    python strategy_turtle.py --limit 300          # 调试：只看前300只
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
    """对单只股票跑海龟策略，返回所有单位腿(unit leg)的交易记录。"""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    if n < cfg["entry_lookback"] + cfg["atr_period"] + 5:
        return []

    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    st = df["isST"] if "isST" in df.columns else None

    # N = ATR：Wilder 平滑(ewm alpha=1/period)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / cfg["atr_period"], adjust=False).mean()

    # 突破信号：收盘价 > 前 entry_lookback 日最高价(不含当日)
    prior_high = high.rolling(cfg["entry_lookback"]).max().shift(1)
    signal = (close > prior_high).fillna(False).to_numpy()
    # 离场信号：收盘价 < 前 exit_lookback 日最低价(不含当日)
    prior_low = low.rolling(cfg["exit_lookback"]).min().shift(1)

    n_risk = cfg["n_risk"]            # 止损距离(单位: N)
    step = cfg["add_step"]            # 加仓间隔(单位: N)
    max_units = cfg["max_units"]

    trades = []
    i = 0
    while i < n - 1:
        if not signal[i] or (st is not None and st[i] == 1 and not cfg["include_st"]):
            i += 1
            continue
        entry_bar = i + 1             # 下一根开盘买入
        if entry_bar >= n:
            break
        ep = float(open_[entry_bar])
        N = float(atr[entry_bar])
        if not np.isfinite(N) or N <= 0:
            i += 1
            continue

        units = [ep]                  # 各单位入场价(金字塔)
        master_stop = ep - n_risk * N
        next_add = ep + step * N
        j = entry_bar + 1
        while j < n:
            o, h, l, c = (float(open_[j]), float(high[j]),
                          float(low[j]), float(close[j]))
            # 1) 止损优先：主止损(随加仓上移)，盘中触及按止损价，跳空低开按开盘价
            if o <= master_stop:
                reason, exit_price, exit_bar = "STOP", o, j
                break
            if l <= master_stop:
                reason, exit_price, exit_bar = "STOP", master_stop, j
                break
            # 2) 跌破 N 日低点离场 → 下一根开盘卖出(若已是最后一根则按当日收盘)
            if c < prior_low[j]:
                if j + 1 >= n:
                    reason, exit_price, exit_bar = "EXIT", c, j
                else:
                    reason, exit_price, exit_bar = "EXIT", float(open_[j + 1]), j + 1
                break
            # 3) 加仓：触及下一加仓位，最多 max_units 个，每根最多加 1 个
            if len(units) < max_units and h >= next_add:
                ap = max(o, next_add)
                units.append(ap)
                master_stop = ap - n_risk * N
                next_add = ap + step * N
            # 4) 时停：仅在第 time_stop 根判定一次(原版 2N×N 日规则)，未盈利 2N 则平仓
            if cfg["time_stop"] > 0 and j - entry_bar == cfg["time_stop"] \
                    and c < units[0] + 2 * N:
                reason, exit_price, exit_bar = "TIME", c, j
                break
            j += 1
        else:
            reason, exit_price, exit_bar = "END", float(close[n - 1]), n - 1

        for unit_no, up in enumerate(units, 1):
            trades.append({
                "code": code,
                "entry_date": df["date"][entry_bar],
                "entry_price": round(up, 4),
                "exit_date": df["date"][exit_bar],
                "exit_price": round(exit_price, 4),
                "reason": reason,
                "unit": unit_no,
                "n_units": len(units),
                "bars": exit_bar - entry_bar + 1,
                "gross_ret": round(exit_price / up - 1, 6),
                "ret": round(net_return(up, exit_price, cfg), 6),
                "atr": round(N, 4),
            })
        i = exit_bar + 1              # 跳过本仓位，继续找下一个突破
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
        print("\n未产生任何交易。可尝试 --include-st 或检查数据目录。")
        return

    rets = [t["ret"] for t in trades]
    gross = [t["gross_ret"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    gross_win, gross_loss = sum(wins), -sum(losses)

    n_pos = len(set((t["code"], t["entry_date"]) for t in trades))  # 仓位数
    avg_units = len(trades) / n_pos

    print("=" * 62)
    print("总体结果(已扣交易成本，按单位腿统计)")
    print("=" * 62)
    print(f"仓位数          : {n_pos}   (单位腿共 {len(trades)} 条，平均每仓 {avg_units:.2f} 个单位)")
    print(f"胜率            : {len(wins) / len(trades):.1%}  (盈利腿 {len(wins)} / 亏损腿 {len(losses)})")
    print(f"平均单腿净收益  : {mean(rets):.2%}   平均盈利腿 {mean(wins):.2%} / 平均亏损腿 {mean(losses):.2%}")
    print(f"平均每腿成本    : {mean(gross) - mean(rets):.2%}   (佣金+印花税+滑点，含最低佣金)")
    print(f"累计收益(简单和): {sum(rets):.2%}   (各腿净收益直接相加，近似多仓位组合)")
    print(f"盈亏比(ProfitFactor): {gross_win / gross_loss if gross_loss > 0 else float('inf'):.2f}")
    print(f"平均持仓K线数   : {mean(t['bars'] for t in trades):.1f}")
    print(f"最大单仓单位数  : {max(t['n_units'] for t in trades)}")
    print()

    print("=" * 62)
    print("按出场方式")
    print("=" * 62)
    print(f"{'出场':<6}{'腿数':>7}{'占比':>8}{'平均收益':>10}{'胜率':>9}")
    for reason in ("STOP", "EXIT", "TIME", "END"):
        sub = [t for t in trades if t["reason"] == reason]
        if not sub:
            continue
        sub_ret = [t["ret"] for t in sub]
        sr_wins = sum(1 for r in sub_ret if r > 0)
        print(f"{reason:<6}{len(sub):>7}{len(sub) / len(trades):>8.1%}"
              f"{mean(sub_ret):>10.2%}{sr_wins / len(sub):>9.1%}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="经典海龟交易法则回测(System 1 默认：20日突破/10日离场)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    parser.add_argument("--data", default="a_share_daily_hfq",
                        help="数据目录(默认 a_share_daily_hfq，后复权)")
    parser.add_argument("--entry-lookback", type=int, default=20, help="突破回看日数(默认20=System1)")
    parser.add_argument("--exit-lookback", type=int, default=10, help="离场回看日数(默认10=System1)")
    parser.add_argument("--atr-period", type=int, default=20, help="N=ATR 周期(默认20)")
    parser.add_argument("--n-risk", type=float, default=2.0, help="止损距离 N 的倍数(默认2)")
    parser.add_argument("--add-step", type=float, default=0.5, help="加仓间隔 N 的倍数(默认0.5)")
    parser.add_argument("--max-units", type=int, default=4, help="最大单位数(默认4)")
    parser.add_argument("--time-stop", type=int, default=0,
                        help="2N×N日时停：第N根仍未盈利2N则平仓(0=关闭；原版S1=10/S2=20)")
    parser.add_argument("--include-st", action="store_true", help="包含ST股票(默认剔除)")
    parser.add_argument("--capital", type=float, default=10000, help="每单位投入金额(元，默认10000)")
    parser.add_argument("--commission", type=float, default=0.00025, help="佣金率/边(默认万2.5)")
    parser.add_argument("--min-commission", type=float, default=5.0, help="单笔最低佣金(元，默认5)")
    parser.add_argument("--stamp", type=float, default=0.0005, help="印花税率(仅卖出，默认千0.5)")
    parser.add_argument("--slip", type=float, default=0.001, help="滑点率/边(默认0.1%)")
    parser.add_argument("--limit", type=int, default=0, help="只回测前 N 只(0=全部)")
    parser.add_argument("--codes-file", default=None,
                        help="只回测该文件内列出的股票代码(每行一个，如 sh.600000)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="并发进程数(默认 min(8,CPU核数))")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

    cfg = {
        "entry_lookback": args.entry_lookback, "exit_lookback": args.exit_lookback,
        "atr_period": args.atr_period, "n_risk": args.n_risk,
        "add_step": args.add_step, "max_units": args.max_units,
        "time_stop": args.time_stop, "include_st": args.include_st,
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

    ts = "关" if args.time_stop == 0 else f"{args.time_stop}日"
    print(f"参数: 突破{args.entry_lookback}日/离场{args.exit_lookback}日 | N=ATR{args.atr_period} | "
          f"止损{args.n_risk}N 加仓{args.add_step}N×{args.max_units}仓 时停{ts} | "
          f"{'含ST' if args.include_st else '剔除ST'}")
    print(f"成本: 佣金{args.commission:.4%}/边(最低{args.min_commission:.0f}元) 印花税{args.stamp:.4%}(卖) "
          f"滑点{args.slip:.2%}/边 每单位投入{args.capital:.0f}元")
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

    out = "trades_turtle.csv"
    if all_trades:
        pd.DataFrame(all_trades).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n交易明细已保存: {out}")


if __name__ == "__main__":
    main()

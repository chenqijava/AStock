# -*- coding: utf-8 -*-
"""
strategy_dividend_band.py — 高股息前10 · 高抛低吸
================================================

玩法：以全市场(默认主板)中股息率最高的 top_n(10) 只股票为交易池，在池内围绕
均线做高抛低吸(跌深了买、涨高了卖)，赚震荡的钱；持有期有硬止损与时停，不恋战。

一、高股息选取(时间变动、无未来函数)
------------------------------------
  每个 --rebalance-days(20) 个交易日重选一次(再平衡日 t)：
    股息率 = 近 --lookback-days(365) 天内除权除息日的每股税前现金股利之和 / t 日收盘价
  注意：**股息率必须用不复权价**(a_share_daily)——hfq 会把历史分红复进价格，
  用 hfq 价算收益率会系统性低估；交易/回测则用 hfq(避免除权跳空被当成真跌)。
  过滤：剔除 ST、价格 < --min-price(2元)、可选 peTTM<=0(--no-require-pe-pos 关)，
  每股股利 <=0(从未分红)的股票无收益率、天然选不上。
  按收益率降序取前 top_n 只 → 该期间(到下次再平衡)的交易池。

二、高抛低吸(均线偏离带)
------------------------
  对池内每只股票(t 日收盘判定)：
    偏离度 dev = close / MA(ma_period=20) - 1
    买：dev <= ---buy-pct(4%)  → 次日开盘买入(跌深了接)
    卖：dev >= +--sell-pct(4%) → 次日开盘卖出(涨高了抛)
  风控：硬止损 --stop-pct(8%)；时停 --time-stop(30) 日收盘平仓；
        再平衡日跌出前10(未再选中) → 当日收盘强制离场。
  同一只股票同一时间只持一个仓位；信号日与买入日都须在选期内(交界日不进场)。

三、基准对照(判断高抛低吸是否值得)
----------------------------------
  同时计算"买入持有高股息篮子"：每期等权买入当时 top_n，持到下期再平衡开盘卖出，
  hfq 全收益复利 → 若高抛低吸净值跑不赢它，说明震荡钱不值得赚。

数据依赖
--------
  a_share_daily(不复权, 算股息率) + a_share_daily_hfq(交易/回测) +
  dividend.csv(由 download_dividends.py 生成: code,ex_date,cash_ps)。

用法
----
    python download_dividends.py                          # 先拉股息(一次)
    python strategy_dividend_band.py                      # 主板 top10 高抛低吸
    python strategy_dividend_band.py --top-n 20 --buy-pct 0.06 --sell-pct 0.06
    python strategy_dividend_band.py --codes-file codes_main_board.txt --universe all
"""

import argparse
import glob
import logging
import multiprocessing
import os
import re
import sys
import time

import numpy as np
import pandas as pd

from strategy_ma_cross import net_return   # noqa: E402 复用 A股成本模型
from sim_portfolio import simulate           # noqa: E402 复用组合模拟

USECOLS_RAW = ["date", "close", "isST", "peTTM"]
USECOLS_HFQ = ["date", "open", "high", "low", "close", "isST"]


# ---------------------------------------------------------------------------
# 股息数据
# ---------------------------------------------------------------------------
def load_dividends(path: str) -> dict:
    """读 dividend.csv → {code: (ex_dates升序np数组, cash_ps对应np数组)}。"""
    df = pd.read_csv(path)
    out = {}
    for code, g in df.groupby("code"):
        g = g.sort_values("ex_date")
        out[code] = (g["ex_date"].to_numpy(),
                     g["cash_ps"].astype(float).to_numpy())
    return out


def trailing_div(divs, t_list, lookback: int) -> np.ndarray:
    """对每个再平衡日 t 求近 lookback 天内(不含t本身以前更早的)每股股利之和。

    divs = (ex_dates升序, cash); t_list = 再平衡日(ISO字符串)。
    返回与 t_list 等长的数组；区间为 (t-365d, t]。
    """
    if divs is None or not len(divs[0]):
        return np.zeros(len(t_list))
    ex, cash = divs
    cum = np.concatenate([[0.0], np.cumsum(cash)])
    lo = np.searchsorted(ex, t_list, side="right")     # 全部 <= t 的索引上限
    # t 的 365 天前
    t_prev = [(pd.Timestamp(t) - pd.Timedelta(days=lookback)).strftime("%Y-%m-%d")
              for t in t_list]
    hi = np.searchsorted(ex, t_prev, side="right")     # 全部 <= t-365 的索引上限
    return cum[lo] - cum[hi]


# ---------------------------------------------------------------------------
# 选股(Pass 1)：逐股算各再平衡日股息率 → 主进程按日排序取 top_n
# ---------------------------------------------------------------------------
def selection_worker(args):
    path, cfg, reb_dates = args
    code = os.path.basename(path).rsplit(".", 1)[0]
    try:
        df = pd.read_csv(path, usecols=cfg["usecols_raw"])
    except Exception as exc:                   # noqa: BLE001
        return code, np.full(len(reb_dates), np.nan)
    df = df.set_index("date")
    close = df["close"].astype(float)
    divs = cfg["dividends"].get(code)
    out = np.full(len(reb_dates), np.nan)
    # 只在有收盘价的再平衡日计算
    present = close.index.intersection(reb_dates)
    if not len(present):
        return code, out
    k_map = {d: k for k, d in enumerate(reb_dates)}
    t_idx = [k_map[t] for t in present]
    tr = trailing_div(divs, present, cfg["lookback_days"])
    for t, k, divsum in zip(present, t_idx, tr):
        c = float(close[t])
        if not (c > 0) or c < cfg["min_price"]:
            continue
        if int(df.loc[t, "isST"]) == 1:
            continue
        if cfg["require_pe_pos"] and float(df.loc[t, "peTTM"]) <= 0:
            continue
        if divsum <= 0:
            continue
        out[k] = divsum / c
    return code, out


# ---------------------------------------------------------------------------
# 交易(Pass 2)：选期内围绕均线高抛低吸，逐股独立回测
# ---------------------------------------------------------------------------
def trade_stock(args):
    path, cfg, intervals, rebs = args
    code = os.path.basename(path).rsplit(".", 1)[0]
    try:
        df = pd.read_csv(path, usecols=cfg["usecols_hfq"])
    except Exception as exc:                   # noqa: BLE001
        logging.warning("%s 读取失败: %s", code, exc)
        return code, [], {}
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    if n < cfg["ma_period"] + 10:
        return code, [], {}

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    low = df["low"].astype(float)
    ma = close.rolling(cfg["ma_period"]).mean()
    dev = close / ma - 1
    cal = cfg["cal"]                       # 主进程给的统一交易日历(ISO日期)
    R = cfg["R"]

    # 选期: 日期区间 [start, end)，end 为空串表示直到数据末尾。
    # 用各自 K 线的日期逐日对齐(股票停牌缺行不影响位移)。
    dates_np = df["date"].to_numpy()
    active = np.zeros(n, bool)
    for start, end in intervals:
        m = dates_np >= start
        if end:
            m &= dates_np < end
        active |= m

    # 买入持有基准：每个选中再平衡期(日历窗)：open[cal[R[k]]] → close[cal[R[k+1]-1]]
    # (hfq 全收益)。行号用 searchsorted 对齐该股自己的日期，缺行天然跳过。
    cal_np = np.asarray(cal)
    bench = {}
    for k in rebs:
        i0 = int(np.searchsorted(dates_np, cal_np[int(R[k])], side="left"))
        d1 = cal_np[int(R[k + 1]) - 1] if k + 1 < len(R) else cal_np[-1]
        i1 = int(np.searchsorted(dates_np, d1, side="right")) - 1
        if i1 >= i0 and float(open_.iloc[i0]) > 0:
            bench[k] = float(close.iloc[i1] / open_.iloc[i0])

    trades = []
    stop_pct, sell_pct, buy_pct = cfg["stop_pct"], cfg["sell_pct"], cfg["buy_pct"]
    time_stop = cfg["time_stop"]
    last_exit = -1
    dev_np = dev.to_numpy()
    open_np = open_.to_numpy()
    close_np = close.to_numpy()
    low_np = low.to_numpy()
    dates = df["date"].tolist()
    active_np = active

    idx = np.nonzero((dev_np <= -buy_pct) & active_np)[0]
    for i in idx:
        if i <= last_exit:
            continue
        entry_bar = i + 1
        if entry_bar >= n:
            break
        if not active_np[entry_bar]:
            continue                    # 交界日(跌出选期)不进场
        entry_price = float(open_np[entry_bar])
        stop_price = entry_price * (1 - stop_pct)

        reason, exit_bar, exit_price = "TIME", None, None
        for j in range(entry_bar, n):
            o, l, c = float(open_np[j]), float(low_np[j]), float(close_np[j])
            # 1) 硬止损优先
            if o <= stop_price:
                reason, exit_bar, exit_price = "SL", j, o
                break
            if l <= stop_price:
                reason, exit_bar, exit_price = "SL", j, stop_price
                break
            # 2) 高抛：收盘偏离>=sell_pct → 次日开盘卖(选期内)否则当日收盘
            if j > entry_bar and dev_np[j] >= sell_pct:
                if j + 1 < n and active_np[j + 1]:
                    reason, exit_bar, exit_price = "HIGH", j + 1, float(open_np[j + 1])
                else:
                    reason, exit_bar, exit_price = "HIGH", j, c
                break
            # 3) 时停
            if j - entry_bar >= time_stop:
                reason, exit_bar, exit_price = "TIME", j, c
                break
            # 4) 再平衡未再选中 → 选期末收盘强平
            if active_np[j] and (j + 1 >= n or not active_np[j + 1]):
                reason, exit_bar, exit_price = "DESEL", j, c
                break
        else:
            reason, exit_bar, exit_price = "END", n - 1, float(close_np[n - 1])

        trades.append({
            "code": code,
            "entry_date": dates[entry_bar],
            "entry_price": round(entry_price, 4),
            "exit_date": dates[exit_bar],
            "exit_price": round(exit_price, 4),
            "reason": reason,
            "bars": exit_bar - entry_bar + 1,
            "gross_ret": round(exit_price / entry_price - 1, 6),
            "ret": round(net_return(entry_price, exit_price, cfg), 6),
        })
        last_exit = exit_bar
    return code, trades, bench


# ---------------------------------------------------------------------------
# 文件列表与汇总
# ---------------------------------------------------------------------------
def build_files(data_dir: str, universe: str, codes_file: str) -> list:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    files = [f for f in files if os.path.basename(f) != "stock_list.csv"]
    if codes_file:
        keep = {ln.strip() for ln in open(codes_file, encoding="utf-8") if ln.strip()}
        files = [f for f in files if os.path.basename(f).rsplit(".", 1)[0] in keep]
    elif universe == "main":
        files = [f for f in files if re.match(r"^(sh\.60|sz\.00)", os.path.basename(f))]
    return files


def summarize(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("\n无交易。可放宽 --buy-pct 或扩大选股池。")
        return
    rets = trades["ret"].to_numpy()
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    pf = (wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    print("=" * 62)
    print(f"个股层(已扣成本): {len(trades)}笔  胜率{len(wins)/len(trades):.1%}  "
          f"平均净{rets.mean():+.2%}  PF{pf:.2f}  平均持{np.mean(trades['bars']):.0f}根")
    print("=" * 62)
    print("\n按出场方式:")
    print(f"{'出场':<7}{'笔数':>6}{'占比':>8}{'平均收益':>10}{'胜率':>9}")
    for reason in ("HIGH", "SL", "TIME", "DESEL", "END"):
        sub = trades[trades["reason"] == reason]
        if sub.empty:
            continue
        sr = sub["ret"].to_numpy()
        print(f"{reason:<7}{len(sub):>6}{len(sub)/len(trades):>8.1%}"
              f"{sr.mean():>10.2%}{(sr > 0).mean():>9.1%}")
    t = trades.copy()
    t["y"] = t["entry_date"].astype(str).str[:4]
    g = t.groupby("y").agg(笔数=("ret", "size"), 平均净=("ret", "mean"),
                           胜率=("ret", lambda s: (s > 0).mean() * 100))
    print("\n逐年(交易层):")
    for y, row in g.iterrows():
        print(f"  {y}: {int(row['笔数']):5d}笔  平均净{row['平均净']:+.2%}  胜率{row['胜率']:.0f}%")


def benchmark_nav(sel: dict, bench_map: dict, R: list) -> np.ndarray:
    """买入持有 top_n 篮子：每期等权，open[期初] → close[期末]，hfq 全收益复利。

    bench_map: {code: {再平衡期k: 区间收益比}}，主进程按期聚合。
    """
    nav = [1.0]
    for k in range(len(R)):
        ratios = [bench_map[c][k] for c in sel[k] if c in bench_map and k in bench_map[c]]
        if not ratios:
            nav.append(nav[-1])
        else:
            nav.append(nav[-1] * float(np.mean(ratios)))
    return np.array(nav)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="高股息前10 · 高抛低吸(再平衡选股 + 均线偏离带 + 买入持有基准)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    ap.add_argument("--data", default="a_share_daily_hfq", help="交易/回测数据(hfq)")
    ap.add_argument("--data-raw", default="a_share_daily", help="不复权数据(算股息率)")
    ap.add_argument("--dividend", default="dividend.csv", help="分红数据(下载脚本生成)")
    ap.add_argument("--universe", default="main", choices=["main", "all"])
    ap.add_argument("--codes-file", default=None)
    # 选股
    ap.add_argument("--rebalance-days", type=int, default=20, help="再平衡周期(默认20交易日)")
    ap.add_argument("--lookback-days", type=int, default=365, help="股息率回溯窗口(默认365天)")
    ap.add_argument("--top-n", type=int, default=10, help="高股息前N只(默认10)")
    ap.add_argument("--min-yield", type=float, default=0.0, help="最低股息率门槛(默认0)")
    ap.add_argument("--min-price", type=float, default=2.0, help="不选低价股(默认2元)")
    ap.add_argument("--require-pe-pos", dest="require_pe_pos", action="store_true",
                    default=True, help="要求 peTTM>0(默认开)")
    ap.add_argument("--no-require-pe-pos", dest="require_pe_pos", action="store_false")
    # 高抛低吸
    ap.add_argument("--ma-period", type=int, default=20, help="偏离基准均线(默认20)")
    ap.add_argument("--buy-pct", type=float, default=0.04, help="买: 收盘偏离均线<=-该值(默认4%)")
    ap.add_argument("--sell-pct", type=float, default=0.04, help="卖: 收盘偏离均线>=+该值(默认4%)")
    ap.add_argument("--stop-pct", type=float, default=0.08, help="硬止损(默认8%)")
    ap.add_argument("--time-stop", type=int, default=30, help="时停交易日(默认30)")
    # 组合
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--notional", type=float, default=100_000,
                    help="单笔名义本金(默认10万=总资金10%)")
    ap.add_argument("--max-exposure", type=float, default=0.0,
                    help="总仓位上限(默认0=不设限, 由现金决定)")
    ap.add_argument("--commission", type=float, default=0.00025)
    ap.add_argument("--min-commission", type=float, default=5.0)
    ap.add_argument("--stamp", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.001)
    ap.add_argument("--limit", type=int, default=0, help="调试: 只回测前N只")
    ap.add_argument("--out-suffix", default="", help="输出文件名后缀")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

    cfg = {
        "usecols_raw": USECOLS_RAW, "usecols_hfq": USECOLS_HFQ,
        "dividends": load_dividends(args.dividend),
        "lookback_days": args.lookback_days, "min_price": args.min_price,
        "require_pe_pos": args.require_pe_pos, "min_yield": args.min_yield,
        "ma_period": args.ma_period, "buy_pct": args.buy_pct,
        "sell_pct": args.sell_pct, "stop_pct": args.stop_pct,
        "time_stop": args.time_stop,
        "capital": args.capital, "commission": args.commission,
        "min_commission": args.min_commission, "stamp": args.stamp,
        "slip": args.slip, "lot": 100,
    }
    del cfg["min_yield"]                       # 门槛在主进程排名时应用

    files = build_files(args.data, args.universe, args.codes_file)
    if args.limit and args.limit < len(files):
        files = files[:args.limit]
    if not files:
        sys.stderr.write("未找到数据文件\n")
        sys.exit(1)

    # ---- 交易日历(与 sim_portfolio 一致: sh.600000)与再平衡日 ----
    cal = sorted(pd.read_csv(os.path.join(args.data, "sh.600000.csv"),
                             usecols=["date"])["date"].tolist())
    start_k = max(args.ma_period, 60)
    R = list(range(start_k, len(cal), args.rebalance_days))
    reb_dates = [cal[i] for i in R]
    cfg["cal"] = cal                        # 传入交易进程：选期/基准都要按日期对齐
    cfg["R"] = R
    print(f"板块: {len(files)}只 | 再平衡: 每{args.rebalance_days}日({len(R)}期, "
          f"{reb_dates[0]}~{reb_dates[-1]}) | top{args.top_n} | "
          f"买<=-{args.buy_pct:.0%}/卖>=+{args.sell_pct:.0%} @MA{args.ma_period}")

    # ---- Pass 1: 选股 ----
    t0 = time.time()
    tasks = [(os.path.join(args.data_raw, os.path.basename(f)), cfg, reb_dates)
             for f in files]
    yields_by_k: dict = {k: [] for k in range(len(R))}
    n_yield = 0
    if args.workers > 1:
        with multiprocessing.Pool(processes=args.workers) as pool:
            for code, ys in pool.imap_unordered(selection_worker, tasks, chunksize=16):
                for k, y in enumerate(ys):
                    if np.isfinite(y):
                        yields_by_k[k].append((float(y), code))
                        n_yield += 1
    else:
        for task in tasks:
            code, ys = selection_worker(task)
            for k, y in enumerate(ys):
                if np.isfinite(y):
                    yields_by_k[k].append((float(y), code))
                    n_yield += 1
    print(f"选股: {n_yield} 个(期×股)有效股息率, 用时{time.time()-t0:.0f}s")

    sel: dict = {}
    med_yield = []
    for k in range(len(R)):
        lst = sorted(yields_by_k[k], reverse=True)
        if args.min_yield > 0:
            lst = [(y, c) for y, c in lst if y >= args.min_yield]
        picked = lst[:args.top_n]
        sel[k] = [c for _, c in picked]
        if picked:
            med_yield.append(float(np.median([y for y, _ in picked])))
    if med_yield:
        print(f"每期选中 {int(np.median([len(v) for v in sel.values()]))} 只(中位), "
              f"选中股息率中位数 {np.median(med_yield):.1%}")

    # 每只股票的被选中区间：按再平衡期合并连续期 → 日期区间 [start, end)。
    # end 用下一再平衡日(跨过期边界)；最后一段 end="" 表示直到数据末尾。
    code_intervals: dict = {}
    code_rebs: dict = {}
    for code in files:
        c = os.path.basename(code).rsplit(".", 1)[0]
        rebs = sorted(k for k in range(len(R)) if c in sel[k])
        code_rebs[c] = rebs
        intv = []
        for k in rebs:
            end_d = cal[R[k + 1]] if k + 1 < len(R) else ""
            if intv and intv[-1][1] == cal[R[k]]:
                intv[-1][1] = end_d
            else:
                intv.append([cal[R[k]], end_d])
        code_intervals[c] = intv

    # ---- Pass 2: 高抛低吸回测 ----
    t0 = time.time()
    tasks2 = [(f, cfg, code_intervals[os.path.basename(f).rsplit(".", 1)[0]],
               code_rebs[os.path.basename(f).rsplit(".", 1)[0]])
              for f in files]
    all_trades = []
    bench_map: dict = {}
    if args.workers > 1:
        with multiprocessing.Pool(processes=args.workers) as pool:
            for code, trades, bench in pool.imap_unordered(trade_stock, tasks2, chunksize=16):
                all_trades.extend(trades)
                if bench:
                    bench_map[code] = bench
    else:
        for task in tasks2:
            code, trades, bench = trade_stock(task)
            all_trades.extend(trades)
            if bench:
                bench_map[code] = bench
    print(f"高抛低吸回测: {len(all_trades)}笔, 用时{time.time()-t0:.0f}s")

    if not all_trades:
        sys.stderr.write("无交易。可放宽 --buy-pct 或换参数\n")
        sys.exit(1)
    df = pd.DataFrame(all_trades)
    summarize(df)
    out = f"trades_dividend_band{args.out_suffix}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n交易明细: {out}")

    # ---- 组合模拟 ----
    print()
    print("=" * 62)
    print(f"组合模拟(资金 {args.capital/1e4:.0f}万, 每笔 {args.notional/1e4:.0f}万, "
          f"top{args.top_n}只高股息)")
    print("=" * 62)
    res = simulate(df, args.data, args.capital, args.notional,
                   args.commission, args.min_commission, args.stamp, args.slip,
                   max_exposure=args.max_exposure or None)
    print(f"期末净值        : {res['nav'][-1]/1e4:.1f}万  (初始 {args.capital/1e4:.0f}万)")
    print(f"总收益率        : {res['total_ret']:+.1%}    年化(CAGR) {res['cagr']:+.1%}")
    print(f"期间            : {res['dates'][0]} ~ {res['dates'][-1]}")
    print(f"最大回撤(日频)  : {res['mdd']:.1%}")
    print(f"建仓 {res['funded']} 笔 / 跳过 {res['skipped']} 笔 (命中率 {res['fund_rate']:.1%})")
    print(f"并发持仓: 最大 {res['max_open']} | 平均 {res['avg_open']:.1f} | 期末 {res['final_open']}")
    print(f"日频夏普(年化)  : {res['sharpe']:.2f}")

    # ---- 买入持有基准对比 ----
    bn = benchmark_nav(sel, bench_map, R)
    yrs = (pd.Timestamp(reb_dates[-1]) - pd.Timestamp(reb_dates[0])).days / 365.25
    b_total = bn[-1] - 1
    b_cagr = (bn[-1]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    b_mdd = float(((np.maximum.accumulate(bn) - bn) / np.maximum.accumulate(bn)).max())
    print()
    print("=" * 62)
    print("基准: 买入持有高股息篮子(等权, hfq 全收益, 同期)")
    print("=" * 62)
    print(f"期末净值        : {bn[-1]:.2f}  (初始1)")
    print(f"总收益率        : {b_total:+.1%}    年化(CAGR) {b_cagr:+.1%}")
    print(f"最大回撤        : {b_mdd:.1%}")
    print(f"高抛低吸 vs 持有: {res['total_ret']:+.1%} vs {b_total:+.1%} → "
          f"{'高抛低吸胜' if res['total_ret'] > b_total else '持有篮子胜'}")

    # ---- 逐年 ----
    navdf = pd.DataFrame({"date": res["dates"], "nav": res["nav"]})
    navdf["y"] = navdf["date"].astype(str).str[:4]
    print("\n逐年资金归属(高抛低吸, 日频净值):")
    for y, g in navdf.groupby("y"):
        r = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1
        md = float(((g["nav"].cummax() - g["nav"]) / g["nav"].cummax()).max())
        print(f"  {y}: 收益{r:+.1%}  年内回撤{md:.1%}  期末{int(g['nav'].iloc[-1]/1e4)}万")
    nav_out = f"trades_dividend_band{args.out_suffix}_nav_{int(args.capital//10000)}w.csv"
    navdf.drop(columns="y").to_csv(nav_out, index=False, encoding="utf-8-sig")
    print(f"净值曲线已保存: {nav_out}")


if __name__ == "__main__":
    main()

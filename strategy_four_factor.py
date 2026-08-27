# -*- coding: utf-8 -*-
"""
strategy_four_factor.py — 四因子超跌反弹(缩量收敛博弈)
====================================================

把《超跌反弹》交易手册系统化为可回测模型。核心思想：找"超跌 + 抛压枯竭"的股票，
用短周期(+5%止盈 / -3%止损 / 5日时停)做快速博弈，不恋战。

选股(四因子，全部满足才入候选池，t 日收盘判定)
------------------------------------------------
  F1 短期跌幅    ret5 = close/close[5日前]-1 < -short_drop(-6%)
                 → 5日跌得越深，反弹弹性越大。
  F2 中期跌幅    ytd  = close/close[ytd_days日前]-1 < -ytd_drop(-30%)
                 → 年内/一年充分超跌。用 250 交易日滚动窗口近似"年内"
                   (日历年初存在跨年跳变且样本端点敏感，回测统一用滚动年)。
  F3 缩量系数    shrink = turn / MA5(turn) < shrink_thresh(0.75)
                 → 当日换手不足自身5日均的四分之三，割肉盘枯竭、抛压衰竭。
  F4 波动骤降    std5(turn)[t]/std5(turn)[t-5] < vol_ratio_thresh(0.85)
                 → 换手波动收敛 = 恐慌情绪释放完毕。

加分过滤(非强制，可 --no-require-pe-pos / --min-price 0 关闭)：
  基本面 peTTM > 0(盈利，数据无净利润字段，用 PE 正负近似)；
  不选低价股 close >= min_price(默认2元，避开仙股/退市风险)；
  市值偏小 mc 在 [min_mc, max_mc](默认 <=300亿)，mc=成交额/换手率×100 的近期中位数。

综合打分排序(买入优先级)
------------------------
  score = -(5日跌幅) × 0.6 + (1 - 缩量系数) × 0.4
  每交易日按 score 降序取前 top_n(默认8, 5~10)只。

买入时机(t+1 开盘, 用日线可模拟的部分)
--------------------------------------
  要求 t+1 开盘相对 t 收盘的跳空 gap ∈ [min_gap, max_gap]：
    默认 [-5%, +1%] —— 低开(-2%左右为理想)或平开可建仓，跳过高开(手册
    "平开/高开等二次探底"属分时层面，日线无法模拟二次探底 → 直接放弃)；
    也顺带落实"买入时涨幅不超过3%"与"不抢已反弹超+3%"。
  手册的"反弹启动首日量能>8日均量2倍"是分时确认条件，若用于入场需用到
  t+1 当日成交量(收盘才知道) → 会引入未来函数，故不入场条件(见文档注释)。

持仓与卖出
----------
  固定 5 个交易日卖出(--time-stop 5)；优先 -3% 止损(--stop-pct)；
  再判 +5% 止盈(--tp-pct，触价成交/跳空按开盘)；可选动态离场
  (--exit-mode ma：收盘跌破 MA3/MA5 离场，作为手册"跌破均线部分/全线清仓"
  的单仓位近似)；都不触发则第5日收盘平仓。

仓位与风控
----------
  单笔名义本金 notional(默认10万=总资金10%)，总仓位上限 max_exposure
  (默认0.5=5成，sim_portfolio 新参数)；同日信号按 score 优先建仓。
  三不抢由条件本身覆盖：超跌且缩量 → 排除已反弹/高位/天量股。

实测结论(2026-08-27, 主板3193只 hfq, 2016-2026)
------------------------------------------------
  - 按手册原样(每日无差别取前8只)：PF 0.82 平均净-0.33%，组合10年-93% —— 净亏，
    与套件其余策略同结论(超跌反弹毛边不足抵成本)。
  - 打分排序(跌得越深得分越高)是轻微负贡献：被选中均收益 < 被跳过(各阈值一致)。
  - **决定性发现：四因子信号的边际在"稀有日"——当日全市场四因子信号越少越值钱。**
    按密度分桶(0-10/10-25/25-50/50-100/100-150/150+ 信号/日)原始信号平均净：
    +0.34% / +0.07% / -0.26% / -0.33% / -0.35% / +0.01% —— 与 strategy_panic
    的密度方向**相反**(恐慌策略=普跌恐慌日才赚；四因子信号已含缩量+波动收敛，
    普跌日只是大盘回调无选择性，反而是个别股票深度去杠杆后的独特形态才有肉)。
  - 加 --max-density 稀有日过滤后整体转正且随阈值单调(非凑巧)：
    <15: PF1.27 单笔+0.41% CAGR+3.0% 回撤4.7% 夏普0.75(每笔最优)；
    <25: PF1.13 +0.20% CAGR+3.7% 回撤11.5% 夏普0.70(总收益更高、交易更多)。
    注意：这是弱边际(PF~1.1-1.3)，远不及 strategy_panic(主板+hfq+恐慌60: PF1.85、
    100万 CAGR~9.5% 回撤~8.5%)，且样本偏小、存在幸存者偏差，宜作补充而非主策略。

简化与取舍(务必知晓)
--------------------
  - "年内"用滚动250日近似；"净利润增长"数据缺失未实现；
  - 分时条件(平盘整理/二次探底/首日放量确认/分批止盈)日线无法无偏模拟；
  - 动态止盈 MA3 半仓离场简化为整仓离场(组合模拟按整仓计价)。

用法
----
    python strategy_four_factor.py                          # 主板 + 四因子默认参数(按手册)
    python strategy_four_factor.py --max-density 15         # ★ 稀有日过滤, 实测转正(PF1.27)
    python strategy_four_factor.py --top-n 5 --ytd-drop -0.50
    python strategy_four_factor.py --codes-file codes_50_200.txt
    python strategy_four_factor.py --universe all --max-mc 0 --min-mc 0 --min-price 0
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

USECOLS_FF = ["date", "open", "high", "low", "close",
              "amount", "turn", "peTTM", "isST", "tradestatus"]


def backtest_stock(cfg: dict, code: str, df: pd.DataFrame) -> list:
    """四因子超跌反弹单股回测。返回交易记录(含 score/shrink/ret5 供打分排序)。"""
    # 剔除停牌行(tradestatus=0)，避免把停牌日的换手=0 当成缩量信号
    df = df[df["tradestatus"] == 1].sort_values("date").reset_index(drop=True)
    n = len(df)
    if n < cfg["ytd_days"] + 20:
        return []

    s = df
    close = s["close"].astype(float)
    open_ = s["open"].astype(float)
    high = s["high"].astype(float)
    low = s["low"].astype(float)

    # --- 四因子 ---
    ret5 = close / close.shift(cfg["short_days"]) - 1                     # F1 5日跌幅
    ytd = close / close.shift(cfg["ytd_days"]) - 1                        # F2 年内/一年跌幅
    tma5 = s["turn"].rolling(5, min_periods=4).mean()                     # F3 5日均换手(含当日)
    shrink = s["turn"] / tma5                                             # 缩量系数
    vstd = s["turn"].rolling(5, min_periods=4).std()                      # F4 5日换手标准差
    vstd_prev = vstd.shift(5)                                             # 前5日标准差
    vratio = vstd / vstd_prev                                             # 波动收敛比

    sig = ((ret5 < -cfg["short_drop"]) & (ytd < -cfg["ytd_drop"])
           & (shrink < cfg["shrink_thresh"]) & (vratio < cfg["vol_ratio_thresh"]))
    if cfg["require_pe_pos"]:
        sig &= (s["peTTM"] > 0).fillna(False)
    if cfg["min_price"] > 0:
        sig &= (close >= cfg["min_price"])
    sig = sig.fillna(False)

    # --- 市值过滤(整股级别) ---
    if cfg["min_mc"] > 0 or cfg["max_mc"] > 0:
        valid = (s["turn"] > 0) & (s["amount"] > 0)
        if int(valid.sum()) < 5:
            return []
        mc = float(np.median((s.loc[valid, "amount"] * 100.0
                              / s.loc[valid, "turn"]).tail(20))) / 1e8
        if not (np.isfinite(mc) and mc >= cfg["min_mc"]
                and (cfg["max_mc"] <= 0 or mc <= cfg["max_mc"])):
            return []

    # --- 打分(买入优先级排序) ---
    score = (-ret5) * cfg["score_w1"] + (1 - shrink) * cfg["score_w2"]

    ma3 = ma5 = None
    if cfg["exit_mode"] == "ma":
        ma3 = close.rolling(3).mean()
        ma5 = close.rolling(5).mean()

    trades = []
    stop_pct, tp_pct = cfg["stop_pct"], cfg["tp_pct"]
    time_stop = cfg["time_stop"]
    last_exit = -1
    idx = np.nonzero(sig.to_numpy())[0]
    for i in idx:
        if i <= last_exit:
            continue                        # 持仓中(单仓位)，跳过重叠信号
        if s["isST"].iloc[i] == 1 and not cfg["include_st"]:
            continue
        entry_bar = i + 1                   # t+1 开盘买入
        if entry_bar >= n:
            break
        gap = open_.iloc[entry_bar] / close.iloc[i] - 1
        if not (cfg["min_gap"] <= gap <= cfg["max_gap"]):
            continue                        # 高开(或崩盘大幅低开) → 放弃本信号
        entry_price = float(open_.iloc[entry_bar])
        stop_price = entry_price * (1 - stop_pct)
        tp_price = entry_price * (1 + tp_pct)

        reason = "TIME"
        exit_bar = exit_price = None
        for j in range(entry_bar, n):
            o, h, l, c = (float(open_.iloc[j]), float(high.iloc[j]),
                          float(low.iloc[j]), float(close.iloc[j]))
            # 1) 止损优先(无条件砍仓，防飞刀续杀)
            if o <= stop_price:
                reason, exit_price, exit_bar = "SL", o, j
                break
            if l <= stop_price:
                reason, exit_price, exit_bar = "SL", stop_price, j
                break
            # 2) 止盈(+5%立即卖出；触价按限价单成交，跳空高开按开盘)
            if o >= tp_price:
                reason, exit_price, exit_bar = "TP", o, j
                break
            if h >= tp_price:
                reason, exit_price, exit_bar = "TP", tp_price, j
                break
            # 3) 动态止盈(备选)：收盘跌破均线离场
            if cfg["exit_mode"] == "ma":
                if c < ma3.iloc[j]:
                    reason, exit_price, exit_bar = "MA3", c, j
                    break
                if c < ma5.iloc[j]:
                    reason, exit_price, exit_bar = "MA5", c, j
                    break
            # 4) 固定周期卖出(默认5个交易日，不恋战)
            if j - entry_bar >= time_stop:
                reason, exit_price, exit_bar = "TIME", c, j
                break
        else:                               # 数据末尾仍未触发 → 按最后收盘平仓
            reason = "END"
            exit_bar, exit_price = n - 1, float(close.iloc[n - 1])

        trades.append({
            "code": code,
            "entry_date": s["date"].iloc[entry_bar],
            "entry_price": round(entry_price, 4),
            "exit_date": s["date"].iloc[exit_bar],
            "exit_price": round(exit_price, 4),
            "reason": reason,
            "bars": exit_bar - entry_bar + 1,
            "gross_ret": round(exit_price / entry_price - 1, 6),
            "ret": round(net_return(entry_price, exit_price, cfg), 6),
            "score": round(float(score.iloc[i]), 4),
            "ret5": round(float(ret5.iloc[i]), 4),
            "shrink": round(float(shrink.iloc[i]), 4),
            "gap": round(float(gap), 4),
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
    """交易层汇总：总体 + 按出场方式 + 逐年。"""
    if trades.empty:
        print("\n无交易。可放宽 --short-drop/--ytd-drop/--shrink-thresh 或换板块。")
        return
    rets = trades["ret"].to_numpy()
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    pf = (wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    print("=" * 62)
    print(f"个股层(已扣成本): {len(trades)}笔  胜率{len(wins)/len(trades):.1%}  "
          f"平均净{rets.mean():+.2%}  PF{pf:.2f}")
    print("=" * 62)

    print("\n按出场方式:")
    print(f"{'出场':<6}{'笔数':>6}{'占比':>8}{'平均收益':>10}{'胜率':>9}")
    for reason in ("SL", "TP", "MA3", "MA5", "TIME", "END"):
        sub = trades[trades["reason"] == reason]
        if sub.empty:
            continue
        sr = sub["ret"].to_numpy()
        print(f"{reason:<6}{len(sub):>6}{len(sub)/len(trades):>8.1%}"
              f"{sr.mean():>10.2%}{(sr > 0).mean():>9.1%}")

    t = trades.copy()
    t["y"] = t["entry_date"].astype(str).str[:4]
    g = t.groupby("y").agg(笔数=("ret", "size"), 平均净=("ret", "mean"),
                           胜率=("ret", lambda s: (s > 0).mean() * 100))
    print("\n逐年(交易层):")
    for y, row in g.iterrows():
        print(f"  {y}: {int(row['笔数']):5d}笔  平均净{row['平均净']:+.2%}  胜率{row['胜率']:.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="四因子超跌反弹——短周期博弈(回测+打分排序+组合模拟+逐年)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    ap.add_argument("--data", default="a_share_daily_hfq", help="价格数据目录(hfq)")
    ap.add_argument("--universe", default="main", choices=["main", "all"],
                    help="板块: main=沪深主板(默认) | all=全部")
    ap.add_argument("--codes-file", default=None, help="自定义名单(优先于 --universe)")
    # 四因子
    ap.add_argument("--short-days", type=int, default=5, help="短期跌幅窗口(默认5)")
    ap.add_argument("--short-drop", type=float, default=-0.06,
                    help="5日跌幅阈值(默认-6%；手册建议-5~-10)")
    ap.add_argument("--ytd-days", type=int, default=250, help="中期跌幅窗口(默认250≈年内)")
    ap.add_argument("--ytd-drop", type=float, default=-0.30,
                    help="年内跌幅阈值(默认-30%；-50%更佳)")
    ap.add_argument("--shrink-thresh", type=float, default=0.75,
                    help="缩量系数上限 当日换手/5日均换手(默认0.75)")
    ap.add_argument("--vol-thresh", type=float, default=0.85,
                    help="波动收敛比上限 5日std/前5日std(默认0.85)")
    # 加分过滤
    ap.add_argument("--require-pe-pos", dest="require_pe_pos", action="store_true",
                    default=True, help="要求 peTTM>0(盈利, 默认开)")
    ap.add_argument("--no-require-pe-pos", dest="require_pe_pos", action="store_false",
                    help="关闭盈利过滤")
    ap.add_argument("--min-price", type=float, default=2.0, help="不选低价股(默认2元, 0=关)")
    ap.add_argument("--min-mc", type=float, default=0.0, help="市值下界(亿元, 默认0)")
    ap.add_argument("--max-mc", type=float, default=300.0,
                    help="市值上界(亿元, 默认300=市值偏小；0=无上界)")
    # 打分与排序
    ap.add_argument("--score-w1", type=float, default=0.6,
                    help="打分权重：5日跌幅(默认0.6)")
    ap.add_argument("--score-w2", type=float, default=0.4,
                    help="打分权重：缩量(默认0.4)")
    ap.add_argument("--top-n", type=int, default=8,
                    help="每交易日按score取前N只(默认8, 手册5~10)")
    ap.add_argument("--panic-threshold", type=int, default=0,
                    help="恐慌日过滤: 仅取当日全市场四因子信号数>=该值的日子(默认0=关；"
                         "对四因子信号集实测密度与收益负相关, 该方向大概率无效)")
    ap.add_argument("--max-density", type=int, default=0,
                    help="稀有日过滤: 仅取当日四因子信号数<该值的日子(默认0=关；"
                         "实测低密度日收益更高, 建议试25/10)")
    ap.add_argument("--reverse-score", action="store_true",
                    help="取每交易日score最低N只(实测用：判断手册排序是否反向择优)")
    # 买入时机
    ap.add_argument("--min-gap", type=float, default=-0.05,
                    help="次日最低允许低开(默认-5%；崩盘式大幅低开不接)")
    ap.add_argument("--max-gap", type=float, default=0.01,
                    help="次日最高允许高开(默认+1%≈平开；高开放弃)")
    # 出场
    ap.add_argument("--exit-mode", default="time", choices=["time", "ma"],
                    help="time=5日时停(默认) | ma=跌破MA3/MA5动态离场")
    ap.add_argument("--stop-pct", type=float, default=0.03, help="止损(默认-3%无条件砍仓)")
    ap.add_argument("--tp-pct", type=float, default=0.05, help="止盈(默认+5%立即卖出)")
    ap.add_argument("--time-stop", type=int, default=5, help="固定持股周期(默认5个交易日)")
    ap.add_argument("--include-st", action="store_true", help="包含ST股票(默认剔除)")
    # 组合模拟
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--notional", type=float, default=100_000,
                    help="单笔名义本金(默认10万=总资金10%)")
    ap.add_argument("--max-exposure", type=float, default=0.5,
                    help="总仓位上限(默认0.5=5成；0=不设限)")
    # 成本
    ap.add_argument("--commission", type=float, default=0.00025)
    ap.add_argument("--min-commission", type=float, default=5.0)
    ap.add_argument("--stamp", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.001)
    ap.add_argument("--limit", type=int, default=0, help="调试: 只回测前N只")
    ap.add_argument("--out-suffix", default="", help="输出文件名后缀(如 reverse)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

    cfg = {
        "short_days": args.short_days, "short_drop": args.short_drop,
        "ytd_days": args.ytd_days, "ytd_drop": args.ytd_drop,
        "shrink_thresh": args.shrink_thresh, "vol_ratio_thresh": args.vol_thresh,
        "require_pe_pos": args.require_pe_pos, "min_price": args.min_price,
        "min_mc": args.min_mc, "max_mc": args.max_mc,
        "min_gap": args.min_gap, "max_gap": args.max_gap,
        "exit_mode": args.exit_mode, "stop_pct": args.stop_pct,
        "tp_pct": args.tp_pct, "time_stop": args.time_stop,
        "include_st": args.include_st,
        "score_w1": args.score_w1, "score_w2": args.score_w2, "top_n": args.top_n,
        "usecols": USECOLS_FF,
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

    board = {"main": "沪深主板", "all": "全部"}[args.universe]
    print(f"板块: {board}({len(files)}只) | 因子: 5日跌<{-args.short_drop:.0%} "
          f"年内跌<{-args.ytd_drop:.0%}({args.ytd_days}日) 缩量<{args.shrink_thresh:.2f} "
          f"波动收敛<{args.vol_thresh:.2f}")
    exp_txt = f" 总仓位{args.max_exposure:.0%}" if args.max_exposure else ""
    print(f"过滤: 市值[{args.min_mc:.0f},{args.max_mc:.0f}]亿 价格>={args.min_price}元 "
          f"{'盈利(peTTM>0)' if args.require_pe_pos else '无盈利过滤'} | "
          f"跳空[{args.min_gap:+.0%},{args.max_gap:+.0%}] | 每笔{args.notional/1e4:.0f}万{exp_txt}")
    print(f"出场: {args.exit_mode}模式 止损{-args.stop_pct:.0%} 止盈+{args.tp_pct:.0%} "
          f"时停{args.time_stop}日 | 打分: 跌{-args.score_w1:.0%}+缩量{-args.score_w2:.0%} "
          f"每日取{args.top_n}只")

    # 1) 逐股回测
    t0 = time.time()
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

    df = pd.DataFrame(all_trades)

    # 1.5) 恐慌日密度分布(诊断: 判断孤立超跌是否接飞刀)
    cnt = df["entry_date"].value_counts()
    df["density"] = df["entry_date"].map(cnt)
    hist = cnt.value_counts().sort_index()
    bands = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 150), (150, 10_000)]
    print("信号日密度分布(每日四因子信号数 → 天数):")
    for lo, hi in bands:
        nd = int(((cnt >= lo) & (cnt < hi)).sum())
        if nd:
            print(f"  {lo:>3}-{hi:>4}/日: {nd}天")
    # 原始信号按密度分桶的净收益(诊断：孤立超跌是否接飞刀)
    print("原始信号按当日密度分桶 → 平均净收益:")
    df["_band"] = pd.cut(df["density"], [b[0] for b in bands] + [1e9],
                         labels=[f"{b[0]}-{b[1]}" for b in bands], right=False)
    for bnd, g in df.groupby("_band", observed=True):
        rr = g["ret"].to_numpy()
        print(f"  {bnd:>12}: {len(g):5d}笔  平均净{rr.mean():+.3%}  胜率{(rr > 0).mean():.0%}")
    df = df.drop(columns="_band")
    # 恐慌日过滤(可关，默认关)
    if args.panic_threshold > 0:
        before = len(df)
        df = df[df["density"] >= args.panic_threshold]
        panic_days = int(df["entry_date"].nunique())
        print(f"恐慌日过滤(>= {args.panic_threshold} 信号/日): {before} -> {len(df)} 笔 | "
              f"恐慌日 {panic_days} 天")
        if df.empty:
            sys.stderr.write("恐慌过滤后无信号\n")
            sys.exit(1)
    # 稀有日过滤(反向：只取低密度日)
    if args.max_density > 0:
        before = len(df)
        df = df[df["density"] < args.max_density]
        days = int(df["entry_date"].nunique())
        print(f"稀有日过滤(< {args.max_density} 信号/日): {before} -> {len(df)} 笔 | "
              f"覆盖 {days} 天")
        if df.empty:
            sys.stderr.write("稀有日过滤后无信号\n")
            sys.exit(1)

    # 2) 打分排序：每交易日按 score 取前 top_n 只(手册：取前5-10只；--reverse-score 取最低)
    df = df.sort_values(["entry_date", "score"],
                        ascending=[True, args.reverse_score]).reset_index(drop=True)
    before = len(df)
    df = df.groupby("entry_date", as_index=False).head(cfg["top_n"])
    order_txt = "score最低" if args.reverse_score else "score最高"
    print(f"打分排序: 信号 {before} 笔 → 每日前{cfg['top_n']}只({order_txt}) {len(df)} 笔")

    # 3) 交易层汇总
    summarize(df)
    df["priority"] = df["score"]            # 组合建仓顺序：与打分排序一致
    out = f"trades_four_factor{args.out_suffix}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n交易明细: {out}")

    # 4) 组合净值模拟
    print()
    print("=" * 62)
    cap_txt = f", 总仓位上限 {args.max_exposure:.0%}" if args.max_exposure else ""
    print(f"组合模拟(资金 {args.capital/1e4:.0f}万, 每笔 {args.notional/1e4:.0f}万{cap_txt})")
    print("=" * 62)
    res = simulate(df, args.data, args.capital, args.notional,
                   args.commission, args.min_commission, args.stamp, args.slip,
                   priority_asc=args.reverse_score, max_exposure=args.max_exposure or None)
    print(f"期末净值        : {res['nav'][-1]/1e4:.1f}万  (初始 {args.capital/1e4:.0f}万)")
    print(f"总收益率        : {res['total_ret']:+.1%}    年化(CAGR) {res['cagr']:+.1%}")
    print(f"期间            : {res['dates'][0]} ~ {res['dates'][-1]}")
    print(f"最大回撤(日频)  : {res['mdd']:.1%}")
    print(f"建仓 {res['funded']} 笔 / 跳过 {res['skipped']} 笔 (资金不足/仓位上限, 命中率 {res['fund_rate']:.1%})")
    print(f"并发持仓: 最大 {res['max_open']} | 平均 {res['avg_open']:.1f} | 期末 {res['final_open']}")
    print(f"日频夏普(年化)  : {res['sharpe']:.2f}")
    if res["ret_taken"] or res["ret_skipped"]:
        mt = float(np.mean(res["ret_taken"])) if res["ret_taken"] else 0.0
        ms = float(np.mean(res["ret_skipped"])) if res["ret_skipped"] else 0.0
        print(f"[诊断] 被选中均收益 {mt:+.3%} vs 被跳过均收益 {ms:+.3%} "
              f"({len(res['ret_taken'])}/{len(res['ret_skipped'])}笔)")

    # 5) 逐年资金归属
    navdf = pd.DataFrame({"date": res["dates"], "nav": res["nav"]})
    navdf["y"] = navdf["date"].astype(str).str[:4]
    print("\n逐年资金归属(日频净值):")
    for y, g in navdf.groupby("y"):
        r = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1
        md = float(((g["nav"].cummax() - g["nav"]) / g["nav"].cummax()).max())
        print(f"  {y}: 收益{r:+.1%}  年内回撤{md:.1%}  期末{int(g['nav'].iloc[-1]/1e4)}万")

    nav_out = f"trades_four_factor{args.out_suffix}_nav_{int(args.capital//10000)}w.csv"
    navdf.drop(columns="y").to_csv(nav_out, index=False, encoding="utf-8-sig")
    print(f"净值曲线已保存: {nav_out}")


if __name__ == "__main__":
    main()

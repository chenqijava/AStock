# -*- coding: utf-8 -*-
"""
strategy_low_dip.py — 日线回测版短线低吸策略
==============================================

一、基础规则(本金5万, A股主板)
  1. 资金仓位: 总持仓上限4成(2万), 单只最多1万, 同时持仓≤2只
  2. T+1: 当日买入次日才能卖出; 仅尾盘模拟14:45 低吸建仓(收盘价)
  3. 盈亏标准(日线高低点判断):
     - 持仓最低价跌超 3% -> 止损全卖
     - 持仓最高价涨超 3% -> 止盈; 3%~8% 减半仓; ≥8% 全仓卖出
  4. 硬性风控:
     - 单日账户亏损超总资金1% -> 当日停止所有买入
     - 月度最大回撤超 8% -> 总仓位降至 2 成(当月不再恢复)

二、日线选股条件(仅60/000主板)
  1. 剔除 ST/停牌/涨跌停; 流通市值 50~300亿; 近5日日均成交额≥2亿
  2. 量化主线条件: 放量(成交量≥5日均量1.2倍) + 所属行业近3日涨幅市场前5

三、日线适配逻辑
  - 止盈止损都用日线 高/低 判断, 同一根日线内按保守路径排序: 先止损后止盈
  - 滑点: 买入价上浮0.4%, 卖出价下浮0.4%
  - 前复权日线 + 真实成本(佣金万2.5/最低5元, 印花税千0.5卖出)

输出: 期末总资产 / 总收益率 / 最大回撤 / 交易次数 / 胜率 / 盈亏比 / 年化

用法
----
    python strategy_low_dip.py                 # 默认: 5万, 主板
"""

import argparse
import glob
import os
import re
import sys
import time
from statistics import mean

import numpy as np
import pandas as pd

MAIN_RE = re.compile(r"^(sh\.60|sz\.00)")

COST_COMM = 0.00025     # 佣金 万2.5
COST_MIN_COMM = 5.0     # 最低5元
COST_STAMP = 0.0005     # 印花税 千0.5 (卖出)
SLIP = 0.004            # 滑点 0.4%/边

STOP_PCT = 0.03         # 止损 3%
TP3 = 0.03              # 止盈下限 3%
TP8 = 0.08              # 全仓止盈 8%

VOL_RATIO_MIN = 1.2     # 放量倍数
AMT5_MIN = 2e8          # 近5日日均成交额 >= 2亿
MC_LOW, MC_HIGH = 50.0, 300.0   # 流通市值区间(亿元)
SECTOR_TOP = 5          # 行业近3日涨幅前5
LIMIT_PCT = 9.7         # 涨跌停判定阈值(主板非ST ±10%)


def fetch_industry(cache_path: str = "industry_map.csv") -> dict:
    """从 baostock 拉取行业映射 code->industry, 带本地缓存。"""
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, dtype={"code": str})
        return dict(zip(df["code"], df["industry"]))
    try:
        import baostock as bs
    except ImportError:
        sys.stderr.write("缺少 baostock, 无法拉取行业; 安装 baostock 或用 --industry\n")
        sys.exit(1)
    bs.login()
    rs = bs.query_stock_industry()
    rows = []
    while rs.next():
        d = rs.get_row_data()
        rows.append((d[1], d[3]))  # code, industry
    bs.logout()
    m = {c: ind for c, ind in rows if c and ind}
    pd.DataFrame(sorted(m.items()), columns=["code", "industry"]).to_csv(
        cache_path, index=False, encoding="utf-8-sig")
    return m


def load_main_board(data_dir: str, limit: int = 0) -> pd.DataFrame:
    """载入主板(60/000)日线。只读模拟所需列。"""
    cols = ["date", "code", "high", "low", "close", "volume",
            "amount", "turn", "isST", "pctChg", "tradestatus"]
    files = [f for f in glob.glob(os.path.join(data_dir, "*.csv"))
             if MAIN_RE.match(os.path.basename(f))
             and os.path.basename(f) != "stock_list.csv"]
    if limit:
        files = files[:limit]
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, usecols=cols))
        except Exception:                                   # noqa: BLE001
            continue
    if not frames:
        sys.exit("无有效数据文件")
    out = pd.concat(frames, ignore_index=True)
    out = out[out["tradestatus"] == 1].copy()
    return out


def precompute(df: pd.DataFrame, industry: dict) -> pd.DataFrame:
    """按代码滚动计算过滤指标 + 行业3日涨幅前5标记。"""
    for c in ["volume", "amount", "turn", "pctChg", "isST"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    valid = df["turn"] > 0
    df["mc"] = np.where(valid, df["amount"] * 100.0 / df["turn"] / 1e8, np.nan)

    g = df.groupby("code", sort=False)
    df["amt_ma5"] = g["amount"].transform(
        lambda s: s.rolling(5, min_periods=5).mean())
    df["vol_ma5"] = g["volume"].transform(
        lambda s: s.rolling(5, min_periods=5).mean())
    df["vol_ratio"] = df["volume"] / df["vol_ma5"]

    df["industry"] = df["code"].map(industry)
    ind0 = df[df["industry"].notna()]
    ind_rt = (ind0.groupby(["industry", "date"])["pctChg"].mean()
              .reset_index().sort_values(["industry", "date"]))
    ind_rt["gain3"] = ind_rt.groupby("industry")["pctChg"].transform(
        lambda s: s.rolling(3, min_periods=3).sum())
    ind_rt = ind_rt.dropna(subset=["gain3"])
    ind_rt["top5"] = ind_rt.groupby("date")["gain3"].rank(
        ascending=False, method="min") <= SECTOR_TOP
    df = df.merge(ind_rt[["industry", "date", "gain3", "top5"]],
                  on=["industry", "date"], how="left")
    return df


def run_strategy(df: pd.DataFrame, capital: float, max_total: float,
                 max_pos: float, start_date: str = "2016-10-01") -> dict:
    """低吸策略完整回测(选股过滤 → 逐日组合模拟)。返回统计字典。

    若 df 已含 __cand 列则直接用(便于变体测试), 否则按标准筛选计算。
    """
    if "__cand" not in df.columns:
        mask = (
            df["isST"].fillna(0).eq(0)
            & df["mc"].between(MC_LOW, MC_HIGH)
            & df["amt_ma5"].ge(AMT5_MIN)
            & df["vol_ratio"].ge(VOL_RATIO_MIN)
            & df["top5"].eq(True)
            & df["pctChg"].abs().lt(LIMIT_PCT)
            & df["close"].gt(0)
        )
        df = df.copy()
        df["__cand"] = mask
    else:
        df = df.copy()
        mask = df["__cand"].astype(bool)

    # 候选按日期分组
    pairs = df.loc[mask, ["date", "code", "close", "gain3", "vol_ratio"]]
    cands_by_date: dict = {}
    for r in pairs.itertuples(index=False):
        cands_by_date.setdefault(r.date, []).append(
            (r.code, float(r.close), float(r.gain3), float(r.vol_ratio)))
    for k in cands_by_date:
        v = cands_by_date[k]
        v.sort(key=lambda c: (c[2], c[3]), reverse=True)

    # 行定位: code -> {date: df行号}
    code_off: dict = {}
    for c, sub in df.groupby("code", sort=False):
        code_off[c] = dict(zip(sub["date"], sub.index))
    # 列数组(加速标量访问)
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    cl = df["close"].to_numpy()
    _ = cl

    cal = sorted(df["date"].unique())
    cal = [d for d in cal if d >= start_date]
    di = {d: i for i, d in enumerate(cal)}

    cash = capital
    nav = np.full(len(cal), np.nan)
    n_open_series = np.zeros(len(cal))
    positions = []              # (code, shares, cost, entry_date, last_px, stop_pct)
    trades = []                 # (code, entry, shares, cost, px, reason, ret)
    month_nav = {}
    month_reduced = set()
    px_today = {}               # code -> 今日(或最后可见)收盘价, 供盘中估值

    def buy_fee(cost):
        return max(COST_MIN_COMM, cost * COST_COMM)

    def sell_fee(gross):
        return (max(COST_MIN_COMM, gross * COST_COMM)
                + gross * COST_STAMP + gross * SLIP)

    def bar(code, d):
        off = code_off.get(code, {})
        if d not in off:
            return None
        i = off[d]
        return hi[i], lo[i], cl[i]

    prev_equity = capital
    for i, d in enumerate(cal):
        # 当日停牌股用最后可见收盘价估值(不因停牌归零)
        px_today = {}
        # ---- 1) 处理止盈止损(先止损后止盈, 保守) ----
        kept = []
        for pos in positions:
            code, shares, cost, e_date, last_px, stop_pct = pos
            b = bar(code, d)
            if b is None:
                # 停牌/无当日行情: 保持原样, NAV 用 last_px
                px_today[code] = last_px
                kept.append(pos)
                continue
            high, low, close = b
            px_today[code] = close
            basis = cost / shares
            # 止损线: 减仓后残余仓位已提到成本线(保本), 首次开仓为 -3%
            if low <= basis * (1 + stop_pct) and stop_pct < 0:
                # 止损全卖
                px = basis * (1 + stop_pct) * (1 - SLIP)
                gross = shares * px
                fee = sell_fee(gross)
                cash += gross - fee
                trades.append((code, e_date, shares, cost, px, "STOP", px/basis-1))
                continue
            elif low <= basis * (1 + stop_pct) and stop_pct >= 0:
                # 保本线: 残余仓位跌破成本 -> 平掉残余
                px = basis * (1 + stop_pct) * (1 - SLIP)
                gross = shares * px
                fee = sell_fee(gross)
                cash += gross - fee
                trades.append((code, e_date, shares, cost, px, "BREAK", px/basis-1))
                continue
            if high >= basis * (1 + TP3):
                if high >= basis * (1 + TP8):
                    # ≥8% 全卖
                    px = basis * (1 + TP8) * (1 - SLIP)
                    gross = shares * px
                    fee = sell_fee(gross)
                    cash += gross - fee
                    trades.append((code, e_date, shares, cost, px, "TP8", px/basis-1))
                    continue
                # 3%~8% 减半, 残余仓位止损提至成本线
                sell_n = shares // 2
                if sell_n <= 0:
                    kept.append((code, shares, cost, e_date, close, stop_pct))
                    continue
                px = basis * (1 + TP3) * (1 - SLIP)
                gross = sell_n * px
                fee = sell_fee(gross)
                cash += gross - fee
                trades.append((code, e_date, sell_n,
                               cost * (sell_n/shares), px, "TP3", px/basis-1))
                shares -= sell_n
                cost = cost * shares / (shares + sell_n)
                if shares > 0:
                    kept.append((code, shares, cost, e_date, close, 0.0))  # 止损移至成本线
                continue
            kept.append((code, shares, cost, e_date, close, stop_pct))
        positions = kept

        # 卖出后盘中市值(风控口径: 以今日收盘价计近似)
        mv_now = 0.0
        for pos in positions:
            code, shares, cost, e_date, last_px, stop_pct = pos
            mv_now += shares * px_today[code]
        equity_now = cash + mv_now

        # ---- 2) 风控 ----
        can_buy = True
        if prev_equity > 0 and prev_equity - equity_now > capital * 0.01:
            can_buy = False          # 单日亏损>总资金1%, 今日停买
        month = d[:7]
        max_total_eff = max_total
        if month in month_nav and month not in month_reduced:
            mmdd = (month_nav[month] - equity_now) / month_nav[month]
            if mmdd > 0.08:
                month_reduced.add(month)   # 当月仓位降至2成
                max_total_eff = min(max_total_eff, capital * 0.2)
        elif month in month_reduced:
            max_total_eff = min(max_total_eff, capital * 0.2)

        # ---- 3) 筛选候选(排除已持仓) 并尾盘建仓 ----
        cands = [c for c in cands_by_date.get(d, [])
                 if not any(p[0] == c[0] for p in positions)]
        if can_buy:
            for code, cprice, _g, _v in cands:
                if len(positions) >= 2:
                    break
                cur_cost = sum(p[2] for p in positions)
                # 单只上限 max_pos(未持有该股), 总仓上限 max_total_eff
                shares = int(min(max_pos, cash) // (cprice * 100)) * 100
                if shares <= 0:
                    continue
                total_cost = cur_cost + shares * cprice
                if total_cost > max_total_eff:
                    avail = max_total_eff - cur_cost
                    if avail <= 0:
                        continue
                    shares = int(avail // (cprice * 100)) * 100
                    if shares <= 0:
                        continue
                cost = shares * cprice
                fee = buy_fee(cost)
                if cash < cost + fee:
                    continue
                cash -= cost + fee
                positions.append((code, shares, cost, d, cprice, -STOP_PCT))

        # 收盘盯市 == 尾盘 (建仓用收盘价, 持仓也用收盘价), 等效盘中
        mv = 0.0
        for pos in positions:
            code, shares, cost, e_date, last_px, _sp = pos
            mv += shares * px_today.get(code, last_px)
        nav[i] = cash + mv
        n_open_series[i] = len(positions)
        prev_equity = nav[i]

        if month not in month_nav:
            month_nav[month] = nav[i]

    # ---- 期末平仓 ----
    for pos in list(positions):
        code, shares, cost, e_date, last_px, _sp = pos
        b = bar(code, cal[-1])
        px = (b[2] if b else last_px) * (1 - SLIP)
        gross = shares * px
        fee = sell_fee(gross)
        cash += gross - fee
        trades.append((code, e_date, shares, cost, px, "END", px/(cost/shares)-1))

    # ---- 统计 ----
    first = int(np.argmax(np.isfinite(nav)))
    dates = cal[first:]
    navv = np.nan_to_num(nav[first:])
    total_ret = navv[-1]/capital - 1
    yrs = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    cagr = (navv[-1]/capital) ** (1/yrs) - 1 if yrs > 0 else np.nan
    peak = np.maximum.accumulate(navv)
    mdd = float(((peak - navv) / peak).max())
    rets = np.diff(navv)/navv[:-1]
    sharpe = float(np.mean(rets)/np.std(rets)*np.sqrt(242)) if np.std(rets) > 0 else np.nan

    wins = [t for t in trades if t[6] > 0]
    losses = [t for t in trades if t[6] <= 0]
    avg_w = mean(t[6] for t in wins) if wins else 0.0
    avg_l = mean(t[6] for t in losses) if losses else 0.0
    plr = abs(avg_w/avg_l) if avg_l else float("inf")

    return {
        "dates": dates, "nav": navv, "n_open": n_open_series[first:],
        "trades": trades, "capital": capital,
        "total_ret": total_ret, "cagr": cagr, "mdd": mdd, "sharpe": sharpe,
        "n_trades": len(trades),
        "win_rate": len(wins)/(len(trades) or 1),
        "plr": plr, "avg_w": avg_w, "avg_l": avg_l,
        "avg_open": float(np.mean(n_open_series[first:])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="日线回测版短线低吸策略")
    ap.add_argument("--data", default="a_share_daily_hfq")
    ap.add_argument("--industry", default="industry_map.csv")
    ap.add_argument("--capital", type=float, default=50_000)
    ap.add_argument("--max-total", type=float, default=20_000)
    ap.add_argument("--max-pos", type=float, default=10_000)
    ap.add_argument("--limit", type=int, default=0, help="调试: 只载入前N只")
    ap.add_argument("--start", default="2016-10-01")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 62)
    print(f"短线低吸策略 | 日线回测 | 本金 {args.capital/1e4:.0f}万")
    print(f"持仓: 单只≤{args.max_pos/1e4:.0f}万 总仓≤{args.max_total/1e4:.0f}万 ≤2只")

    print("行业映射...")
    industry = fetch_industry(args.industry)
    print(f"  {len(industry)} 只")

    print("载入主板日线...")
    df = load_main_board(args.data, args.limit)
    n_codes = df["code"].nunique()
    print(f"  {n_codes} 只 | {len(df)} 行 | {df['date'].min()} ~ {df['date'].max()}")

    print("预计算指标 + 模拟...")
    df = precompute(df, industry)
    res = run_strategy(df, args.capital, args.max_total, args.max_pos,
                       start_date=args.start)

    print("=" * 62)
    print(f"期末总资产    : {res['nav'][-1]:,.2f} 元")
    print(f"总收益率      : {res['total_ret']:+.1%}    CAGR {res['cagr']:+.1%}")
    print(f"最大回撤      : {res['mdd']:.1%}")
    print(f"日频夏普      : {res['sharpe']:.2f}")
    print(f"交易次数      : {res['n_trades']} 笔")
    print(f"胜率          : {res['win_rate']:.1%}")
    print(f"盈亏比        : {res['plr']:.2f}  (均盈{res['avg_w']:+.2%} 均亏{res['avg_l']:+.2%})")
    print(f"平均持仓      : {res['avg_open']:.1f} 只")
    pd.DataFrame({"date": res["dates"], "nav": res["nav"]}).to_csv(
        "trades_lowdip_nav.csv", index=False, encoding="utf-8-sig")
    print("净值曲线: trades_lowdip_nav.csv")
    print(f"用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
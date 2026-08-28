# -*- coding: utf-8 -*-
"""
strategy_cb_doublelow.py — 可转债双低轮动策略回测
==================================================

双低值 = 转债收盘价 + 转股溢价率(%，来自 bond_zh_cov_value_analysis)
策略：定期再平衡，从"当时在市"的转债池中按双低值升序选前 top_n 只等权持有，
每周(或每 N 个交易日)重排：卖出掉出前 top_n 的、买入新进前 top_n 的。

数据
----
cb_data/<code>.csv: date,close,bond_value,convert_value,bond_premium,convert_premium
cb_data/cb_list.csv: 全量转债列表(含退市, 用于券池)

核心要点
--------
- "在市"判断：某调仓日该券有收盘价数据即视为在市 —— 天然排除未上市/已退市券，
  且逐日独立判定，无幸存者偏差(不因"现在退市了"而抹掉历史)。
- 转债无印花税：成本 = 佣金(万0.5, 最低1元) + 滑点(0.05%/边)。
- 生成交易明细(entry/exit)喂给 sim_portfolio.simulate 做逐日盯市净值。

用法
----
    python strategy_cb_doublelow.py                    # 默认: top10, 周频
    python strategy_cb_doublelow.py --top-n 15 --interval 5
    python strategy_cb_doublelow.py --capital 50000 --notional 10000
    python strategy_cb_doublelow.py --min-years 1      # 剔除剩余到期<1年的
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# 默认成本(转债: 无印花税, 佣金更低, 最低1元)
COMMISSION = 0.0005      # 万5
MIN_COMM = 1.0
STAMP = 0.0              # 转债无印花税
SLIP = 0.0005            # 滑点 0.05%/边


def load_all(data_dir: str) -> dict:
    """读全部转债为 {code: DataFrame(date,close,convert_premium,...)}。"""
    out = {}
    for f in glob.glob(os.path.join(data_dir, "[0-9]*.csv")):
        code = os.path.basename(f)[:-4]
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            if "close" not in df or "convert_premium" not in df:
                continue
            df = df.dropna(subset=["date"])
            if df.empty:
                continue
            out[code] = df
        except Exception:  # noqa: BLE001
            continue
    return out


def is_valid_cb_code(code: str) -> bool:
    """有效的转债代码: 排除退市板块(4/3开头, 三板块)。"""
    return code and code[0] not in ("4", "3", "2")


def cb_universe_on(all_cb: dict, date: str) -> list:
    """某调仓日在市的转债列表(该日有收盘价且代码有效)。"""
    return [c for c, df in all_cb.items()
            if is_valid_cb_code(c)
            and any(d == date and pd.notna(r) and r > 80
                    for d, r in zip(df["date"], pd.to_numeric(df["close"], errors="coerce")))]


def make_trades(all_cb: dict, universe_dates: list, top_n: int, interval: int,
                min_price: float = None, min_years: float = None,
                cal: list = None) -> pd.DataFrame:
    """生成换仓交易明细。

    每 interval 个交易日为一个调仓窗口。窗口内持有该期选中的 top_n。
    持仓变动(卖出旧、买入新)在窗口起始日成交，成本按持仓变化计。
    返回 trades DataFrame(code, entry_date, entry_price, exit_date, exit_price, ret, priority)。
    """
    # 交易日历(用任一老券的日期或传入)
    if cal is None:
        cal = sorted({d for df in all_cb.values() for d in df["date"]})
    # 调仓日: 从有足够数据的起点开始
    day_idx = {d: i for i, d in enumerate(cal)}
    trades = []
    # 每期选中持仓(集合)，用于生成买卖
    # 简化: 逐调仓日计算目标组合，合并成"每只券的一段持有期"
    # 采用: 对每个调仓日 i(等间距)，选 top_n；持有到下一调仓日。
    # 生成交易 = 该券在"进入目标组合"的调仓日买入, "离开目标组合"的调仓日卖出。
    held = {}  # code -> entry_date, entry_price
    open_pos = {}  # code -> {entry_date, entry_price}
    rebalance_days = [cal[i] for i in range(0, len(cal), interval)]

    for rb_date in rebalance_days:
        uni = cb_universe_on(all_cb, rb_date)
        # 计算每只候选的双低值
        cands = []
        for c in uni:
            df = all_cb[c]
            row = df[df["date"] == rb_date]
            if row.empty:
                continue
            row = row.iloc[0]
            close = row.get("close")
            prem = row.get("convert_premium")
            if pd.isna(close) or pd.isna(prem) or not np.isfinite(close):
                continue
            if min_price and close < min_price:
                continue
            cands.append((c, close, prem))
        cands.sort(key=lambda x: x[1] + x[2])  # 双低值升序
        target = [c for c, _, _ in cands[:top_n]]

        # 卖出: 上期持有但本期不在目标
        for c in list(open_pos):
            if c not in target:
                p = open_pos.pop(c)
                exit_row = all_cb[c][all_cb[c]["date"] == rb_date]
                ex_price = float(exit_row.iloc[0]["close"]) if not exit_row.empty else p["entry_price"]
                ret = ex_price / p["entry_price"] - 1
                trades.append({"code": c, "entry_date": p["entry_date"],
                               "entry_price": p["entry_price"],
                               "exit_date": rb_date, "exit_price": ex_price,
                               "ret": ret, "priority": 0})
        # 买入: 新进入目标
        for c in target:
            if c not in open_pos:
                row = all_cb[c][all_cb[c]["date"] == rb_date].iloc[0]
                open_pos[c] = {"entry_date": rb_date, "entry_price": float(row["close"])}

    # 期末平掉所有剩余持仓(用最后可用收盘价)
    last_date = cal[-1]
    for c, p in open_pos.items():
        df = all_cb[c]
        tail = df[df["date"] <= last_date]
        ex_price = float(tail.iloc[-1]["close"]) if not tail.empty else p["entry_price"]
        ret = ex_price / p["entry_price"] - 1
        trades.append({"code": c, "entry_date": p["entry_date"],
                       "entry_price": p["entry_price"],
                       "exit_date": tail.iloc[-1]["date"] if not tail.empty else p["entry_date"],
                       "exit_price": ex_price, "ret": ret, "priority": 0})

    return pd.DataFrame(trades)


def yearly_table(trades: pd.DataFrame) -> str:
    """按 exit_date 年份汇总已实现交易。"""
    if trades.empty:
        return "无交易"
    t = trades.copy()
    t["year"] = t["exit_date"].astype(str).str[:4]
    lines = []
    for y, g in t.groupby("year"):
        n = len(g)
        wr = (g["ret"] > 0).mean()
        avg = g["ret"].mean()
        lines.append(f"  {y}: {n}笔 均{avg:+.2%} 胜率{wr:.0%}")
    return "\n".join(lines)


def rotation_simulate(all_cb: dict, cal: list, top_n: int, interval: int,
                      capital: float, min_price: float = 80, max_price: float = 300,
                      commission: float = COMMISSION, min_comm: float = MIN_COMM,
                      slip: float = SLIP) -> dict:
    """满仓等权轮动模拟。

    每个调仓日：从在市转债按双低值升序选 top_n，等权分配资金，
    用收盘价成交(含佣金+滑点)，持到下一调仓日重排。
    转债交易单位: 1手=10张, 面值100, 即 1000元/手, 整手 = floor(资金/价格/10)。
    返回与 simulate 同构的 dict。
    """
    # 转债价序列: {code: {date: close}}
    price = {}
    for c, df in all_cb.items():
        price[c] = dict(zip(df["date"], df["close"]))

    # 收盘价 lookup(持仓盯市用)
    def close_of(c, d):
        m = price.get(c)
        if m is None:
            return np.nan
        v = m.get(d)
        return v if v is not None and not (isinstance(v, float) and np.isnan(v)) else np.nan

    # 调仓日
    rb = [cal[i] for i in range(0, len(cal), interval)]
    if not rb or rb[-1] != cal[-1]:
        rb.append(cal[-1])  # 保证期末平仓

    cash = capital
    holdings = []  # [{code, shares, cost}]
    nav = np.full(len(cal), np.nan)
    n_open_series = np.zeros(len(cal))
    trades = []

    for i, d in enumerate(cal):
        # 1) 若是调仓日: 先全卖再全买
        if d in rb:
            # 卖出全部持仓(按当日收盘, 若无价则按上次价)
            for h in holdings:
                px = close_of(h["code"], d)
                if np.isnan(px):
                    px = h["cost"] / h["shares"]
                gross = h["shares"] * px
                fee = max(min_comm, gross * commission) + gross * slip
                cash += gross - fee
                if h.get("entry_date") is not None and h.get("exit") is None:
                    h["exit"] = (d, px)
            holdings = []

            # 选择目标组合
            uni = cb_universe_on(all_cb, d)
            cands = []
            for c in uni:
                row = all_cb[c][all_cb[c]["date"] == d]
                if row.empty:
                    continue
                row = row.iloc[0]
                cl = row.get("close"); pr = row.get("convert_premium")
                if pd.isna(cl) or pd.isna(pr) or not np.isfinite(cl):
                    continue
                if min_price and cl < min_price:
                    continue
                if max_price and cl > max_price:
                    continue
                cands.append((c, cl + pr))  # 双低值
            cands.sort(key=lambda x: x[1])
            target = [c for c, _ in cands[:top_n]]
            if not target:
                continue

            # 等权分配资金
            alloc = cash / len(target)
            for c in target:
                px = close_of(c, d)
                if np.isnan(px) or px <= 0:
                    continue
                shares = int(alloc / (px * 10)) * 10   # 1手=10张
                if shares <= 0:
                    continue
                cost = shares * px
                fee = max(min_comm, cost * commission)
                if cost + fee > cash:
                    continue
                cash -= cost + fee
                holdings.append({"code": c, "shares": shares, "cost": cost,
                                 "entry_date": d})

        # 2) 盯市
        mv = 0.0
        for h in holdings:
            px = close_of(h["code"], d)
            if np.isfinite(px):
                mv += h["shares"] * px
        nav[i] = cash + mv
        n_open_series[i] = len(holdings)

    # 期末: 最后一天若仍是持仓(最后一调仓日到期末)记为平仓价
    # (nav 已含市值, 统计用 last nav)
    last_date = cal[-1]

    start_idx = int(np.argmax(np.isfinite(nav)))
    dates = cal[start_idx:]
    navv = np.nan_to_num(nav[start_idx:], nan=capital)
    total_ret = navv[-1] / capital - 1
    yrs = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    cagr = (navv[-1] / capital) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    peak = np.maximum.accumulate(navv)
    mdd = float(((peak - navv) / peak).max())
    r = np.diff(navv) / navv[:-1]
    sharpe = float(np.mean(r) / np.std(r) * np.sqrt(242)) if np.std(r) > 0 else np.nan
    return {
        "dates": dates, "nav": navv, "n_open": n_open_series[start_idx:],
        "total_ret": total_ret, "cagr": cagr, "mdd": mdd, "sharpe": sharpe,
        "final_open": len(holdings), "max_open": top_n,
        "avg_open": float(n_open_series[start_idx:].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="可转债双低轮动回测",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--data", default="cb_data", help="转债数据目录(默认 cb_data)")
    ap.add_argument("--top-n", type=int, default=10, help="持有只数(默认10)")
    ap.add_argument("--interval", type=int, default=5,
                    help="调仓间隔交易日(默认5≈周频)")
    ap.add_argument("--min-price", type=float, default=None,
                    help="剔除低于该价格的转债(元)")
    ap.add_argument("--capital", type=float, default=50_000, help="初始资金(默认5万)")
    ap.add_argument("--trades-out", default="trades_cb.csv", help="交易明细输出文件")
    args = ap.parse_args()

    print("=" * 62)
    print(f"可转债双低轮动 | 数据 {args.data} | 持 {args.top_n} 只 | 调仓 {args.interval}日")
    all_cb = load_all(args.data)
    print(f"转债数据: {len(all_cb)} 只")

    cal = sorted({d for df in all_cb.values() for d in df["date"]})
    cal = [d for d in cal if d >= "2021-01-01"]
    print(f"交易日历: {len(cal)} 日 ({cal[0]} ~ {cal[-1]})")

    print("\n" + "=" * 62)
    print(f"组合模拟(满仓等权轮动 | 转债成本: 佣金万0.5/最低1元, 无印花税, 滑点0.05%)")
    res = rotation_simulate(all_cb, cal, args.top_n, args.interval,
                            args.capital, min_price=args.min_price)
    print(f"期末净值        : {res['nav'][-1]/1e4:.2f}万  (初始 {args.capital/1e4:.0f}万)")
    print(f"总收益率        : {res['total_ret']:+.1%}    CAGR {res['cagr']:+.1%}")
    print(f"期间            : {res['dates'][0]} ~ {res['dates'][-1]}")
    print(f"最大回撤(日频)  : {res['mdd']:.1%}")
    print(f"并发持仓: 最大 {res['max_open']} | 平均 {res['avg_open']:.1f} | 期末 {res['final_open']}")
    print(f"日频夏普(年化)  : {res['sharpe']:.2f}")
    pd.DataFrame({"date": res["dates"], "nav": res["nav"],
                  "n_open": res["n_open"]}).to_csv(args.trades_out.replace(
                      "trades_cb.csv", "trades_cb_nav.csv"), index=False,
                      encoding="utf-8-sig")
    print(f"净值曲线已保存: trades_cb_nav.csv")


if __name__ == "__main__":
    main()

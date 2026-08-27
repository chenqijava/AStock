#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sim_portfolio.py — 组合级净值模拟(真实资金曲线)
================================================

读取策略回测产生的交易明细(每笔含 entry/exit 日期与价格)，模拟一个真实账户：

- 初始资金 capital，每笔建仓固定名义本金 notional(A股整手100股)；
- 资金充足即建仓；不足则跳过该信号(报告跳过率)；同一股票互不重叠
  (单股回测已保证)，不同股票可同时持有；
- 逐日盯市：NAV = 现金 + 持仓市值(收盘价)，输出日频净值曲线与统计；
- 成本按实际持仓量重算(佣金/边最低5元、印花税卖、滑点/边)；
- 同日信号按固定随机种子打乱建仓顺序(避免按代码排序把 bj./高价股排前
  造成系统偏差)。

用法
----
    python sim_portfolio.py --trades trades_panic.csv --data a_share_daily_hfq \
        --capital 1000000 --notional 20000

也可作为库被其它策略脚本调用：from sim_portfolio import simulate
"""

import argparse
import os

import numpy as np
import pandas as pd


def simulate(trades: pd.DataFrame, data_dir: str, capital: float, notional: float,
             commission: float = 0.00025, min_commission: float = 5.0,
             stamp: float = 0.0005, slip: float = 0.001,
             seed: int = 20260827, priority_asc: bool = False,
             max_exposure: float = None) -> dict:
    """对交易明细做逐日盯市净值模拟。

    返回 dict：dates/nav/n_open(日频序列)，total_ret/cagr/mdd/sharpe，
    funded/skipped/fund_rate，max_open/avg_open/final_open，ret_taken/ret_skipped。

    max_exposure: 总仓位上限(占总资金比例, 如0.5=5成)。建仓后累计持仓成本占比
    超过该值则跳过信号；None=不设上限(沿用旧行为)。资金不足导致的跳过也计入。
    """
    trades = trades.copy()
    rng = np.random.RandomState(seed)
    trades["_rk"] = rng.rand(len(trades))
    # priority 列(如市值)存在时按它优先建仓(priority_asc=False=大优先, True=小优先)，
    # 再随机打平；否则同日随机
    if "priority" in trades.columns:
        trades = trades.sort_values(["entry_date", "priority", "_rk"],
                                    ascending=[True, priority_asc, True]).reset_index(drop=True)
    else:
        trades = trades.sort_values(["entry_date", "_rk"]).reset_index(drop=True)

    cal = sorted(pd.read_csv(os.path.join(data_dir, "sh.600000.csv"),
                             usecols=["date"])["date"].tolist())
    di = {d: i for i, d in enumerate(cal)}
    n = len(cal)

    cache: dict = {}

    def close_of(code: str, date: str):
        s = cache.get(code)
        if s is None:
            try:
                s = pd.read_csv(os.path.join(data_dir, code + ".csv"),
                                usecols=["date", "close"]).set_index("date")["close"]
            except Exception:                                    # noqa: BLE001
                s = None
            cache[code] = s
        if s is None:
            return np.nan
        return float(s.get(date, np.nan))

    comm, minc, stamp_, slip_ = commission, min_commission, stamp, slip
    cash = capital
    pos_cost = 0.0                   # 累计持仓成本(用于总仓位上限)
    nav = np.full(n, np.nan)
    n_open_series = np.zeros(n)
    open_pos = []                     # {code, shares, exit_price, exit_idx, cost}
    ei = skipped = n_taken = 0
    max_open = 0
    ret_taken, ret_skipped = [], []
    for i, d in enumerate(cal):
        # 1) 平仓(先于建仓，释放资金)
        keep = []
        for p in open_pos:
            if p["exit_idx"] == i:
                gross = p["shares"] * p["exit_price"]
                fee = max(minc, gross * comm) + gross * stamp_ + gross * slip_
                cash += gross - fee
                pos_cost -= p["cost"]
            else:
                keep.append(p)
        open_pos = keep
        # 2) 建仓
        while ei < len(trades) and trades.loc[ei, "entry_date"] == d:
            t = trades.loc[ei]
            ei += 1
            ep = float(t["entry_price"])
            if not (ep > 0):
                skipped += 1
                ret_skipped.append(float(t["ret"]))
                continue
            shares = int(notional // (ep * 100)) * 100
            if shares <= 0:
                skipped += 1
                ret_skipped.append(float(t["ret"]))
                continue
            cost = shares * ep
            fee = max(minc, cost * comm)
            if cash < cost + fee:
                skipped += 1
                ret_skipped.append(float(t["ret"]))
                continue
            if max_exposure and (pos_cost + cost) / capital > max_exposure:
                skipped += 1
                ret_skipped.append(float(t["ret"]))
                continue
            cash -= cost + fee
            pos_cost += cost
            ret_taken.append(float(t["ret"]))
            ex = di.get(str(t["exit_date"]), n - 1)
            pos = {"code": str(t["code"]), "shares": shares,
                   "exit_price": float(t["exit_price"]), "exit_idx": ex, "cost": cost}
            open_pos.append(pos)
            n_taken += 1
            if ex == i:               # 同日平仓(跳空止损)
                gross = shares * float(t["exit_price"])
                s_fee = max(minc, gross * comm) + gross * stamp_ + gross * slip_
                cash += gross - s_fee
                pos_cost -= cost
                open_pos.pop()
        # 3) 盯市
        mv = 0.0
        for p in open_pos:
            c = close_of(p["code"], d)
            if np.isfinite(c):
                mv += p["shares"] * c
        nav[i] = cash + mv
        n_open_series[i] = len(open_pos)
        max_open = max(max_open, len(open_pos))

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
        "funded": n_taken, "skipped": skipped,
        "fund_rate": n_taken / (n_taken + skipped) if (n_taken + skipped) else 0.0,
        "max_open": max_open, "avg_open": float(n_open_series[start_idx:].mean()),
        "final_open": len(open_pos),
        "ret_taken": ret_taken, "ret_skipped": ret_skipped,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="组合级净值模拟")
    ap.add_argument("--trades", default="trades_panic.csv", help="交易明细CSV")
    ap.add_argument("--data", default="a_share_daily_hfq", help="价格数据目录(盯市用)")
    ap.add_argument("--capital", type=float, default=1_000_000, help="初始资金(元)")
    ap.add_argument("--notional", type=float, default=20_000, help="每笔名义本金(元)")
    ap.add_argument("--commission", type=float, default=0.00025)
    ap.add_argument("--min-commission", type=float, default=5.0)
    ap.add_argument("--stamp", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.001)
    args = ap.parse_args()

    trades = pd.read_csv(args.trades, encoding="utf-8-sig")
    print(f"交易明细: {len(trades)} 笔 | 资金 {args.capital/1e4:.0f}万 | 每笔 {args.notional/1e4:.1f}万")
    res = simulate(trades, args.data, args.capital, args.notional,
                   args.commission, args.min_commission, args.stamp, args.slip)

    print("=" * 62)
    print(f"期末净值        : {res['nav'][-1]/1e4:.1f}万  (初始 {args.capital/1e4:.0f}万)")
    print(f"总收益率        : {res['total_ret']:+.1%}    年化(CAGR) {res['cagr']:+.1%}")
    print(f"期间            : {res['dates'][0]} ~ {res['dates'][-1]}")
    print(f"最大回撤(日频)  : {res['mdd']:.1%}")
    print(f"建仓 {res['funded']} 笔 / 跳过 {res['skipped']} 笔 (资金不足, 命中率 {res['fund_rate']:.1%})")
    print(f"并发持仓: 最大 {res['max_open']} 笔 | 平均 {res['avg_open']:.1f} 笔 | 期末 {res['final_open']} 笔")
    print(f"日频夏普(年化)  : {res['sharpe']:.2f}")
    if res["ret_taken"] or res["ret_skipped"]:
        import statistics
        mt = statistics.mean(res["ret_taken"]) if res["ret_taken"] else 0.0
        ms = statistics.mean(res["ret_skipped"]) if res["ret_skipped"] else 0.0
        print(f"[诊断] 被选中均收益 {mt:+.3%} vs 被跳过均收益 {ms:+.3%} "
              f"({len(res['ret_taken'])}/{len(res['ret_skipped'])}笔)")

    out = os.path.splitext(args.trades)[0] + f"_nav_{int(args.capital//10000)}w.csv"
    pd.DataFrame({"date": res["dates"], "nav": res["nav"],
                  "n_open": res["n_open"]}).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"净值曲线已保存: {out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""列出定稿配置(10万/每笔2万/小市值优先/稳健信号)实际建仓的92笔明细。
完全复刻 simulate() 的 notional 分支资金逻辑 + priority 排序。"""
import os, numpy as np, pandas as pd

trades = pd.read_csv("trades_panic_v15.csv", encoding="utf-8-sig")
trades["priority"] = trades["mc"]
trades = trades.copy()
rng = np.random.RandomState(20260827)
trades["_rk"] = rng.rand(len(trades))
trades = trades.sort_values(["entry_date", "priority", "_rk"],
                            ascending=[True, True, True]).reset_index(drop=True)

data_dir = "a_share_daily_hfq"
cal = sorted(pd.read_csv(os.path.join(data_dir, "sh.600000.csv"),
             usecols=["date"])["date"].tolist())
di = {d: i for i, d in enumerate(cal)}
n = len(cal)
cache = {}
def close_of(code, date):
    s = cache.get(code)
    if s is None:
        try:
            s = pd.read_csv(os.path.join(data_dir, code + ".csv"),
                            usecols=["date", "close"]).set_index("date")["close"]
        except Exception:
            s = None
        cache[code] = s
    if s is None: return np.nan
    return float(s.get(date, np.nan))

capital, notional = 100_000.0, 20_000.0
comm, minc, stamp_, slip_ = 0.00025, 5.0, 0.0005, 0.001
cash = capital
open_pos, funded = [], []
ei = skipped = 0
funded_cost = {}
for i, d in enumerate(cal):
    keep = []
    for p in open_pos:
        if p["exit_idx"] == i:
            gross = p["shares"] * p["exit_price"]
            fee = max(minc, gross * comm) + gross * stamp_ + gross * slip_
            cash += gross - fee
        else:
            keep.append(p)
    open_pos = keep
    day_trades = []
    while ei < len(trades) and trades.loc[ei, "entry_date"] == d:
        day_trades.append(trades.loc[ei]); ei += 1
    for t in day_trades:
        ep = float(t["entry_price"])
        if not (ep > 0):
            skipped += 1; continue
        shares = int(notional // (ep * 100)) * 100
        if shares <= 0:
            skipped += 1; continue
        cost = shares * ep
        fee = max(minc, cost * comm)
        if cash < cost + fee:
            skipped += 1; continue
        cash -= cost + fee
        ex = di.get(str(t["exit_date"]), n - 1)
        funded.append({**dict(t), "shares": shares, "cost": cost})
        open_pos.append({"code": str(t["code"]), "shares": shares,
                         "exit_price": float(t["exit_price"]), "exit_idx": ex, "cost": cost})
        if ex == i:
            gross = shares * float(t["exit_price"])
            s_fee = max(minc, gross * comm) + gross * stamp_ + gross * slip_
            cash += gross - s_fee
            open_pos.pop()

print(f"建仓 {len(funded)} 笔 / 跳过 {skipped}")
print(f"{'日期':<11}{'代码':<11}{'股数':>5} {'成本':>8} {'入价':>8}{'出价':>8}{'离场':>5}{'收益':>8}{'天数':>4} 市值")
tot_ret = 0.0
for t in funded:
    ep = float(t["entry_price"]); xp = float(t["exit_price"])
    net = (xp / ep - 1) * 100
    tot_ret += net
    print(f"{str(t['entry_date']):<11}{str(t['code']):<11}{int(t['shares']):>5} "
          f"{float(t['cost']):>8.0f} {ep:>7.2f} {xp:>7.2f} {str(t['reason']):<5}"
          f"{net:>+7.2f}% {int(t['bars']):>3} {float(t['mc']):.0f}")
print(f"均收益 {tot_ret/len(funded):+.2f}%  (简单平均, 未加权)")
pd.DataFrame([{"entry_date": t["entry_date"], "code": t["code"],
               "shares": t["shares"], "cost": t["cost"],
               "entry_price": t["entry_price"], "exit_date": t["exit_date"],
               "exit_price": t["exit_price"], "reason": t["reason"],
               "ret": t["ret"], "bars": t["bars"], "mc": t["mc"]}
              for t in funded]).to_csv(
    "trades_panic_v15_funded_10w.csv", index=False, encoding="utf-8-sig")
print("已保存: trades_panic_v15_funded_10w.csv")

# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from sim_portfolio import simulate

base = pd.read_csv("trades_panic_v15.csv", encoding="utf-8-sig")

def run(label, pri_col, asc=True):
    t = base.copy()
    t["priority"] = t[pri_col]
    r = simulate(t, "a_share_daily_hfq", 100_000, 20_000,
                 priority_asc=asc, seed=20260827)
    mt = sum(r["ret_taken"])/len(r["ret_taken"]) if r["ret_taken"] else 0
    ms = sum(r["ret_skipped"])/len(r["ret_skipped"]) if r["ret_skipped"] else 0
    print(f"{label:26s} 总{r['total_ret']:+7.1%} CAGR{r['cagr']:+5.1%} "
          f"回撤{r['mdd']:5.1%} 夏普{r['sharpe']:4.2f} 选{mt:+.2%} 跳{ms:+.2%}")

run("小市值优先", "mc", asc=True)
run("大市值优先", "mc", asc=False)
run("低放量优先", "vol_ratio", asc=True)
run("高放量优先", "vol_ratio", asc=False)

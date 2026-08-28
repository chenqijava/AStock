# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
base = pd.read_csv("trades_panic_v15.csv", encoding="utf-8-sig")

def fund_order(label, keys):
    t = base.copy()
    t = t.sort_values([c for c, _ in reversed(keys)], kind="mergesort").reset_index(drop=True)
    t["priority"] = t.index.astype(float)
    t["_rk"] = np.random.RandomState(20260827).rand(len(t))
    t = t.sort_values(["entry_date", "priority", "_rk"],
                      ascending=[True, True, True]).reset_index(drop=True)
    d = t[t["entry_date"] == "2020-02-05"]
    print(f"{label}: 前5个信号 ->", d.head(5)["code"].tolist())
    print(f"          mc: {[round(m,1) for m in d.head(5)['mc']]}")

fund_order("小市值优先", [("mc", True)])
fund_order("大市值优先", [("mc", False)])
fund_order("高放量优先", [("vol_ratio", False)])

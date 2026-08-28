# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
base = pd.read_csv("trades_panic_v15.csv", encoding="utf-8-sig")
print("vol_ratio 分布 (稳健信号 429 笔):")
print(base["vol_ratio"].describe().round(2))
print()
low = base[base["vol_ratio"].between(1.5, 2.0)]
mid = base[base["vol_ratio"].between(2.0, 3.0)]
high = base[base["vol_ratio"] > 3.0]
for name, g in [("1.5~2.0x", low), ("2.0~3.0x", mid), (">3.0x", high)]:
    print(f"{name:10s} {len(g):3d}笔  均收益{float(np.mean(g['ret'])):+.2%}  胜率{float(np.mean(g['ret']>0)):.0%}")

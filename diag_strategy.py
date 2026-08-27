# -*- coding: utf-8 -*-
"""诊断：对中市值股票逐只比对 金叉数 vs 策略实际交易数。"""
import glob
import os

import pandas as pd

from strategy_ma_cross import backtest_stock, USECOLS

MC = set(open("codes_50_200.txt", encoding="utf-8").read().split())
FILES = [f for f in glob.glob("a_share_daily/*.csv")
         if os.path.basename(f) != "stock_list.csv"
         and os.path.basename(f).rsplit(".", 1)[0] in MC][:200]

CFG = dict(ma_fast=5, ma_slow=24, atr_period=14, window=6, pullback=0.1,
           tp_atr=4.0, sl_atr=2.0, time_stop=10, include_st=False, usecols=USECOLS,
           capital=10000, commission=0.00025, min_commission=5.0,
           stamp=0.0005, slip=0.001, lot=100)

total_cross = total_trade = 0
for f in FILES:
    code = os.path.basename(f).rsplit(".", 1)[0]
    df = pd.read_csv(f, usecols=USECOLS).sort_values("date").reset_index(drop=True)
    c = df["close"]
    ma5 = c.rolling(5).mean()
    ma24 = c.rolling(24).mean()
    cross = int(((ma5 > ma24) & (ma5.shift(1) <= ma24.shift(1)))
                .fillna(False).sum())
    n_tr = len(backtest_stock(CFG, code, df))
    total_cross += cross
    total_trade += n_tr
    print(f"{code}: crosses={cross:>3} trades={n_tr:>3}")

print(f"\nTOTAL: crosses={total_cross} trades={total_trade} "
      f"rate={total_trade / total_cross:.1%}")

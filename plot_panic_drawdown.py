# -*- coding: utf-8 -*-
"""
plot_panic_drawdown.py — 收益曲线 + 回撤双面板图
===============================================
对比默认(最优收益) vs 稳健型(低回撤) 的净值与水下回撤, PNG 输出。

用法
    python plot_panic_drawdown.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 中文字体(Windows)
for _f in ("Microsoft YaHei", "SimHei", "SimSun"):
    try:
        matplotlib.rcParams["font.sans-serif"] = [_f]
        break
    except Exception as exc:                    # noqa: BLE001
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

# (nav文件, 标签, 颜色, 主次)
SERIES = [
    ("nav_base.csv",     "默认策略（总收益最优）：+118.9% / CAGR 8.2% / 回撤 8.5% / 夏普1.28", "#b00"),
    ("nav_v15_n10w.csv", "稳健型（低回撤）：+113.5% / CAGR 7.9% / 回撤 3.9% / 夏普1.34",      "#06c"),
]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(13, 8), dpi=130,
        gridspec_kw={"height_ratios": [3, 1.6], "hspace": 0.12})

    for fname, label, color in SERIES:
        path = os.path.join(here, fname)
        if not os.path.exists(path):
            print(f"[跳过] 缺文件: {fname}")
            continue
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        nav = df["nav"].to_numpy(dtype=float)
        t = df["date"]

        ax0.plot(t, nav / 1e4, label=label, color=color, lw=1.8)

        # 回撤: 水下曲线(负值朝下)
        peak = np.maximum.accumulate(nav)
        dd = (nav / peak - 1) * 100
        ax1.fill_between(t, dd, 0, color=color, alpha=0.32)
        ax1.plot(t, dd, color=color, lw=1.0)

    ax0.set_title("恐慌日超跌反弹 — 收益曲线 vs 回撤（100万初始，后复权含成本）", fontsize=13)
    ax0.set_ylabel("净值（万元）")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left", fontsize=9)
    ax0.xaxis.set_major_locator(mdates.YearLocator())
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax1.set_ylabel("回撤（%）")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.set_xlabel("日期")

    fig.tight_layout()
    out = os.path.join(here, "panic_equity_drawdown.png")
    fig.savefig(out)
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
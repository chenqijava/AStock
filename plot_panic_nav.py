# -*- coding: utf-8 -*-
"""
plot_panic_nav.py — 恐慌策略净值曲线对比图
==========================================
读取各配置的 nav_*.csv(均为 100万初始), 画成 {date: nav} 曲线对比并保存 PNG。

用法
    python plot_panic_nav.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# 中文字体(Windows 微软雅黑/宋体; 缺则回退)
for _f in ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi"):
    try:
        matplotlib.rcParams["font.sans-serif"] = [_f]
        break
    except Exception as exc:                      # noqa: BLE001
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

# 显示配置的标签与行样式(基线不动, 变体按序)
PLOTS = [
    ("nav_base.csv",     "基线(无过滤)",       {"color": "#222", "lw": 2.2, "ls": "-"}),
    ("nav_v15.csv",      "vol≥1.5, n=2万",     {"color": "#888", "lw": 1.6, "ls": "--"}),
    ("nav_v15_n5w.csv",  "vol≥1.5, n=5万",     {"color": "#b00", "lw": 1.6, "ls": "-."}),
    ("nav_v15_n10w.csv", "vol≥1.5, n=10万",    {"color": "#06c", "lw": 1.9, "ls": "-"}),
    ("nav_v15_mc50.csv", "vol≥1.5, mc50-500",  {"color": "#096", "lw": 1.4, "ls": ":"}),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 7), dpi=130)
    made = []
    for fname, label, style in PLOTS:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
        if not os.path.exists(path):
            print(f"[跳过] 缺文件: {fname}")
            continue
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        ax.plot(df["date"], df["nav"] / 1e4, label=label, **style)
        made.append(fname)

    ax.set_title("恐慌日超跌反弹 — 净值对比(100万初始, 后复权, 含成本)", fontsize=13)
    ax.set_ylabel("净值(万元)")
    ax.set_xlabel("日期")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = "panic_nav_compare.png"
    fig.savefig(out)
    print(f"已保存: {out} (使用 {len(made)} 个配置: {', '.join(made)})")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""生成幸存者偏差量化分析可视化图表"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
fig.suptitle('幸存者偏差量化分析 — 恐慌超跌策略', fontsize=16, fontweight='bold', y=0.98)

# ---- 1. 核心指标对比柱状图 ----
ax1 = axes[0, 0]
metrics = ['CAGR', '总收益', '夏普', '胜率', 'PF']
now_vals = [8.1, 117.7, 1.28, 77.8, 4.61]
full_vals = [8.0, 115.7, 1.25, 76.7, 4.36]
x = np.arange(len(metrics))
w = 0.35
bars1 = ax1.bar(x - w/2, now_vals, w, label='当前样本(幸存者)', color='#2E86AB', alpha=0.85)
bars2 = ax1.bar(x + w/2, full_vals, w, label='全样本(含退市股)', color='#E63946', alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=10)
ax1.set_title('核心指标对比', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.set_ylabel('数值')
ax1.grid(axis='y', alpha=0.3)
for bars in [bars1, bars2]:
    for b in bars:
        v = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=8)

# ---- 2. 偏差幅度条形图 ----
ax2 = axes[0, 1]
delta_labels = ['ΔCAGR', 'Δ总收益', 'Δ夏普', 'Δ胜率', 'ΔPF']
deltas = [-0.1, -2.0, -0.03, -1.1, -0.25]
colors = ['#E63946' if d < 0 else '#2E86AB' for d in deltas]
bars = ax2.barh(delta_labels, deltas, color=colors, alpha=0.85)
ax2.set_title('偏差幅度 (全样本 - 当前样本)', fontsize=12, fontweight='bold')
ax2.set_xlabel('偏差值')
ax2.axvline(x=0, color='black', linewidth=0.8)
ax2.grid(axis='x', alpha=0.3)
for bar, d in zip(bars, deltas):
    offset = -0.15 if d < 0 else 0.05
    ax2.text(d + offset, bar.get_y() + bar.get_height()/2, f'{d:+.2f}', 
             va='center', fontsize=9, fontweight='bold')

# ---- 3. 退市股 vs 在市股个股层对比 ----
ax3 = axes[1, 0]
categories = ['胜率%', '平均净收益%', 'PF']
in_market = [77.8, 6.44, 4.61]
delisted = [50.0, 1.03, 1.43]
x3 = np.arange(len(categories))
w3 = 0.35
b1 = ax3.bar(x3 - w3/2, in_market, w3, label='在市股(1656笔)', color='#2E86AB', alpha=0.85)
b2 = ax3.bar(x3 + w3/2, delisted, w3, label='退市股(4笔)', color='#E63946', alpha=0.85)
ax3.set_xticks(x3)
ax3.set_xticklabels(categories, fontsize=10)
ax3.set_title('退市股 vs 在市股 个股层对比', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)
for bars in [b1, b2]:
    for b in bars:
        v = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2, v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

# ---- 4. 净值曲线对比 ----
ax4 = axes[1, 1]
years = list(range(2016, 2027))  # 2016..2026 = 11 years
now_nav = [100, 101.7, 125.1, 127.3, 150, 152.3, 172.4, 172.4, 188.2, 209.4, 217.7]
full_nav = [100, 102.2, 124.0, 126.0, 149.1, 151.4, 171.4, 171.4, 186.7, 207.8, 215.7]
ax4.plot(years, now_nav, 'o-', color='#2E86AB', linewidth=2, markersize=5, label='当前样本(幸存者)')
ax4.plot(years, full_nav, 's-', color='#E63946', linewidth=2, markersize=5, label='全样本(含退市股)')
ax4.set_title('净值曲线对比 (起始100万)', fontsize=12, fontweight='bold')
ax4.set_xlabel('年份')
ax4.set_ylabel('净值(万元)')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)
ax4.set_xticks(years[::2])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('survivorship_bias_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
print("图表已保存: survivorship_bias_chart.png")

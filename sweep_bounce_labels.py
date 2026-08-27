# -*- coding: utf-8 -*-
"""
sweep_bounce_labels.py — 超跌反弹: 标签分组扫描(①②)
====================================================

对每个超跌反弹信号打两类标签(均用盘后才可知的信息，无未来函数)，
然后按标签分桶统计净收益，看能否把正边际再细分出高置信子集：

  ① 盘后缩量确认(vol_filter):  信号日(恐慌日)收盘量 = volume[i] / 前8日均量
     —— 量能萎缩说明抛压衰竭而非放量恐慌继杀，属"阴跌型超跌"更值得接。
       分桶看 vol_ratio ∈ (0,0.75] / (0.75,1.0] / (1.0,1.5] / (1.5,2.0] / >2.0
       并测试阈值变体: 只看 ≤0.75 / ≤1.0 / ≤1.5 / ≤2.0 (过滤掉高量比信号)。

  ② 强势股错杀标签(ma_slope):  跌前20日 MA20 斜率
     slope = MA20[信号日-回看天数] - MA20[信号日-回看天数-20]
     —— 上涨趋势中被急跌打下来的是"错杀"抽理回调，与一路阴跌的真龙头区分。
       >0  强势股错杀(up)
       <=0 弱势(弱/横/阴跌, down)

  ③ 联合: 强势+缩量 vs 其余。

基线参数取自 sweep_bounce 最优解(主板 hfq):
  15日跌≥20% 且收阳 → 次日开盘买 | 收盘>MA20离场 | 8%硬止损 | 25根时停 | 剔除ST
  成本与 strategy_ma_cross.py 一致(A股成本模型)。

用法
----
    python sweep_bounce_labels.py                                # 主板 hfq 全量
    python sweep_bounce_labels.py --limit 300                     # 快速试跑前300只
    python sweep_bounce_labels.py --min-n 30                      # 分桶最小样本数
"""

import argparse
import glob
import logging
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd

from strategy_ma_cross import net_return, USECOLS

VOL_USECOLS = list(USECOLS) + ["volume"]   # 只比基类多 volume 一列


def label_bounce_trades(cfg: dict, code: str, df: pd.DataFrame) -> list:
    """单股超跌反弹回测+标签。返回交易记录(含 yield_ratio / ma_slope)。"""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    dd = cfg["down_days"]
    pre = cfg["pre_days"]
    if n < dd + max(cfg["ma_period"], pre) + 20:
        return []

    open_, high, low, close = (df[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
    volume = df["volume"].to_numpy(dtype=float)
    st = df["isST"].to_numpy(dtype=float) if "isST" in df.columns else None
    ma_p = cfg["ma_period"]

    # MA(离场用) 直接调用 pandas 滚动，保持与基类一致
    close_s = df["close"]
    ma = close_s.rolling(ma_p).mean().to_numpy()

    # 超跌: 近 dd 日累计跌幅; 企稳: 当日收阳
    # (注意方向: 与 strategy_bounce 一致, ret_dd = close/close.shift(dd)-1 <= 阈值 即 下跌)
    ret_dd = (close_s / close_s.shift(dd) - 1).to_numpy()
    signal = (ret_dd <= cfg["down_thresh"]) & (close > open_)

    # 标签① 盘后缩量确认: 前8日均量(不含信号日)为分母, 含信号日当天收盘量
    if cfg["vol_n"] > 0:
        vol_ma = (volume[:n - 1])  # 用 rolling 计算前N日均量
    else:
        vol_ma = None

    stop_pct = cfg["stop_pct"]
    time_stop = cfg["time_stop"]
    date = df["date"].to_numpy()
    trades = []
    last_exit = -1
    for i in np.nonzero(signal)[0]:
        if i <= last_exit:
            continue                       # 持仓中跳过
        if st is not None and st[i] == 1 and not cfg["include_st"]:
            continue                       # 跳过 ST
        entry_bar = i + 1
        if entry_bar >= n:
            break
        entry_price = float(open_[entry_bar])

        # ---- 标签计算(均只用 ≤ 信号日收盘的信息) ----
        # ① 量比 = 信号日成交量 / 前vol_n日均量
        if cfg["vol_n"] > 0:
            pf = i - cfg["vol_n"]
            if pf >= 0:
                v0 = volume[pf:i].mean()
            elif i > 0:
                v0 = volume[:i].mean()
            else:
                v0 = np.nan
            vol_ratio = volume[i] / v0 if v0 and not np.isnan(v0) else np.nan
        else:
            vol_ratio = np.nan
        # ② 跌前pre_days 的 MA20 斜率: MA20[i-dd] - MA20[i-dd-pre]
        j_lo = i - dd - pre
        ma_slope = float(ma[i - dd] - ma[j_lo]) if j_lo >= 0 and not np.isnan(ma[i - dd]) else np.nan

        # ---- 出场(与基类逐字节一致): 止损优先 / MA反弹 / 时停 / END ----
        reason = "MA"
        exit_bar = exit_price = None
        for j in range(entry_bar + 1, n):
            o, c = float(open_[j]), float(close[j])
            if o <= entry_price * (1 - stop_pct):
                reason, exit_price, exit_bar = "STOP", o, j
                break
            if low[j] <= entry_price * (1 - stop_pct):
                reason, exit_price, exit_bar = "STOP", entry_price * (1 - stop_pct), j
                break
            if c > ma[j]:
                if j + 1 >= n:
                    exit_bar, exit_price = n - 1, float(close[n - 1])
                else:
                    exit_bar, exit_price = j + 1, float(open_[j + 1])
                break
            if time_stop > 0 and j - entry_bar >= time_stop:
                reason, exit_price, exit_bar = "TIME", c, j
                break
        else:
            reason = "END"
            exit_bar, exit_price = n - 1, float(close[n - 1])

        trades.append({
            "code": code,
            "entry_date": date[entry_bar],
            "entry_price": round(entry_price, 4),
            "exit_date": date[exit_bar],
            "reason": reason,
            "bars": exit_bar - entry_bar + 1,
            "gross_ret": round(exit_price / entry_price - 1, 6),
            "ret": round(net_return(entry_price, exit_price, cfg), 6),
            "vol_ratio": float(vol_ratio) if not np.isnan(vol_ratio) else np.nan,
            "ma_slope": float(ma_slope) if not np.isnan(ma_slope) else np.nan,
            "signal_date": date[i],
        })
        last_exit = exit_bar
    return trades


def process_one(args):
    path, cfg = args
    code = os.path.basename(path).rsplit(".", 1)[0]
    try:
        df = pd.read_csv(path, usecols=VOL_USECOLS)
    except Exception as exc:                   # noqa: BLE001
        logging.warning("%s 读取失败: %s", code, exc)
        return []
    return label_bounce_trades(cfg, code, df)


# ---------------------------------------------------------------------------
# 统计与分桶
# ---------------------------------------------------------------------------
def bucket_stats(trades):
    """trades 接受 DataFrame(含 ret/gross_ret 列)或 dict 列表。"""
    if len(trades) == 0:
        return None
    if isinstance(trades, pd.DataFrame):
        rets = trades["ret"].to_numpy(dtype=float)
        gross = trades["gross_ret"].to_numpy(dtype=float)
    else:
        rets = np.array([t["ret"] for t in trades])
        gross = np.array([t["gross_ret"] for t in trades])
    wins = rets[rets > 0]
    g = wins.sum()
    l = -rets[rets <= 0].sum()
    return {
        "n": len(trades),
        "pf": float(g / l) if l > 0 else float("inf"),
        "win": len(wins) / len(trades),
        "avg_net": float(rets.mean()),
        "avg_gross": float(gross.mean()),
        "cost": float(gross.mean() - rets.mean()),
        "sum": float(rets.sum()),
    }


def print_bucket(title, trades, min_n=30):
    s = bucket_stats(trades)
    if s is None or s["n"] < min_n:
        n = 0 if s is None else s["n"]
        print(f"{title:<44}{n:>6,}  (样本不足)")
        return
    print(f"{title:<44}{s['n']:>6,}  PF {s['pf']:>5.2f}  胜率 {s['win']:>6.1%}  "
          f"均净 {s['avg_net']:>+7.2%}  毛 {s['avg_gross']:>+7.2%}  成本 {s['cost']:.2%}  "
          f"Σ净 {s['sum']:>+8.0%}")


def summarize(df: pd.DataFrame, filters: dict, min_n: int) -> None:
    flt = (df["vol_ratio"].notna() & df["ma_slope"].notna())
    base = df[flt]
    s = bucket_stats(base)
    print("\n" + "=" * 108)
    print(f"基线(全部信号, 需可打上标签): {s['n']:,}笔  PF {s['pf']:.2f}  胜率 {s['win']:.1%}  "
          f"均净 {s['avg_net']:+.2%}  Σ净 {s['sum']:+.0%}")
    print("=" * 108)

    # ---- ① 信号日量比分桶 ----
    print("\n① 信号日量比 = 当日成交量 / 前8日均量 (盘后确认, 次日开盘买)")
    vol_bins = [(0, 0.75, "≤0.75(显著缩量)"), (0.75, 1.0, "0.75~1.0(温和缩量)"),
                (1.0, 1.5, "1.0~1.5(平量)"), (1.5, 2.0, "1.5~2.0(放量)"),
                (2.0, 1e9, ">2.0(显著放量)")]
    for lo, hi, lab in vol_bins:
        print_bucket(f"  量比 {lab}", base[(base["vol_ratio"] > lo) & (base["vol_ratio"] <= hi)], min_n)
    print()
    for thr in (0.75, 1.0, 1.5, 2.0):
        print_bucket(f"  过滤: 只留 量比 ≤ {thr}", base[base["vol_ratio"] <= thr], min_n)
        print_bucket(f"  互补: 量比 >  {thr}", base[base["vol_ratio"] > thr], min_n)

    # ---- ② 跌前 MA20 斜率标签 ----
    print("\n② 跌前20日 MA20 斜率 (>0 强势股错杀 / ≤0 弱势)")
    print_bucket("  强势(斜率>0): 跌前处于上升趋势", base[base["ma_slope"] > 0], min_n)
    print_bucket("  弱势(斜率≤0): 跌前横盘/阴跌", base[base["ma_slope"] <= 0], min_n)

    # ---- ③ 联合 ----
    print("\n③ 联合(强势+缩量 vs 其余)")
    strong = base["ma_slope"] > 0
    shrink = base["vol_ratio"] <= 1.5
    print_bucket("  强势 & 量比≤1.5", base[strong & shrink], min_n)
    print_bucket("  强势 & 量比>1.5", base[strong & ~shrink], min_n)
    print_bucket("  弱势 & 量比≤1.5", base[~strong & shrink], min_n)
    print_bucket("  弱势 & 量比>1.5", base[~strong & ~shrink], min_n)

    # ---- 按出场方式交叉: 强势桶内部 ----
    print("\n④ 强势桶内出场结构")
    strong_df = base[strong]
    for reason in ("MA", "STOP", "TIME"):
        sub = strong_df[strong_df["reason"] == reason]
        print_bucket(f"  强势 & 出场{reason}", sub, min_n // 2)
    print("\n⑤ 弱势桶内出场结构")
    weak_df = base[~strong]
    for reason in ("MA", "STOP", "TIME"):
        sub = weak_df[weak_df["reason"] == reason]
        print_bucket(f"  弱势 & 出场{reason}", sub, min_n // 2)


def main() -> None:
    ap = argparse.ArgumentParser(description="超跌反弹标签分组扫描(缩量确认+强弱标签)")
    ap.add_argument("--data", default="a_share_daily_hfq")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=30, help="分桶最小样本数")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--codes-file", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)

    cfg = {
        # sweep_bounce 最优基线: 15日跌20%收阳 / MA20 / 8%止损 / 25时停
        "down_days": 15, "down_thresh": -0.20, "ma_period": 20,
        "stop_pct": 0.08, "time_stop": 25,
        "include_st": False,
        "vol_n": 8, "pre_days": 20,              # 标签参数
        "usecols": VOL_USECOLS,
        "capital": 10000, "commission": 0.00025,
        "min_commission": 5.0, "stamp": 0.0005, "slip": 0.001, "lot": 100,
    }

    files = sorted(glob.glob(os.path.join(args.data, "*.csv")))
    files = [f for f in files if os.path.basename(f) != "stock_list.csv"]
    if args.codes_file:
        keep = {ln.strip() for ln in open(args.codes_file, encoding="utf-8") if ln.strip()}
        files = [f for f in files if os.path.basename(f).rsplit(".", 1)[0] in keep]
    if args.limit and args.limit < len(files):
        files = files[:args.limit]
    print(f"共 {len(files)} 只(hfq 主板), 并发 {args.workers} 进程...")

    t0 = time.time()
    tasks = [(f, cfg) for f in files]
    all_trades = []
    if args.workers > 1:
        with multiprocessing.Pool(processes=args.workers) as pool:
            for trades in pool.imap_unordered(process_one, tasks, chunksize=16):
                all_trades.extend(trades)
    else:
        for f in files:
            all_trades.extend(process_one((f, cfg)))
    print(f"回测完成 {len(all_trades)} 笔, 用时 {time.time() - t0:.0f}s")

    df = pd.DataFrame(all_trades)
    out = "trades_bounce_labels.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"明细已保存: {out}")

    # 输出示例前3笔(验证标签字段)
    print("\n样例信号:")
    for _, r in df.head(3).iterrows():
        print(f"  {r['code']} 信号{r['signal_date']} 入场{r['entry_date']} "
              f"量比{r['vol_ratio']:.2f} 斜率{r['ma_slope']:+.3f} "
              f"出场{r['reason']} 净{r['ret']:+.2%}")

    summarize(df, {}, args.min_n)


if __name__ == "__main__":
    main()
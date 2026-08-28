# -*- coding: utf-8 -*-
"""
survivorship_bias_analysis.py — 幸存者偏差量化分析
===================================================
1) 从 delisted_codes.txt 读取 2016-2026 期间退市的沪深主板股票
2) 从 baostock 下载这些退市股的后复权日线数据(已预下载则跳过)
3) 在"当前样本 + 退市股"的全样本上重跑恐慌策略
4) 对比"有退市股" vs "无退市股"的回测差异，量化偏差幅度
"""
import os
import re
import sys
import time
import multiprocessing
import numpy as np
import pandas as pd

from strategy_bounce import process_one, USECOLS_VOL
from strategy_ma_cross import USECOLS, net_return
from sim_portfolio import simulate
from strategy_panic import build_files, make_mc_fetcher


DATA_DIR = "a_share_daily_hfq"
DELIST_DIR = "a_share_delisted_hfq"
START_DATE = "2016-01-01"
END_DATE = "2026-08-31"


def make_mc_fetcher_combined(data_dir):
    """市值 fetcher，先查 data_dir，找不到再查 DELIST_DIR。"""
    from strategy_panic import make_mc_fetcher
    mc_main = make_mc_fetcher(data_dir)
    mc_delist = make_mc_fetcher(DELIST_DIR) if os.path.exists(DELIST_DIR) else None
    cache = {}

    def mc_of(code):
        if code in cache:
            return cache[code]
        v = mc_main(code)
        if (v is None or (isinstance(v, float) and np.isnan(v))) and mc_delist is not None:
            v = mc_delist(code)
        cache[code] = v
        return v

    return mc_of


def get_delisted_codes():
    """从 delisted_codes.txt 读取退市股列表(已通过 baostock query_all_stock 差集获取)。"""
    txt_path = os.path.join(os.path.dirname(__file__), "delisted_codes.txt")
    if not os.path.exists(txt_path):
        print("delisted_codes.txt 不存在，请先运行退市股识别脚本")
        return []
    with open(txt_path, "r") as f:
        codes = [line.strip() for line in f if line.strip()]
    print(f"从 delisted_codes.txt 读取: {len(codes)} 只退市股")
    return codes


def download_delisted(codes, out_dir):
    """从 baostock 下载退市股后复权日线数据。"""
    import baostock as bs
    os.makedirs(out_dir, exist_ok=True)
    lg = bs.login()

    downloaded = []
    for i, code in enumerate(codes):
        out_path = os.path.join(out_dir, code + ".csv")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            downloaded.append(code)
            continue

        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume,amount,turn,isST",
            start_date=START_DATE, end_date=END_DATE,
            frequency="d", adjustflag="2"  # 2=后复权
        )
        if rs.error_code != "0":
            continue

        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())

        if len(rows) < 30:  # 数据太少跳过
            continue

        df = pd.DataFrame(rows, columns=rs.fields)
        for col in ["open", "high", "low", "close", "volume", "amount", "turn"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["isST"] = df["isST"].apply(lambda x: 1 if str(x) == "1" else 0)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        downloaded.append(code)

        if (i + 1) % 20 == 0:
            print(f"  退市股下载: [{i+1}/{len(codes)}]")

    bs.logout()
    return downloaded


def run_panic_backtest(data_dir, universe="main", label=""):
    """跑恐慌策略回测，返回结果字典。"""
    cfg = {
        "down_days": 15, "down_thresh": -0.20,
        "ma_period": 20, "stop_pct": 0.08,
        "time_stop": 25, "include_st": False,
        "atr_period": 14,
        "min_atr_drop": 0.0, "max_atr_pct": 0.08,
        "vol_n": 0,
        "usecols": USECOLS,
        "capital": 1_000_000, "commission": 0.00025,
        "min_commission": 5.0, "stamp": 0.0005,
        "slip": 0.001, "lot": 100,
    }

    # 构建文件列表
    if universe == "main":
        files = build_files(data_dir, "main", None)
    elif universe == "main+delisted":
        files = build_files(data_dir, "main", None)
        delist_dir = DELIST_DIR
        if os.path.exists(delist_dir):
            dfiles = sorted([
                os.path.join(delist_dir, f) for f in os.listdir(delist_dir)
                if f.endswith(".csv") and re.match(r"^(sh\.60|sz\.00)", f)
                and not f.startswith("sh.688")
                and not f.startswith("sz.300")
            ])
            files = files + dfiles
    else:
        files = build_files(data_dir, "main", None)

    print(f"\n{'='*62}")
    print(f"[{label}] 板块: {universe} ({len(files)} 只)")
    print(f"{'='*62}")

    t0 = time.time()
    all_trades = []
    tasks = [(f, cfg) for f in files]
    workers = min(8, os.cpu_count() or 1)
    if workers > 1:
        with multiprocessing.Pool(processes=workers) as pool:
            for _code, trades in pool.imap_unordered(process_one, tasks, chunksize=16):
                all_trades.extend(trades)
    else:
        for f in files:
            _code, trades = process_one((f, cfg))
            all_trades.extend(trades)
    print(f"个股回测: {len(all_trades)}笔 信号, 用时{time.time()-t0:.0f}s")

    if not all_trades:
        print("无任何信号")
        return None

    df = pd.DataFrame(all_trades)
    df = df.sort_values(["code", "entry_date"]).reset_index(drop=True)
    cnt = df["entry_date"].value_counts()
    df["density"] = df["entry_date"].map(cnt)

    # 恐慌日过滤
    panic = df[df["density"] >= 60].copy()
    panic_days = int(panic["entry_date"].nunique())

    # 市值过滤 — 退市股需要从退市目录读取市值
    if universe == "main+delisted":
        mc_of = make_mc_fetcher_combined(data_dir)
    else:
        mc_of = make_mc_fetcher(data_dir)
    panic["mc"] = [mc_of(c) for c in panic["code"]]
    panic = panic[(panic["mc"] >= 100) & (panic["mc"] <= 500)]
    if panic.empty:
        print("市值过滤后无信号")
        return None

    print(f"恐慌日过滤: 保留 {len(panic)} 笔 | 恐慌日 {panic_days} 天")

    # 交易层统计
    rets = panic["ret"].to_numpy()
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    pf = (wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    print(f"个股层: {len(panic)}笔  胜率{len(wins)/len(rets):.1%}  "
          f"平均净{rets.mean():+.2%}  PF{pf:.2f}")

    # 统计退市股的交易
    delisted_in_trades = panic[panic["code"].str.match(r"^(sh\.60|sz\.00)")
        & ~panic["code"].str.startswith("sh.688")
        & ~panic["code"].str.startswith("sz.300")]
    # 用退市目录里的文件名判断
    delist_codes = set()
    if os.path.exists(DELIST_DIR):
        delist_codes = {f.replace(".csv", "") for f in os.listdir(DELIST_DIR) if f.endswith(".csv")}
    delisted_trades = panic[panic["code"].isin(delist_codes)]
    if len(delisted_trades) > 0:
        d_rets = delisted_trades["ret"].to_numpy()
        d_wins = d_rets[d_rets > 0]
        d_pf = (d_wins.sum() / -d_rets[d_rets <= 0].sum()) if d_rets[d_rets <= 0].sum() < 0 else float("inf")
        print(f"  其中退市股: {len(delisted_trades)}笔  胜率{len(d_wins)/len(d_rets):.1%}  "
              f"平均净{d_rets.mean():+.2%}  PF{d_pf:.2f}")
    else:
        print(f"  退市股无信号")

    # 组合模拟 — 全样本时需要合并退市股数据目录
    panic["priority"] = panic["mc"]
    if universe == "main+delisted" and os.path.exists(DELIST_DIR):
        # 创建合并目录：把退市股数据硬链接到主目录副本
        merged_dir = os.path.join(os.path.dirname(data_dir), "_merged_hfq_temp")
        os.makedirs(merged_dir, exist_ok=True)
        # 先复制主目录文件清单（用硬链接，不占额外空间）
        import shutil
        for f in os.listdir(data_dir):
            src = os.path.join(data_dir, f)
            dst = os.path.join(merged_dir, f)
            if os.path.isfile(src) and not os.path.exists(dst):
                try:
                    os.link(src, dst)
                except Exception:
                    shutil.copy2(src, dst)
        # 再链接退市股数据
        for f in os.listdir(DELIST_DIR):
            if f.endswith(".csv"):
                src = os.path.join(DELIST_DIR, f)
                dst = os.path.join(merged_dir, f)
                if not os.path.exists(dst):
                    try:
                        os.link(src, dst)
                    except Exception:
                        shutil.copy2(src, dst)
        sim_dir = merged_dir
    else:
        sim_dir = data_dir

    res = simulate(panic, sim_dir, 1_000_000, 20_000,
                   0.00025, 5.0, 0.0005, 0.001, priority_asc=True)

    print(f"期末净值: {res['nav'][-1]/1e4:.1f}万  总收益: {res['total_ret']:+.1%}  "
          f"CAGR {res['cagr']:+.1%}  回撤 {res['mdd']:.1%}  夏普 {res['sharpe']:.2f}")
    print(f"建仓 {res['funded']} 笔 / 跳过 {res['skipped']} 笔 (命中率 {res['fund_rate']:.1%})")

    # 逐年
    navdf = pd.DataFrame({"date": res["dates"], "nav": res["nav"]})
    navdf["y"] = navdf["date"].astype(str).str[:4]
    print("逐年:")
    for y, g in navdf.groupby("y"):
        r = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1
        md = float(((g["nav"].cummax() - g["nav"]) / g["nav"].cummax()).max())
        print(f"  {y}: 收益{r:+.1%}  回撤{md:.1%}  期末{int(g['nav'].iloc[-1]/1e4)}万")

    return {
        "total_ret": res["total_ret"],
        "cagr": res["cagr"],
        "mdd": res["mdd"],
        "sharpe": res["sharpe"],
        "trades": len(panic),
        "panic_days": panic_days,
        "win_rate": len(wins) / len(rets),
        "pf": pf,
        "fund_rate": res["fund_rate"],
        "nav": res["nav"][-1],
    }


def main():
    # 1) 获取退市股列表
    print("=" * 62)
    print("步骤 1: 获取 2016-2026 期间退市的沪深主板股票")
    print("=" * 62)

    delisted_codes = get_delisted_codes()
    print(f"期间退市的沪深主板股票: {len(delisted_codes)} 只")

    # 检查退市股数据是否已下载到 DELIST_DIR
    existing = []
    missing = []
    for code in delisted_codes:
        path = os.path.join(DELIST_DIR, code + ".csv")
        if os.path.exists(path) and os.path.getsize(path) > 100:
            existing.append(code)
        else:
            missing.append(code)
    print(f"  退市数据目录中已有: {len(existing)} 只")
    print(f"  退市数据目录中缺失: {len(missing)} 只")

    # 2) 下载缺失的退市股数据
    print()
    print("=" * 62)
    print("步骤 2: 下载退市股后复权日线数据")
    print("=" * 62)

    if missing:
        print(f"需要下载 {len(missing)} 只退市股数据...")
        downloaded = download_delisted(missing, DELIST_DIR)
        print(f"成功下载: {len(downloaded)} 只")
    else:
        print("所有退市股数据已存在，跳过下载")

    # 3) 对比回测
    print()
    print("=" * 62)
    print("步骤 3: 回测对比 — 当前样本 vs 全样本(含退市股)")
    print("=" * 62)

    # A) 当前样本(仅 in-market)
    res_now = run_panic_backtest(DATA_DIR, "main", "当前样本(幸存者)")

    # B) 全样本(含退市股)
    res_full = run_panic_backtest(DATA_DIR, "main+delisted", "全样本(含退市股)")

    # 4) 偏差汇总
    print()
    print("=" * 62)
    print("步骤 4: 幸存者偏差量化汇总")
    print("=" * 62)
    if res_now and res_full:
        print(f"{'指标':<12} {'当前样本':>12} {'全样本':>12} {'偏差':>10} {'方向':>6}")
        print("-" * 56)
        for key, label, fmt in [
            ("cagr", "CAGR", "{:+.1%}"),
            ("mdd", "最大回撤", "{:.1%}"),
            ("sharpe", "夏普", "{:.2f}"),
            ("total_ret", "总收益", "{:+.1%}"),
            ("trades", "交易笔数", "{:d}"),
            ("win_rate", "胜率", "{:.1%}"),
            ("pf", "PF", "{:.2f}"),
        ]:
            v_now = res_now[key]
            v_full = res_full[key]
            delta = v_full - v_now
            d_str = fmt.format(delta) if isinstance(delta, float) else str(delta)
            direction = "不利" if (key in ["cagr", "sharpe", "total_ret", "win_rate", "pf"] and delta < 0) or \
                               (key in ["mdd"] and delta > 0) else "有利" if delta != 0 else "中性"
            print(f"{label:<12} {fmt.format(v_now):>12} {fmt.format(v_full):>12} {d_str:>10} {direction:>6}")

        print()
        cagr_bias = res_full["cagr"] - res_now["cagr"]
        if abs(cagr_bias) < 0.005:
            print(f"结论: 幸存者偏差 |ΔCAGR| < 0.5%，策略结论不变，偏差可忽略")
        elif abs(cagr_bias) < 0.02:
            print(f"结论: 幸存者偏差 ΔCAGR = {cagr_bias:+.1%}，有边际影响但不改变正收益结论")
        else:
            print(f"结论: 幸存者偏差 ΔCAGR = {cagr_bias:+.1%}，影响显著，需重新评估策略表现")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
download_cb.py — 使用 AKShare 拉取 A 股可转债历史价值序列 + 基本信息
========================================================================

功能
----
1. 用 ak.bond_zh_cov() 获取全量可转债列表（含退市/在市，约 1050 只），
   生成 cb_list.csv（代码/名称/正股/初始转股价/评级/上市日/到期日）。
2. 逐只调用 ak.bond_zh_cov_value_analysis(symbol=代码) 拉取历史逐日
   【收盘价/纯债价值/转股价值/纯债溢价率/转股溢价率】，每只一个文件 <code>.csv。
   —— 转股溢价率已直接给出，双低值 = 收盘价 + 转股溢价率，无需自建转股价序列。
3. 断点续传：已下载的券自动跳过(--force 强制重下)；单只失败自动重试后跳过，
   不中断整个任务。
4. 多进程并发下载(--workers)：akshare 内部 requests，进程间独立，可加速。
   注意：并发须克制(默认 4)，防止东财/集思录 IP 限流拉黑(参照 baostock 教训)。

依赖
----
    pip install akshare pandas

用法
----
    python download_cb.py                              # 全量, 输出 ./cb_data
    python download_cb.py --start 2021-01-01           # 只保留 2021 以来的行
    python download_cb.py --limit 5 --workers 1        # 测试: 只下前5只
    python download_cb.py --workers 8 --sleep 0.1      # 加速(慎防限流)
"""

import argparse
import csv
import logging
import multiprocessing
import os
import sys
import time
from datetime import date, timedelta

try:
    import akshare as ak
except ImportError:  # pragma: no cover
    sys.stderr.write("[ERROR] 未安装 akshare，请先执行: pip install akshare\n")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    pd = None

DEFAULT_OUTPUT_DIR = "cb_data"
DEFAULT_START = "2021-01-01"   # 用户确认回测区间 2021 至今
MAX_RETRY = 3                  # 单只失败重试次数
MAX_LOGIN_TRIES = 4

# value_analysis 返回列(中文) -> 统一英文列
VA_MAP = {
    "日期": "date",
    "收盘价": "close",
    "纯债价值": "bond_value",
    "转股价值": "convert_value",
    "纯债溢价率": "bond_premium",
    "转股溢价率": "convert_premium",
}
VA_COLS = ["date", "close", "bond_value", "convert_value",
           "bond_premium", "convert_premium"]


def fetch_value_analysis(code: str) -> list:
    """拉取单只转债的历史价值序列，返回 [(date,close,bond_v,conv_v,bond_p,conv_p), ...]。"""
    df = ak.bond_zh_cov_value_analysis(symbol=code)
    if df is None or df.empty:
        return []
    df = df.rename(columns=VA_MAP)
    out = []
    for _, row in df.iterrows():
        rec = [str(row.get("date", "")).split(" ")[0]]
        for col in ("close", "bond_value", "convert_value",
                    "bond_premium", "convert_premium"):
            v = row.get(col)
            rec.append("" if pd.isna(v) else f"{float(v):.6f}")
        out.append(rec)
    return out


def fetch_cb_list() -> list:
    """拉全量转债列表(含退市)，返回 dict list。"""
    df = ak.bond_zh_cov()
    if df is None or df.empty:
        return []
    # bond_zh_cov 列: 债券代码/债券简称/申购日期/.../正股代码/正股简称/正股价/转股价/转股价值/债评级
    colmap = {}
    for c in df.columns:
        if c in ("债券代码", "债券简称", "申购日期", "正股代码", "正股简称",
                 "正股价", "转股价", "转股价值", "债评级", "发行规模"):
            colmap[c] = c
    recs = []
    for _, r in df.iterrows():
        recs.append({c: r.get(c) for c in colmap})
    return recs


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(VA_COLS)
        w.writerows(rows)


def load_one(args, code: str):
    """下载单只转债并写文件。返回状态: new/skip/fail。"""
    path = os.path.join(args.output, code + ".csv")
    if not args.force and os.path.exists(path):
        return "skip"
    for attempt in range(1, MAX_RETRY + 1):
        try:
            rows = fetch_value_analysis(code)
            if rows:
                # 截断到 start 之后(保持列含日期以便回测对齐)
                rows = [r for r in rows if r[0] >= args.start]
                write_csv(path, rows)
                return "new"
            logging.warning("%s 返回空数据", code)
            return "fail"
        except Exception as exc:  # noqa: BLE001
            if attempt < MAX_RETRY:
                time.sleep(1.0 * attempt)
            else:
                logging.error("%s 下载失败: %s", code, exc)
    return "fail"


def _worker(task):
    args, code = task
    return code, load_one(args, code)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AKShare 拉取可转债历史价值序列 + 列表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help="输出目录(默认 cb_data)")
    parser.add_argument("--start", default=DEFAULT_START,
                        help="只保留该日期(含)之后的行情(默认 2021-01-01)")
    parser.add_argument("--workers", type=int, default=4,
                        help="并发进程数(默认4，克制防限流)")
    parser.add_argument("--sleep", type=float, default=0.05,
                        help="每只请求后额外休眠秒数(默认0.05)")
    parser.add_argument("--limit", type=int, default=0,
                        help="只下载前 N 只(测试用, 0=全部)")
    parser.add_argument("--force", action="store_true",
                        help="强制重下已存在文件")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-5s | %(message)s")

    os.makedirs(args.output, exist_ok=True)

    # 1) 转债列表
    logging.info("拉取可转债列表...")
    cb_list = fetch_cb_list()
    if not cb_list:
        logging.error("转债列表为空，终止")
        sys.exit(1)
    logging.info("转债列表 %d 只", len(cb_list))

    # 写 cb_list.csv
    list_path = os.path.join(args.output, "cb_list.csv")
    cols = list(cb_list[0].keys())
    with open(list_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(cb_list)
    logging.info("列表已写入 %s", list_path)

    codes = [r.get("债券代码", "") for r in cb_list]
    codes = [c for c in codes if c]
    if args.limit:
        codes = codes[: args.limit]
    logging.info("待下载 %d 只", len(codes))

    tasks = [(args, c) for c in codes]
    t0 = time.time()
    counters = {"new": 0, "skip": 0, "fail": 0}
    done = 0

    if args.workers > 1:
        with multiprocessing.Pool(args.workers) as pool:
            for code, status in pool.imap_unordered(_worker, tasks):
                counters[status] += 1
                done += 1
                if done % 20 == 0 or done == len(tasks):
                    logging.info("进度 %d/%d | 新增%d 跳过%d 失败%d | %.0fs",
                                 done, len(tasks), counters["new"],
                                 counters["skip"], counters["fail"],
                                 time.time() - t0)
                if args.sleep:
                    time.sleep(args.sleep)
    else:
        for code in codes:
            status = load_one(args, code)
            counters[status] += 1
            done += 1
            if done % 20 == 0 or done == len(tasks):
                logging.info("进度 %d/%d | 新增%d 跳过%d 失败%d | %.0fs",
                             done, len(tasks), counters["new"],
                             counters["skip"], counters["fail"],
                             time.time() - t0)
            if args.sleep:
                time.sleep(args.sleep)

    logging.info("完成：新增 %d，跳过(已有) %d，失败 %d | 用时%.0fs",
                 counters["new"], counters["skip"], counters["fail"],
                 time.time() - t0)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
update_cb.py — 可转债每日增量更新：刷新在市转债列表 + 追加最新价值行
========================================================================

功能
----
1. 刷新 cb_list.csv（重拉 bond_zh_cov，获取最新在市转债及转股价/评级）。
2. 对每个在市转债，用 ak.bond_zh_cov_value_analysis 拉最新一行，追加到 <code>.csv
   （幂等：文件末行日期已是目标日期则跳过）。
3. 只更新"当前在市"转债（按 cb_list 的申购/到期状态判定），已退市券跳过。

用法
----
    python update_cb.py                    # 刷新全部在市转债
    python update_cb.py --data cb_data     # 指定目录
    python update_cb.py --workers 4        # 并发数(默认4)

调度
----
每天收盘后运行一次即可持续追加：
    python D:/量化/AStock/update_cb.py
"""

import argparse
import csv
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import akshare as ak
except ImportError:  # pragma: no cover
    sys.stderr.write("[ERROR] 未安装 akshare，请先执行: pip install akshare\n")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    pd = None

from download_cb import (  # noqa: E402  复用常量与取数函数
    VA_MAP, VA_COLS, fetch_cb_list, fetch_value_analysis,
)

MAX_RETRY = 3


def tail_date(path):
    """读 CSV 最后一行日期(幂等跳过用)。"""
    try:
        with open(path, "rb") as f:
            try:
                f.seek(-256, os.SEEK_END)
            except OSError:
                f.seek(0)
            chunk = f.read()
    except OSError:
        return None
    for line in reversed(chunk.decode("utf-8-sig", errors="ignore").splitlines()):
        line = line.strip()
        if line:
            return line.split(",", 1)[0].strip()
    return None


def append_one(data_dir, code):
    """拉该券最新序列，把新行追加到文件。返回状态: new/ok/skip/fail。"""
    path = os.path.join(data_dir, code + ".csv")
    for attempt in range(1, MAX_RETRY + 1):
        try:
            rows = fetch_value_analysis(code)
            if not rows:
                return "fail"
            rows = [r for r in rows if r[0]]  # 非空日期
            # 取最后一行(最新交易日)
            latest = rows[-1]
            last_date = tail_date(path)
            if last_date and latest[0] <= last_date:
                return "skip"
            if not os.path.exists(path):
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f)
                    w.writerow(VA_COLS)
                    w.writerow(latest)
                return "new"
            with open(path, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(latest)
            return "ok"
        except Exception as exc:  # noqa: BLE001
            if attempt < MAX_RETRY:
                time.sleep(1.0 * attempt)
            else:
                logging.error("%s 更新失败: %s", code, exc)
    return "fail"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="可转债每日增量更新",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", default="cb_data", help="数据目录(默认 cb_data)")
    parser.add_argument("--workers", type=int, default=4,
                        help="并发线程数(默认4)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-5s | %(message)s")

    os.makedirs(args.data, exist_ok=True)

    # 1) 刷新列表
    logging.info("刷新转债列表...")
    cb_list = fetch_cb_list()
    if not cb_list:
        logging.error("转债列表为空，终止")
        sys.exit(1)
    list_path = os.path.join(args.data, "cb_list.csv")
    cols = list(cb_list[0].keys())
    with open(list_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(cb_list)
    logging.info("列表已刷新: %d 只", len(cb_list))

    codes = [r.get("债券代码", "") for r in cb_list if r.get("债券代码")]
    logging.info("待更新 %d 只", len(codes))

    t0 = time.time()
    counters = {"new": 0, "ok": 0, "skip": 0, "fail": 0}
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(append_one, args.data, c): c for c in codes}
            for i, fut in enumerate(as_completed(futs), 1):
                counters[fut.result()] += 1
                if i % 100 == 0:
                    logging.info("进度 %d/%d | 新增%d 追加%d 跳过%d 失败%d | %.0fs",
                                 i, len(codes), counters["new"], counters["ok"],
                                 counters["skip"], counters["fail"], time.time() - t0)
    else:
        for i, c in enumerate(codes, 1):
            counters[append_one(args.data, c)] += 1
            if i % 100 == 0:
                logging.info("进度 %d/%d | 用时%.0fs", i, len(codes), time.time() - t0)

    logging.info("完成：新增文件%d 追加%d 跳过%d 失败%d | 用时%.0fs",
                 counters["new"], counters["ok"], counters["skip"],
                 counters["fail"], time.time() - t0)


if __name__ == "__main__":
    main()

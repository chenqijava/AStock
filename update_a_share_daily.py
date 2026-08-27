# -*- coding: utf-8 -*-
"""
update_a_share_daily.py — 每日增量更新：拉取上一交易日全部 A 股日线，追加到历史文件
===============================================================================

功能
----
1. 用 baostock 的 query_daily_history_k_AStock(day) 一次取回某日全部 A 股日线
   (约 5000+ 条，单次请求)，然后逐只追加到全量下载生成的 <code>.csv / .parquet 文件。
2. 幂等：文件最后一行日期已是目标日期则自动跳过，可安全重复运行。
3. 与 download_a_share_daily.py 共用输出目录与 17 列字段格式(每日接口多出的
   adjustflag 列会被剔除，保证与全量文件列一致)。
4. 写盘阶段多线程并发(--workers)，数据源为单次请求，天然高效。

用法
----
    python update_a_share_daily.py                          # 拉最近已收盘交易日
    python update_a_share_daily.py --day 2026-08-25         # 指定日期
    python update_a_share_daily.py --output data            # 指定目录(须与全量一致)
    python update_a_share_daily.py --workers 8              # 写盘并发数(默认8)

调度建议
--------
每天收盘后(如 16:00)用 Windows 任务计划程序运行一次，即可持续追加：
    python D:/量化/AStock/update_a_share_daily.py --output D:/量化/AStock/a_share_daily
"""

import argparse
import csv
import logging
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

try:
    import baostock as bs
except ImportError:  # pragma: no cover
    sys.stderr.write("[ERROR] 未安装 baostock，请先执行: pip install baostock\n")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    pd = None

from download_a_share_daily import (      # noqa: E402  复用常量与工具函数
    K_FIELDS, SOCKET_TIMEOUT, get_trade_dates, is_a_share, _login_retry,
)

KEEP = K_FIELDS.split(",")                # 目标 17 列


def resolve_day(day: str) -> str:
    """确定要拉取的日期：指定则用之，否则取最近一个已收盘的交易日(严格早于今天)。"""
    if day:
        return day
    today = date.today()
    start = (today - timedelta(days=30)).isoformat()
    dates = get_trade_dates(start, today.isoformat())
    published = [d for d in dates if d < today.isoformat()]
    if not published:
        raise RuntimeError("最近 30 天内没有已收盘的交易日，请用 --day 指定日期")
    return published[-1]


def project_row(fields, row):
    """把每日接口的一行(含 adjustflag 等)映射成 17 列，缺列补空串。"""
    m = dict(zip(fields, row))
    return [m.get(f, "") for f in KEEP]


def tail_date(path):
    """读 CSV 最后一行日期(用于幂等跳过)。只读文件尾部，开销极小。"""
    try:
        with open(path, "rb") as f:
            try:
                f.seek(-512, os.SEEK_END)
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


def append_csv(path, day, row):
    """追加一行到 CSV。返回状态: new/ok/skip/fail。"""
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(KEEP)
            w.writerow(row)
        return "new"
    last = tail_date(path)
    if last and last >= day:              # 已有该日(或更新)数据，跳过
        return "skip"
    orig_size = os.path.getsize(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(row)
    if tail_date(path) != day:            # 写入校验失败(罕见)，回滚到原长度
        try:
            with open(path, "r+b") as f:
                f.truncate(orig_size)
        except OSError:
            pass
        return "fail"
    return "ok"


def append_parquet(path, day, row):
    """追加一行到 Parquet(读改写)。返回状态: new/ok/skip/fail。"""
    if pd is None:
        raise RuntimeError("缺少 pandas，无法输出 parquet")
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise RuntimeError("缺少 pyarrow，无法输出 parquet (pip install pyarrow)")
    rec = dict(zip(KEEP, row))
    new_df = pd.DataFrame([rec])
    for col in new_df.columns:
        if col not in ("date", "code"):
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
    if not os.path.exists(path):
        new_df.to_parquet(path)
        return "new"
    old = pd.read_parquet(path)
    if str(old["date"].iloc[-1]) >= day:
        return "skip"
    tmp = path + ".tmp"
    pd.concat([old, new_df], ignore_index=True).to_parquet(tmp)
    os.replace(tmp, path)
    return "ok"


def do_append(args, day, code, row):
    path = os.path.join(args.output, code + "." + args.format)
    try:
        if args.format == "csv":
            return append_csv(path, day, row)
        return append_parquet(path, day, row)
    except Exception as exc:              # noqa: BLE001
        logging.error("%s 追加失败: %s", code, exc)
        return "fail"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="每日增量拉取上一交易日全部 A 股日线并追加到历史文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("调度建议\n")[1] if "调度建议\n" in __doc__ else "",
    )
    parser.add_argument("--day", help="要拉取的日期 YYYY-MM-DD(默认最近已收盘交易日)")
    parser.add_argument("--output", default="a_share_daily",
                        help="输出目录(默认 a_share_daily，须与全量下载一致)")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv",
                        help="文件格式(默认 csv)")
    parser.add_argument("--workers", type=int, default=8,
                        help="写盘并发线程数(默认8)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-5s | %(message)s")

    # ---- 登录(须在任何查询之前) ----
    socket.setdefaulttimeout(SOCKET_TIMEOUT)
    lg = _login_retry()
    if lg is None or lg.error_code != "0":
        sys.stderr.write("baostock 登录失败(服务器可能暂时不可用)，请稍后重试\n")
        sys.exit(1)
    logging.info("baostock 登录成功")

    try:
        day = resolve_day(args.day)
        logging.info("目标日期: %s", day)

        # 单次请求取回该日全部证券日线
        rs = bs.query_daily_history_k_AStock(day)
        if rs.error_code != "0":
            raise RuntimeError("query_daily_history_k_AStock 失败: %s %s"
                               % (rs.error_code, rs.error_msg))
        fields = list(rs.fields)
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        logging.info("该日共取到 %d 行(全部证券)", len(rows))
        if not rows:
            logging.warning("该日无数据，结束")
            return

        # 按代码分组，仅保留 A 股，并统一成 17 列
        idx_code = fields.index("code")
        by_code = {}
        for row in rows:
            code = row[idx_code]
            if is_a_share(code):
                by_code.setdefault(code, project_row(fields, row))
        logging.info("其中 A 股 %d 只", len(by_code))

        os.makedirs(args.output, exist_ok=True)
        counters = {"new": 0, "ok": 0, "skip": 0, "fail": 0}
        items = list(by_code.items())
        t0 = time.time()
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(do_append, args, day, code, row): code
                        for code, row in items}
                for i, fut in enumerate(as_completed(futs), 1):
                    counters[fut.result()] += 1
                    if i % 1000 == 0:
                        logging.info("已处理 %d/%d 只 | 已用%.0fs",
                                     i, len(items), time.time() - t0)
        else:
            for code, row in items:
                counters[do_append(args, day, code, row)] += 1

        logging.info("完成：新增 %d 只，跳过(已有该日) %d 只，新文件 %d 只，失败 %d 只 | 用时%.0fs",
                     counters["ok"], counters["skip"], counters["new"],
                     counters["fail"], time.time() - t0)
    finally:
        try:
            bs.logout()
        except Exception:                 # noqa: BLE001
            pass


if __name__ == "__main__":
    main()

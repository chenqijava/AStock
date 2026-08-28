# -*- coding: utf-8 -*-
"""
update_a_share_daily_hfq.py — 每日增量更新：为 a_share_daily_hfq(后复权)补最新交易日
=====================================================================================

背景
----
实盘信号用后复权数据与回测口径一致(避免除权缺口被误判成超跌)。但 daily_loop 的
update_a_share_daily.py 只更新不复权目录(a_share_daily)，hfq 目录会逐日落后。
本脚本为 hfq 目录增量补最新交易日数据。

原理
----
baostock 后复权(adjustflag="1")以各股上市首日为基准、历史锚定——同一股票用不同
end_date 拉取，共同历史日期的 hfq 价格完全一致(已验证)。因此只需对每只股票拉
[文件末尾日期, 目标日] 区间，把末尾之后的新行追加进文件即可，历史行无需改动。

与不复权增量(update_a_share_daily.py)的差异
--------------------------------------------
不复权有日级批量接口 query_daily_history_k_AStock，一次拉全市场某天；
后复权没有批量接口，只能逐股 query_history_k_data_plus(code, adjustflag="1")。
故本脚本按股票逐个拉(多进程并发提速)，全市场约 5000 只、每只取最近几日，约几分钟。

用法
----
    python update_a_share_daily_hfq.py                       # 补最近已收盘交易日
    python update_a_share_daily_hfq.py --day 2026-08-27      # 指定日期
    python update_a_share_daily_hfq.py --output a_share_daily_hfq
    python update_a_share_daily_hfq.py --workers 8           # 并发数(默认8)

注意
----
1. 只追加增量，不重写历史。若某只股票源数据被重下导致历史锚定基准变化(极少见)，
   可用 download_a_share_daily.py --adjust 1 整只重下。
2. hfq 目录应与 download_a_share_daily.py --adjust 1 生成的全量一致(17列)。
"""

import argparse
import glob
import logging
import os
import socket
import sys
import time
from datetime import date, timedelta
from multiprocessing import Pool

try:
    import baostock as bs
except ImportError:  # pragma: no cover
    sys.stderr.write("[ERROR] 未安装 baostock，请先执行: pip install baostock\n")
    sys.exit(1)

from download_a_share_daily import (      # noqa: E402  复用常量与工具函数
    K_FIELDS, SOCKET_TIMEOUT, MAX_RETRY, _login_retry, get_trade_dates, is_a_share,
)

KEEP = K_FIELDS.split(",")                # 目标 17 列


def resolve_day(day: str) -> str:
    """确定要补到的日期：指定则用之，否则取最近一个已收盘交易日(严格早于今天)。"""
    if day:
        return day
    today = date.today()
    start = (today - timedelta(days=30)).isoformat()
    dates = get_trade_dates(start, today.isoformat())
    published = [d for d in dates if d < today.isoformat()]
    if not published:
        raise RuntimeError("最近 30 天内没有已收盘的交易日，请用 --day 指定日期")
    return published[-1]


def tail_date(path: str):
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


def fetch_hfq(code: str, start_date: str, end_date: str):
    """拉单只股票 [start, end] 区间的后复权日线(adjustflag="1")，带重试。

    返回 (rows, fields)。rows 每行含 adjustflag 等额外列，由调用方映射成 KEEP。
    """
    last_exc = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            rs = bs.query_history_k_data_plus(
                code, K_FIELDS,
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="1",
            )
            if rs.error_code != "0":
                raise RuntimeError("error_code=%s msg=%s" % (rs.error_code, rs.error_msg))
            rows, fields = [], list(rs.fields)
            while rs.next():
                rows.append(rs.get_row_data())
            return rows, fields
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logging.warning("%s 第%d次请求失败: %s", code, attempt, exc)
            time.sleep(2 * attempt)
            try:
                bs.logout()
            except Exception:  # noqa: BLE001
                pass
            lg = bs.login()
            if lg is None or lg.error_code != "0":
                logging.warning("%s 重登失败，继续重试", code)
    raise RuntimeError("%s 后复权拉取失败(重试%d次): %s" % (code, MAX_RETRY, last_exc))


def project_row(fields, row):
    """把接口返回的一行(含 adjustflag 等)映射成 17 列，缺列补空串。"""
    m = dict(zip(fields, row))
    return [m.get(f, "") for f in KEEP]


# ---------------------------------------------------------------------------
# 多进程 worker —— baostock 非线程安全，每进程独立登录
# ---------------------------------------------------------------------------
_W = None


def _worker_init(output, day):
    global _W
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-5s | %(message)s")
    socket.setdefaulttimeout(SOCKET_TIMEOUT)
    _W = {"output": output, "day": day, "logged_in": False}
    lg = _login_retry()
    if lg is not None and lg.error_code == "0":
        _W["logged_in"] = True


def _worker(code):
    """为单只股票补 hfq 到 day。返回 (code, status, msg)。"""
    w = _W
    if not w["logged_in"]:
        return (code, "fail", "worker 未登录")
    path = os.path.join(w["output"], code + ".csv")
    if not os.path.exists(path):
        return (code, "skip", "无历史文件")
    last = tail_date(path)
    if last is None:
        return (code, "fail", "无法读文件末尾日期")
    if last >= w["day"]:
        return (code, "skip", "已是最新")
    try:
        rows, fields = fetch_hfq(code, last, w["day"])
        keep = [project_row(fields, r) for r in rows]
        added = [r for r in keep if r[0] > last]      # 只取末尾之后的新行
        if not added:
            return (code, "skip", "区间内无新数据")
        # 后复权历史锚定: 末尾之后的新行直接追加到原文件末尾，历史行不动。
        orig_size = os.path.getsize(path)
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            f.seek(max(0, orig_size - 8))             # 检查文件是否以换行结尾
            tail = f.read()
        if tail and not tail.endswith("\n"):
            return (code, "fail", "文件末尾非完整行，请用 download 全量重下")
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            import csv
            for r in added:
                csv.writer(f).writerow(r)
        return (code, "ok", "新增%d行" % len(added))
    except Exception as exc:  # noqa: BLE001
        return (code, "fail", str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="每日为 a_share_daily_hfq(后复权)增量补最新交易日",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    parser.add_argument("--day", help="要补到的日期 YYYY-MM-DD(默认最近已收盘交易日)")
    parser.add_argument("--output", default="a_share_daily_hfq",
                        help="后复权数据目录(默认 a_share_daily_hfq)")
    parser.add_argument("--workers", type=int, default=8,
                        help="并发进程数(默认8，每进程独立登录 baostock)")
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 只(调试用，0=全部)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-5s | %(message)s")

    if not os.path.isdir(args.output):
        sys.stderr.write("目录不存在: %s\n" % args.output)
        sys.exit(1)

    # 先登录(校验连通 & 供 resolve_day 查交易日历)，后面 worker 各自再登
    lg = _login_retry()
    if lg is None or lg.error_code != "0":
        sys.stderr.write("baostock 登录失败，请稍后重试\n")
        sys.exit(1)

    day = resolve_day(args.day)
    logging.info("目标交易日: %s", day)

    codes = []
    for p in glob.glob(os.path.join(args.output, "sh.*.csv")) + \
             glob.glob(os.path.join(args.output, "sz.*.csv")) + \
             glob.glob(os.path.join(args.output, "bj.*.csv")):
        code = os.path.basename(p)[:-4]
        if is_a_share(code):
            codes.append(code)
    codes.sort()
    if args.limit:
        codes = codes[:args.limit]
    if not codes:
        sys.stderr.write("未在 %s 找到 A 股历史文件(应先用 download_a_share_daily.py --adjust 1 生成全量)\n"
                         % args.output)
        sys.exit(1)
    logging.info("待补股票: %d 只", len(codes))

    bs.logout()   # 主进程连接只在上面查交易日历时用，释放后交给 worker

    counters = {"ok": 0, "skip": 0, "fail": 0}
    failed = []
    t0 = time.time()
    n = len(codes)
    with Pool(args.workers, initializer=_worker_init,
              initargs=(args.output, day)) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, codes), 1):
            code, status, msg = res
            counters[status] = counters.get(status, 0) + 1
            if status == "fail":
                failed.append(code)
            if i % 200 == 0 or i == n:
                elapsed = time.time() - t0
                eta = elapsed / i * (n - i) if i else 0
                logging.info("[%d/%d] 新增%d 跳过%d 失败%d | 已用%.0fs ETA %.0fs",
                             i, n, counters.get("ok", 0), counters.get("skip", 0),
                             counters.get("fail", 0), elapsed, eta)

    logging.info("完成：新增 %d 只，跳过(已最新) %d 只，失败 %d 只，共 %d 只，目录 %s",
                 counters.get("ok", 0), counters.get("skip", 0),
                 counters.get("fail", 0), n, os.path.abspath(args.output))
    if failed:
        logging.warning("失败清单(%d): %s", len(failed), ", ".join(failed))


if __name__ == "__main__":
    main()

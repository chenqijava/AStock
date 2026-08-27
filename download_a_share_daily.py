# -*- coding: utf-8 -*-
"""
download_a_share_daily.py — 使用 baostock 拉取 A 股近 N 年全部股票日线数据
========================================================================

功能
----
1. 自动获取 A 股股票列表：沪(主板 60x、科创板 68x) / 深(主板 00x、创业板 30x) / 北交所(bj)。
   默认只包含当前仍上市交易的股票；加 --include-delisted 会按月扫描历史交易日，
   把期间退市、已不交易的股票也一并纳入(更接近“近N年内全部股票”)。
2. 逐只下载日线行情，每只股票一个文件(CSV 或 Parquet)，字段共 17 个：
   date,code,open,high,low,close,preclose,volume,amount,turn,
   tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST
   单位提示(baostock 官方文档)：volume=股，amount=元，turn=换手率%；
   估值字段(peTTM 等)部分交易日无值，返回空字符串。
3. 同时生成 stock_list.csv(code->名称)，方便映射。
4. 断点续传：已成功下载的文件自动跳过(--force 可强制重下)；
   单只失败自动重试后跳过，不会中断整个任务。
5. 请求间隔可调(--sleep)，降低被服务端限流/断连的概率；日志含进度与预计剩余时间。
6. 支持多进程并发下载(--workers N)：baostock 非线程安全，用多进程让每个进程独立
   登录、独立限速，可显著加速全市场回填。

依赖
----
    pip install baostock pandas        # CSV 输出
    pip install pyarrow                # 如需 Parquet 输出

用法
----
    python download_a_share_daily.py                                   # 近10年, 输出 ./a_share_daily
    python download_a_share_daily.py --start 2016-01-01 --end 2026-08-25 \
        --output D:/量化/AStock/data
    python download_a_share_daily.py --format parquet --include-delisted
    python download_a_share_daily.py --limit 5 --sleep 0               # 测试: 只下前5只
    python download_a_share_daily.py --workers 4                       # 4进程并发

说明
----
* 默认不复权(adjustflag=3)；需要前复权/后复权：--adjust 2 / 1。
* 默认 end 取“最近一个已收盘的交易日”(当天盘中的日线尚未发布，避免缺最后一天)；
  收盘后可手动 --end 指定当天日期。
* baostock 非线程安全，并发请用 --workers N(多进程，每进程独立登录)；默认单进程。
  全市场约 5xxx 只，按默认间隔约 1~2 小时，建议放后台运行。
* 每日增量更新：全量回填完成后，用配套脚本 update_a_share_daily.py 每天拉取上一交易日
  数据并追加到各文件(幂等)，无需重复全量下载。
"""

import argparse
import csv
import logging
import multiprocessing
import os
import re
import socket
import sys
import time
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

# ---------------------------------------------------------------------------
# 常量 / 默认值
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "a_share_daily"
DEFAULT_YEARS = 10
DEFAULT_SLEEP = 0.2           # 每次请求间隔(秒)，被限流时可调大
MAX_RETRY = 5                 # 单只失败重试次数
MAX_LOGIN_TRIES = 6           # 登录最大尝试次数(隔段时间自动重试)
SOCKET_TIMEOUT = 30           # 单个 socket 请求超时(秒)，防止服务器挂起导致死等

K_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,turn,"
    "tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
)

# 文件名中含有窗口符号时清理用
UNSAFE = re.compile(r'[\\/:*?"<>|\s]')


def is_a_share(code: str) -> bool:
    """判断代码是否为 A 股股票(排除指数/基金/债券/B股)。

    - sh.600/601/603/605 沪主板, sh.688/689 科创板
    - sz.000/001/002/003 深主板, sz.300/301 创业板
    - bj.43/83/87/92     北交所
    """
    return bool(re.match(r"^(?:sh\.(?:60|68)|sz\.(?:00|30)|bj\.(?:43|83|87|92))", code))


# ---------------------------------------------------------------------------
# baostock 数据获取
# ---------------------------------------------------------------------------
def get_trade_dates(start_date: str, end_date: str) -> list:
    """取 [start, end] 之间的交易日(字符串 'YYYY-MM-DD')。"""
    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    if rs.error_code != "0":
        raise RuntimeError("query_trade_dates 失败: %s %s" % (rs.error_code, rs.error_msg))
    dates = []
    while rs.next():
        row = rs.get_row_data()
        if row[1] == "1":          # is_trading_day
            dates.append(row[0])
    return dates


def query_all_stock(day: str, tries: int = 3) -> list:
    """查询某交易日全部证券列表，失败自动重试返回 [(code, status, name), ...]。"""
    for attempt in range(1, tries + 1):
        try:
            rs = bs.query_all_stock(day)
            if rs.error_code == "0":
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rows or attempt == tries:
                    return rows      # 有数据直接返回；全空也返回(避免死循环)
            elif attempt == tries:
                logging.warning("query_all_stock(%s) 最终失败: %s %s",
                                day, rs.error_code, rs.error_msg)
        except Exception as exc:    # noqa: BLE001
            if attempt == tries:
                logging.warning("query_all_stock(%s) 最终异常: %s", day, exc)
        time.sleep(15 * attempt)
    return []


def get_stock_list(start_date: str, end_date: str, include_delisted: bool,
                   trade_dates: list) -> dict:
    """获取 A 股股票列表 {code: name}。

    include_delisted=False: 取最近一个交易日仍在交易的股票。
    include_delisted=True : 按月(每21个交易日)扫描历史，union 出期间曾出现的股票，
                            从而包含退市股。
    """
    stocks: dict = {}
    if not include_delisted:
        # 注意：不能用 trade_dates[-1]——它可能是"今天"(尚未收盘, 盘中列表为空)。
        # end_date 已收敛到最近一个收盘的交易日。
        rows = query_all_stock(end_date)
        for code, _status, name in rows:
            if is_a_share(code):
                stocks.setdefault(code, name)
        return stocks

    for day in trade_dates[::21]:          # 每月采样一个交易日
        for code, _status, name in query_all_stock(day):
            if is_a_share(code):
                stocks.setdefault(code, name)
        time.sleep(0.1)
    return stocks


def fetch_kline(code: str, start_date: str, end_date: str, adjust: str):
    """下载单只股票日线，失败自动重试(断开后重登)。返回 (rows, fields)。"""
    last_exc = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            rs = bs.query_history_k_data_plus(
                code, K_FIELDS,
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag=adjust,
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
            try:                                  # 长时间运行后连接可能失效，重登
                bs.logout()
            except Exception:                     # noqa: BLE001
                pass
            try:
                lg = bs.login()
            except Exception:                     # noqa: BLE001
                lg = None
            if lg is None or lg.error_code != "0":
                logging.warning("%s 重登失败，继续重试", code)
    raise RuntimeError("%s 下载失败(重试%d次): %s" % (code, MAX_RETRY, last_exc))


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def save_stock(path: str, rows: list, fields: list, fmt: str) -> None:
    """写单只股票文件。CSV 保留原始字符串；Parquet 转数值类型(date/code 除外)。"""
    if fmt == "csv":
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(fields)
            w.writerows(rows)
        return

    # parquet
    if pd is None:
        raise RuntimeError("缺少 pandas，无法输出 parquet")
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise RuntimeError("缺少 pyarrow，无法输出 parquet (pip install pyarrow)")
    df = pd.DataFrame(rows, columns=fields)
    for col in df.columns:
        if col not in ("date", "code"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_parquet(path)


# ---------------------------------------------------------------------------
# 并发下载(multiprocessing) —— baostock 非线程安全，用多进程让每进程独立登录
# ---------------------------------------------------------------------------
_WORKER = None   # 每个进程一份：下载参数与股票表


def _login_retry():
    """带退避重试的 baostock 登录，成功返回结果对象，最终失败返回 None。"""
    lg = None
    for attempt in range(1, MAX_LOGIN_TRIES + 1):
        try:
            lg = bs.login()
        except Exception as exc:                        # noqa: BLE001
            lg = None
            logging.warning("登录异常(第%d次): %s", attempt, exc)
        else:
            if lg.error_code == "0":
                return lg
            logging.warning("登录失败(第%d次): %s %s",
                            attempt, lg.error_code, lg.error_msg)
        wait = min(60, 15 * attempt)                    # 15s, 30s, 45s, 60s...
        logging.info("等待 %ds 后重试登录", wait)
        time.sleep(wait)
    return None


def worker_init(output, fmt, start_date, end_date, adjust, force, sleep, stocks):
    """Pool worker 初始化：每进程独立登录 baostock。"""
    global _WORKER
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-5s | %(message)s")
    socket.setdefaulttimeout(SOCKET_TIMEOUT)
    _WORKER = {
        "output": output, "fmt": fmt,
        "start": start_date, "end": end_date, "adjust": adjust,
        "force": force, "sleep": sleep, "stocks": stocks,
        "logged_in": False,
    }
    lg = _login_retry()
    if lg is not None and lg.error_code == "0":
        _WORKER["logged_in"] = True
        logging.info("worker 登录成功")
    else:
        logging.error("worker 登录失败，将在任务中重试")


def worker_download(code):
    """下载单只股票。返回 (code, status, data, msg)，status ∈ ok/skip/fail。"""
    w = _WORKER
    if not w["logged_in"]:
        try:
            lg = bs.login()
            w["logged_in"] = lg is not None and lg.error_code == "0"
        except Exception:                               # noqa: BLE001
            w["logged_in"] = False
        if not w["logged_in"]:
            return (code, "fail", w["stocks"].get(code, ""), "worker 未登录")

    final = os.path.join(w["output"], code + "." + w["fmt"])
    if os.path.exists(final) and not w["force"]:
        return (code, "skip", "", "")                   # 断点续传：已成功，跳过

    try:
        rows, fields = fetch_kline(code, w["start"], w["end"], w["adjust"])
        tmp = final + ".tmp"
        save_stock(tmp, rows, fields, w["fmt"])
        os.replace(tmp, final)
        time.sleep(w["sleep"])
        return (code, "ok", len(rows), w["stocks"].get(code, ""))
    except Exception as exc:                            # noqa: BLE001
        return (code, "fail", w["stocks"].get(code, ""), str(exc))


def _report_result(i, total, res, counters, failed, t0):
    """记录一只股票的下载结果。i 为已处理序号(乱序也可)，用于估算 ETA。"""
    code, status, data, msg = res
    counters[status] = counters.get(status, 0) + 1
    if status == "ok":
        elapsed = time.time() - t0
        eta = elapsed / i * (total - i) if i else 0
        logging.info("[%d/%d] %s %s 取到%d条 | 已用%.0fs ETA %.0fs",
                     i, total, code, msg, data, elapsed, eta)
    elif status == "fail":
        failed.append(code)
        logging.error("[%d/%d] %s %s 下载失败: %s", i, total, code, msg, data)


def _final_report(counters, failed, total, args):
    logging.info("完成：新增 %d 只，跳过(已存在) %d 只，失败 %d 只，共 %d 只，输出目录 %s",
                 counters.get("ok", 0), counters.get("skip", 0),
                 counters.get("fail", 0), total, os.path.abspath(args.output))
    if failed:
        logging.warning("失败清单(%d): %s", len(failed), ", ".join(failed))


def _run_sequential(args, codes, stocks, start_date, end_date):
    """单进程下载(默认)，行为与原版一致。"""
    global _WORKER
    _WORKER = {
        "output": args.output, "fmt": args.format,
        "start": start_date, "end": end_date, "adjust": args.adjust,
        "force": args.force, "sleep": args.sleep, "stocks": stocks,
        "logged_in": True,                              # 主进程已登录
    }
    total, failed, counters, t0 = len(codes), [], {}, time.time()
    for i, code in enumerate(codes, 1):
        _report_result(i, total, worker_download(code), counters, failed, t0)
    _final_report(counters, failed, total, args)


def _run_pool(args, codes, stocks, start_date, end_date):
    """多进程并发下载。每个 worker 独立登录、独立限速；
    主进程登录在 main() 的 finally 里统一登出。"""
    total, failed, counters, t0 = len(codes), [], {}, time.time()
    initargs = (args.output, args.format, start_date, end_date,
                args.adjust, args.force, args.sleep, stocks)
    with multiprocessing.Pool(processes=args.workers,
                              initializer=worker_init, initargs=initargs) as pool:
        for i, res in enumerate(pool.imap_unordered(worker_download, codes, 8), 1):
            _report_result(i, total, res, counters, failed, t0)
    _final_report(counters, failed, total, args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 baostock 下载 A 股全部股票近 N 年日线数据(每只一个文件)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("说明\n")[1] if "说明\n" in __doc__ else "",
    )
    parser.add_argument("--start", help="起始日期 YYYY-MM-DD(默认 end-10年)")
    parser.add_argument("--end", help="结束日期 YYYY-MM-DD(默认最近已收盘交易日)")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS,
                        help="抓取年数(默认 %d)" % DEFAULT_YEARS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help="输出目录(默认 %s)" % DEFAULT_OUTPUT_DIR)
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv",
                        help="输出格式(默认 csv)")
    parser.add_argument("--include-delisted", action="store_true",
                        help="按月扫描历史，包含期间已退市的股票(更慢)")
    parser.add_argument("--adjust", choices=["1", "2", "3"], default="3",
                        help="复权: 1=后复权 2=前复权 3=不复权(默认3)")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                        help="每次请求间隔秒数(默认 %.1f)" % DEFAULT_SLEEP)
    parser.add_argument("--limit", type=int, default=0,
                        help="只下载前 N 只(调试用，0=全部)")
    parser.add_argument("--force", action="store_true", help="已下载的也重新下载")
    parser.add_argument("--workers", type=int, default=1,
                        help="并发进程数(多进程独立登录；默认1=单进程)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-5s | %(message)s")

    today = date.today().isoformat()

    # ---- 登录(须在任何查询之前，失败自动等待重试) ----
    socket.setdefaulttimeout(SOCKET_TIMEOUT)          # 防止服务器挂起导致死等
    lg = _login_retry()
    if lg is None or lg.error_code != "0":
        sys.stderr.write("baostock 登录失败(服务器可能暂时不可用)，请稍后重试\n")
        sys.exit(1)
    logging.info("baostock 登录成功")

    try:
        _run(args, today, end_raw=date.today().isoformat())
    finally:
        try:
            bs.logout()
        except Exception:                               # noqa: BLE001
            pass


def _run(args: argparse.Namespace, today: str, end_raw: str) -> None:
    """登录成功后的主流程(单独拆出以便 try/finally 统一登出)。"""
    # ---- 确定起止日期 ----
    if args.end:
        end_raw = args.end
    if args.start:
        start_date = args.start
    else:
        start_date = (date.fromisoformat(end_raw)
                      - timedelta(days=365 * args.years)).isoformat()
    if start_date >= end_raw:
        raise ValueError("start(=%s) 必须早于 end(=%s)" % (start_date, end_raw))

    # 结束日期收敛到“最近一个已收盘的交易日”(严格早于今天)
    trade_dates = get_trade_dates(start_date, end_raw)
    published = [d for d in trade_dates if d < today]
    if not published:
        logging.warning("区间内没有找到已收盘的交易日，使用 end 参数原值")
        end_date = end_raw
    else:
        end_date = published[-1]
    logging.info("数据区间: %s ~ %s", start_date, end_date)

    os.makedirs(args.output, exist_ok=True)

    # ---- 股票列表 ----
    stocks = get_stock_list(start_date, end_date, args.include_delisted, trade_dates)
    if not stocks:
        logging.error("未获取到任何 A 股，请检查网络或参数")
        return
    logging.info("共 %d 只 A 股", len(stocks))

    with open(os.path.join(args.output, "stock_list.csv"),
              "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["code", "name"])
        w.writerows(sorted(stocks.items(), key=lambda kv: kv[0]))

    codes = sorted(stocks)
    if args.limit and args.limit < len(codes):
        codes = codes[:args.limit]

    # ---- 逐只下载(单进程或并发) ----
    if args.workers and args.workers > 1:
        _run_pool(args, codes, stocks, start_date, end_date)
    else:
        _run_sequential(args, codes, stocks, start_date, end_date)


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
download_dividends.py — 用 baostock 拉取 A 股分红派息数据
========================================================

输出：dividend.csv（code, ex_date, cash_ps 每股税前现金股利，元/股）
用于高股息选股：股息率(TTM) = 近12个月除权除息日落在窗口内的每股股利之和 / 现价。

要点
----
- baostock 的 query_dividend_data 需按 (code, year) 逐次查询，year='' 只返回最近一笔，
  因此按年循环：--start-year ~ --end-year（回测窗口 2016-2026 + 12月回溯 → 2015 起）。
- ex_date 优先取除权除息日(dividOperateDate)，缺失则退到派息日(dividPayDate)；
  只用 dividCashPsBeforeTax>0 的现金分红（跳过纯送转）。
- 断点续传：已完成的 code 记录在 dividend_progress.txt，重跑自动跳过(--force 重下)。
- 多进程并发：baostock 非线程安全，用多进程让每进程独立登录、独立限速。
- 与日线下载同样的幸存者局限：默认只覆盖当前仍上市的主板股票（与回测口径一致）。
- 代理：baostock 用裸 TCP 连 public-api.baostock.com:10030，直连常被运营商掐线
  (报错 10002007=网络接收错误)。可走 --proxy http://127.0.0.1:10808，实现 HTTP
  CONNECT 隧道(monkeypatch SocketUtil.connect)，所有连接/重连都走代理。

用法
----
    python download_dividends.py                     # 主板 2015~今年, 输出 dividend.csv
    python download_dividends.py --proxy http://127.0.0.1:10808
    python download_dividends.py --universe all      # 全部板块
    python download_dividends.py --codes-file codes_main_board.txt
    python download_dividends.py --workers 8 --sleep 0.05
    python download_dividends.py --limit 20          # 测试前20只
"""

import argparse
import csv
import glob
import logging
import multiprocessing
import os
import re
import socket
import sys
import time
from datetime import date

try:
    import baostock as bs
except ImportError:  # pragma: no cover
    sys.stderr.write("[ERROR] 未安装 baostock，请先执行: pip install baostock\n")
    sys.exit(1)

DEFAULT_SLEEP = 0.05          # 每次请求间隔(秒)
SOCKET_TIMEOUT = 30
MAX_RETRY = 5
MAX_LOGIN_TRIES = 6

# 分红字段固定索引(以 rs.fields 为准，这里是备查)：dividOperateDate / dividPayDate / cashPsBeforeTax
FIELD_EX = "dividOperateDate"
FIELD_PAY = "dividPayDate"
FIELD_CASH = "dividCashPsBeforeTax"


def login_retry():
    """带退避重试的 baostock 登录，最终失败返回 None。"""
    for attempt in range(1, MAX_LOGIN_TRIES + 1):
        try:
            lg = bs.login()
        except Exception:                               # noqa: BLE001
            lg = None
        else:
            if lg is not None and lg.error_code == "0":
                return lg
        wait = min(60, 15 * attempt)
        logging.warning("登录失败(%s)，%ds 后重试", lg.error_code if lg else "异常", wait)
        time.sleep(wait)
    return None


def fetch_dividends(code: str, start_year: int, end_year: int) -> list:
    """拉单只股票 start_year~end_year 的现金分红，返回 [(ex_date, cash_ps), ...]。
    失败自动重试(断开后重登)。"""
    last_exc = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            rows = []
            for y in range(start_year, end_year + 1):
                rs = bs.query_dividend_data(code, year=str(y), yearType="operate")
                if rs.error_code != "0":
                    raise RuntimeError("error_code=%s msg=%s" % (rs.error_code, rs.error_msg))
                fields = list(rs.fields)
                i_ex, i_pay, i_cash = (fields.index(FIELD_EX), fields.index(FIELD_PAY),
                                       fields.index(FIELD_CASH))
                while rs.next():
                    r = rs.get_row_data()
                    try:
                        cash = float(r[i_cash])
                    except (TypeError, ValueError):
                        continue
                    if cash <= 0:
                        continue                     # 纯送转/无现金分红
                    ex = (r[i_ex] or r[i_pay]).strip()   # 除权除息日优先，缺则派息日
                    if not ex:
                        continue
                    rows.append((ex, cash))
            return rows
        except Exception as exc:                        # noqa: BLE001
            last_exc = exc
            logging.warning("%s 第%d次失败: %s", code, attempt, exc)
            time.sleep(2 * attempt)
            try:
                bs.logout()
            except Exception:                           # noqa: BLE001
                pass
            lg = login_retry()
            if lg is None:
                logging.warning("%s 重登失败", code)
    raise RuntimeError("%s 下载失败(重试%d次): %s" % (code, MAX_RETRY, last_exc))


_WORKER = None


def worker_init(start_year, end_year, sleep, proxy=""):
    global _WORKER
    socket.setdefaulttimeout(SOCKET_TIMEOUT)
    patch_socketutil(proxy)
    _WORKER = {"start_year": start_year, "end_year": end_year, "sleep": sleep}
    lg = login_retry()
    _WORKER["logged_in"] = lg is not None and lg.error_code == "0"


def worker_download(code):
    w = _WORKER
    if not w["logged_in"]:
        lg = login_retry()
        w["logged_in"] = lg is not None and lg.error_code == "0"
        if not w["logged_in"]:
            return (code, "fail", [], "worker 未登录")
    try:
        rows = fetch_dividends(code, w["start_year"], w["end_year"])
        time.sleep(w["sleep"])
        return (code, "ok", rows, "")
    except Exception as exc:                            # noqa: BLE001
        return (code, "fail", [], str(exc))


# ---------------------------------------------------------------------------
# 代理(HTTP CONNECT 隧道)：baostock 用裸 TCP 连 public-api.baostock.com:10030，
# 直连易被掐线。monkeypatch SocketUtil.connect 让连接/重连都走代理。
# ---------------------------------------------------------------------------
def _parse_proxy(proxy: str):
    """解析 http://host:port → (host, port)。"""
    if "://" in proxy:
        proxy = proxy.split("://", 1)[1]
    host, _, port = proxy.partition(":")
    return host, int(port or 10808)


def proxy_connect(host, port, proxy_url, timeout=None):
    """经 HTTP CONNECT 代理建立到 (host, port) 的 TCP 隧道，返回已连接的 socket。"""
    ph, pp = _parse_proxy(proxy_url)
    s = socket.create_connection((ph, pp), timeout=timeout)
    req = "CONNECT %s:%d HTTP/1.1\r\nHost: %s:%d\r\n\r\n" % (host, port, host, port)
    s.sendall(req.encode("utf-8"))
    resp = b""
    while b"\r\n\r\n" not in resp:
        chunk = s.recv(4096)
        if not chunk:
            raise OSError("代理未响应 CONNECT")
        resp += chunk
    status_line = resp.split(b"\r\n", 1)[0]
    if b" 200 " not in status_line:
        s.close()
        raise OSError("CONNECT 失败: %s" % status_line.decode("latin-1"))
    return s


def patch_socketutil(proxy_url: str):
    """把 baostock 的 SocketUtil.connect 改为经代理隧道连接(全局，进程内一次)。"""
    if not proxy_url:
        return
    import baostock.util.socketutil as su
    import baostock.common.contants as cons
    import baostock.common.context as context

    def patched_connect(self):
        try:
            s = proxy_connect(cons.BAOSTOCK_SERVER_IP, cons.BAOSTOCK_SERVER_PORT,
                              proxy_url, timeout=socket.getdefaulttimeout())
        except Exception:                       # noqa: BLE001
            print("服务器连接失败，请稍后再试。")
            s = None
        setattr(context, "default_socket", s)

    su.SocketUtil.connect = patched_connect
    logging.info("已启用代理 %s (HTTP CONNECT 隧道)", proxy_url)


def build_codes(data_dir: str, universe: str, codes_file: str) -> list:
    """按板块/名单枚举股票代码(与策略口径一致，避免下载了用不上的板块)。"""
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    files = [f for f in files if os.path.basename(f) != "stock_list.csv"]
    if codes_file:
        keep = {ln.strip() for ln in open(codes_file, encoding="utf-8") if ln.strip()}
        files = [f for f in files if os.path.basename(f).rsplit(".", 1)[0] in keep]
    elif universe == "main":
        files = [f for f in files if re.match(r"^(sh\.60|sz\.00)", os.path.basename(f))]
    return [os.path.basename(f).rsplit(".", 1)[0] for f in files]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="用 baostock 下载 A 股分红派息数据(每股税前现金股利+除权除息日)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    ap.add_argument("--data", default="a_share_daily_hfq", help="用其文件名枚举股票代码")
    ap.add_argument("--universe", default="main", choices=["main", "all"])
    ap.add_argument("--codes-file", default=None)
    ap.add_argument("--start-year", type=int, default=2015, help="分红起始年份(默认2015)")
    ap.add_argument("--end-year", type=int, default=date.today().year,
                    help="分红结束年份(默认今年)")
    ap.add_argument("--output", default="dividend.csv")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    ap.add_argument("--limit", type=int, default=0, help="只下载前N只(调试)")
    ap.add_argument("--force", action="store_true", help="已完成的也重下")
    ap.add_argument("--proxy", default=os.environ.get("BAOSTOCK_PROXY", ""),
                    help="HTTP 代理(如 http://127.0.0.1:10808), 经 CONNECT 隧道连 baostock")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-5s | %(message)s")
    if args.start_year > args.end_year:
        sys.stderr.write("start-year 必须 <= end-year\n")
        sys.exit(1)

    socket.setdefaulttimeout(SOCKET_TIMEOUT)
    patch_socketutil(args.proxy)
    lg = login_retry()
    if lg is None:
        sys.stderr.write("baostock 登录失败，请稍后重试\n")
        sys.exit(1)
    logging.info("baostock 登录成功")

    try:
        codes = build_codes(args.data, args.universe, args.codes_file)
        if args.limit and args.limit < len(codes):
            codes = codes[:args.limit]
        if not codes:
            sys.stderr.write("未找到股票代码\n")
            sys.exit(1)

        prog_file = os.path.splitext(args.output)[0] + "_progress.txt"
        done = set()
        if os.path.exists(prog_file) and not args.force:
            done = {ln.strip() for ln in open(prog_file, encoding="utf-8") if ln.strip()}
        todo = [c for c in codes if c not in done]
        logging.info("共 %d 只，已完成 %d，本次下载 %d 只 (%s~%s 年)",
                     len(codes), len(done), len(todo), args.start_year, args.end_year)
        if not todo:
            logging.info("全部完成，无需下载")
            return

        t0 = time.time()
        n_ok = n_fail = 0
        failed = []
        with open(args.output, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if not os.path.exists(args.output) or os.path.getsize(args.output) == 0:
                w.writerow(["code", "ex_date", "cash_ps"])
            with multiprocessing.Pool(processes=args.workers,
                                      initializer=worker_init,
                                      initargs=(args.start_year, args.end_year,
                                                args.sleep, args.proxy)) as pool:
                for i, (code, status, rows, msg) in enumerate(
                        pool.imap_unordered(worker_download, todo, 8), 1):
                    if status == "ok":
                        for ex, cash in rows:
                            w.writerow([code, ex, cash])
                        with open(prog_file, "a", encoding="utf-8") as pf:
                            pf.write(code + "\n")
                        n_ok += 1
                        elapsed = time.time() - t0
                        eta = elapsed / i * (len(todo) - i) if i else 0
                        logging.info("[%d/%d] %s %d笔分红 | 已用%.0fs ETA%.0fs",
                                     i, len(todo), code, len(rows), elapsed, eta)
                    else:
                        n_fail += 1
                        failed.append(code)
                        logging.error("[%d/%d] %s 失败: %s", i, len(todo), code, msg)

        logging.info("完成: 成功 %d, 失败 %d (共 %d)", n_ok, n_fail, len(todo))
        if failed:
            logging.warning("失败清单(%d): %s", len(failed), ", ".join(failed[:30]))

        # 最终整理: 排序 + 去重
        if os.path.exists(args.output):
            rows = {}
            with open(args.output, encoding="utf-8-sig") as f:
                rd = csv.reader(f)
                header = next(rd)
                for r in rd:
                    if len(r) >= 3:
                        rows[(r[0], r[1])] = r[2]
            tmp = args.output + ".tmp"
            with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(header)
                for (code, ex), cash in sorted(rows.items()):
                    w.writerow([code, ex, cash])
            os.replace(tmp, args.output)
            logging.info("已整理 %s: %d 条分红记录", args.output, len(rows))
    finally:
        try:
            bs.logout()
        except Exception:                               # noqa: BLE001
            pass


if __name__ == "__main__":
    main()

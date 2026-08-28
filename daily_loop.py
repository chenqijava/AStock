# -*- coding: utf-8 -*-
"""
daily_loop.py — 实盘每日定时循环: 每天 16:00 自动更新数据并推送 TG 信号
======================================================================

到点(默认 16:00)依次执行:
  1. 更新数据   update_a_share_daily.py --day <今天> --output a_share_daily
                (今天数据若尚未发布, 每 --retry-interval 重试, 最迟 --retry-until)
  2. 验证今天数据已入库(以 sh.600000.csv 末尾日期为准)
  3. 生成信号并推送   signal_panic.py --tg    (完整报告推到 TG 群)
  4. 记录"今日已完成", 睡到明天

非交易日自动跳过(不发消息)。已完成日期存 daily_loop_state.json,
重启不重复执行。适合开机自启 / Windows 任务计划长时间挂着。

用法
----
    python daily_loop.py                    # 长期运行(建议开机自启)
    python daily_loop.py --once             # 立即跑一次今天(手动/测试)
    python daily_loop.py --time 16:00       # 改定时(默认 16:00)
    python daily_loop.py --retry-until 20:00
    python daily_loop.py --once --no-tg     # 只更新+生成信号, 不推送
"""
import argparse
import datetime as _dt
import json
import logging
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = "daily_loop_state.json"
REF_CODE = "sh.600000"          # 交易日参考股(每个交易日必有数据)


def parse_hhmm(s: str) -> _dt.time:
    h, m = s.split(":", 1)
    return _dt.time(int(h), int(m))


def load_state(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(path: str, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def is_trading_day(day: str) -> bool:
    """baostock 查今天是否为交易日。查询失败时返回 None(不阻塞, 走更新流程自行判断)。"""
    try:
        from download_a_share_daily import get_trade_dates
        import baostock as bs
        socket.setdefaulttimeout(30)
        lg = bs.login()
        if lg is None or lg.error_code != "0":
            logging.warning("baostock 登录失败, 无法确认交易日")
            return None
        try:
            return day in get_trade_dates(day, day)
        finally:
            try:
                bs.logout()
            except Exception:                 # noqa: BLE001
                pass
    except Exception as exc:                  # noqa: BLE001
        logging.warning("查询交易日失败: %s", exc)
        return None


def _run_step(name: str, cmd: list, cwd: str) -> int:
    """跑一个子进程, 实时透传 stdout/stderr 到日志(带 [name] 前缀), 返回退出码。"""
    logging.info("[%s] 开始: %s", name, " ".join(cmd))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            bufsize=1)
    for line in proc.stdout:                       # 实时逐行透传
        logging.info("[%s] %s", name, line.rstrip())
    proc.wait()
    logging.info("[%s] 结束, 退出码 %d", name, proc.returncode)
    return proc.returncode


def run_update(day: str, output: str) -> int:
    """调更新脚本拉取指定日期, 返回子进程退出码。"""
    cmd = [sys.executable, os.path.join(HERE, "update_a_share_daily.py"),
           "--day", day, "--output", output]
    return _run_step("更新", cmd, HERE)


def today_data_present(output: str, day: str) -> bool:
    """参考股文件末尾日期 == 今天, 则今天数据已入库。"""
    path = os.path.join(HERE, output, REF_CODE + ".csv")
    try:
        with open(path, "rb") as f:
            f.seek(-256, os.SEEK_END)
            tail = f.read().decode("utf-8-sig", errors="ignore")
        for line in reversed(tail.splitlines()):
            line = line.strip()
            if line:
                return line.split(",", 1)[0].strip() == day
    except OSError:
        return False
    return False


def run_signal(data: str, tg_config: str, do_tg: bool) -> int:
    """生成信号, --tg 时把报告推到 TG 群。"""
    cmd = [sys.executable, os.path.join(HERE, "signal_panic.py"),
           "--data", data, "--tg-config", tg_config]
    if do_tg:
        cmd.append("--tg")
    return _run_step("信号", cmd, HERE)


def push_notice(tg_config: str, text: str) -> None:
    """发一条纯文本通知到 TG 群(更新失败/数据未发布等)。"""
    try:
        sys.path.insert(0, HERE)
        import tg_notify
        cfg = tg_notify.load_config(os.path.join(HERE, tg_config))
        ok = tg_notify.send(text, cfg, parse_mode="")
        logging.info("TG 通知%s: %s", "成功" if ok else "失败", text)
    except Exception as exc:                  # noqa: BLE001
        logging.error("TG 通知失败: %s", exc)


def do_daily_job(args, day: str) -> str:
    """跑一天的完整流程。返回状态: ok / pending(数据未发布, 稍后重试) /
    error(更新失败) / notrade(非交易日)。"""
    t0 = time.time()
    logging.info("=" * 60)
    logging.info("开始执行 %s 当日流程", day)

    if args.check_trade:
        tr = is_trading_day(day)
        if tr is False:
            logging.info("%s 非交易日, 跳过 (耗时 %.1fs)", day, time.time() - t0)
            return "notrade"
        if tr is None:
            logging.info("无法确认交易日, 按交易日流程处理")
        else:
            logging.info("交易日确认: 是 (耗时 %.1fs)", time.time() - t0)

    t1 = time.time()
    rc = run_update(day, args.output)
    t_update = time.time() - t1
    if rc != 0:
        logging.error("更新失败, 退出码 %d (耗时 %.1fs)", rc, t_update)
        return "error"

    if not today_data_present(args.output, day):
        logging.info("今天数据尚未发布(参考股 %s 未到 %s), 稍后重试 (更新耗时 %.1fs)",
                     REF_CODE, day, t_update)
        return "pending"
    logging.info("数据已入库: %s 末行=%s (更新耗时 %.1fs)", REF_CODE, day, t_update)

    t2 = time.time()
    rc = run_signal(args.data, args.tg_config, args.tg)
    t_signal = time.time() - t2
    if rc != 0:
        logging.error("信号失败, 退出码 %d (耗时 %.1fs)", rc, t_signal)
        return "error"
    logging.info("当日流程完成: 更新 %.1fs + 信号 %.1fs = 总计 %.1fs",
                 t_update, t_signal, time.time() - t0)
    return "ok"


def mark_done(args, day: str, note: str) -> None:
    state = load_state(args.state)
    state["done"] = day
    state["at"] = _dt.datetime.now().isoformat(timespec="seconds")
    state["note"] = note
    save_state(args.state, state)
    logging.info("今日流程完成(%s), 状态已记录", note)


def loop(args) -> None:
    state = load_state(args.state)
    while True:
        now = _dt.datetime.now()
        today = now.date().isoformat()

        if state.get("done") == today and not args.force:
            if args.once:
                logging.info("今天已完成(%s), 退出(--force 可强制重跑)", state.get("note", ""))
                return
            # 今天已做完, 睡到明天触发
            target = _dt.datetime.combine(now.date() + _dt.timedelta(days=1),
                                          args.trigger)
            wait = (target - now).total_seconds()
            logging.info("今日已完成, 睡到明天 %s (距 %d 分钟)",
                         target.strftime("%Y-%m-%d %H:%M"), wait // 60)
            time.sleep(min(wait, 3600))
            continue

        # --once: 立即执行, 不等触发点; 常驻: 等到触发点
        if args.once or now.time() >= args.trigger:
            day = today
            result = do_daily_job(args, day)
            if result == "ok":
                mark_done(args, day, "ok")
                if args.once:
                    return
            elif result in ("pending", "error"):
                if args.once or now.time() > args.retry_until:
                    # --once 模式不重试, 直接报完退出; 常驻模式超重试截止则报
                    if args.tg:
                        push_notice(args.tg_config,
                                    "⚠️ 今日流程未完成(%s): %s 数据未入库或更新失败, 请手动处理"
                                    % (day, "当日数据未发布" if result == "pending" else "更新失败"))
                    else:
                        logging.info("未推送(--no-tg): 今日流程未完成(%s, %s)",
                                     day, result)
                    mark_done(args, day, result)
                    if args.once:
                        return
                else:
                    logging.info("%d 秒后重试(数据未发布或更新失败, 截止 %s)",
                                 args.retry_interval, args.retry_until)
                    time.sleep(args.retry_interval)
                    continue
            else:                             # notrade
                mark_done(args, day, "notrade")
                if args.once:
                    return
        else:
            # 常驻模式未到触发点, 短睡等待
            target = _dt.datetime.combine(now.date(), args.trigger)
            wait = (target - now).total_seconds()
            logging.info("等待触发点 %s (还有 %d 分钟)", args.time, wait // 60)
            time.sleep(min(wait, 3600))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="实盘每日定时循环: 每天16:00自动更新数据并推送TG信号",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    ap.add_argument("--once", action="store_true",
                    help="立即跑一次今天的流程后退出(手动/测试)")
    ap.add_argument("--force", action="store_true",
                    help="忽略'今日已完成'状态, 强制重跑(当天数据重更新后重推用)")
    ap.add_argument("--time", default="16:00", help="每天触发时间 HH:MM(默认16:00)")
    ap.add_argument("--retry-until", default="20:00",
                    help="数据未发布时的重试截止时间 HH:MM(默认20:00)")
    ap.add_argument("--retry-interval", type=int, default=300,
                    help="数据未发布重试间隔秒(默认300)")
    ap.add_argument("--check-interval", type=int, default=30,
                    help="循环检查间隔秒(默认30)")
    ap.add_argument("--output", default="a_share_daily", help="数据目录(默认a_share_daily)")
    ap.add_argument("--data", default="a_share_daily",
                    help="信号用数据目录, 透传给 signal_panic.py(默认同output)")
    ap.add_argument("--tg-config", default="tg_config.json", help="TG配置文件")
    ap.add_argument("--state", default=STATE_FILE, help="状态文件")
    ap.add_argument("--no-tg", action="store_true", help="不推送TG(只更新+生成信号)")
    ap.add_argument("--no-trade-check", action="store_true",
                    help="不查交易日历, 无条件按交易日处理(周末会反复重试)")
    args = ap.parse_args()

    args.tg = not args.no_tg
    args.check_trade = not args.no_trade_check
    args.trigger = parse_hhmm(args.time)
    args.retry_until = parse_hhmm(args.retry_until)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(HERE, "daily_loop.log"),
                                encoding="utf-8"),
        ],
    )
    logging.info("daily_loop 启动 | 触发 %s | 重试截止 %s | 数据 %s | TG推送 %s",
                 args.time, args.retry_until, args.data,
                 "开" if args.tg else "关")

    try:
        loop(args)
    except KeyboardInterrupt:
        logging.info("手动中断, 退出")


if __name__ == "__main__":
    main()

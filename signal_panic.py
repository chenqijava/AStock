# -*- coding: utf-8 -*-
"""
signal_panic.py — 恐慌日超跌反弹：实盘交易信号生成器(盘后运行)
================================================================

把回测策略(strategy_panic.py)变成"明天开盘做什么"的实盘信号。
信号规则与回测完全一致，输出明日的 买入清单 + 卖出清单，并维护持仓台账。

每日盘后流程
------------
1. 更新数据:    python update_a_share_daily.py --output a_share_daily
2. 生成信号:    python signal_panic.py
   -> 输出次日开盘的 买入清单(仅恐慌日) + 卖出清单(持仓出场判定)
3. 次日开盘照单执行
4. 收盘后回填成交(当日执行了什么就填什么):
   python signal_panic.py --buy-fills fills_buy.csv --sell-fills fills_sell.csv
   -> 更新持仓台账 panic_positions.json，再跑一次生成后日信号

信号规则(与 strategy_panic.py 一致)
-----------------------------------
入场(个股层, 三条件同时满足):
  近 down_days(15)日累计跌幅 <= down_thresh(-20%)，当日收阳(close>open)，
  且 ATR(14)/收盘价 <= max_atr_pct(8%)，非ST，沪深主板。
组合层: 当日全市场(主板)触发个股信号的股票数 >= panic_threshold(60) 才是
  恐慌日，才执行买入；孤立超跌是接飞刀(实测净亏损)一律放弃。
恐慌选股: 只在恐慌日信号中选流通市值[min_mc,max_mc]=[100,500]亿，
  同日建仓默认低放量优先(--vol-order low, 回测最优+137.9%/1.31；
  回测结论: 放量越猛胜率越低); --vol-order off 退回 --mc-order small
  小市值优先, high 高放量优先(实验), 大市值优先用 --mc-order large。
入场 = 次日开盘买入。
出场:
  反弹: 收盘 > MA(ma_period=20) -> 次日开盘卖出；
  硬止损: 价格 <= 买入价*(1-8%) -> 止损价/开盘成交(防飞刀续杀)；
  时停: 持有 >= time_stop(25) 根未到位 -> 平仓。

数据
----
信号条件默认用 a_share_daily(未复权，每日可增量更新；回测证实20%跌幅阈值
远超除权缺口量级，未复权对信号影响小)。追求与回测完全一致可用
--data a_share_daily_hfq(后复权)，执行价(止损/股数/涨停提示)自动改用
--real-data(默认同目录下的 a_share_daily)读取真实成交价，信号指标仍按
--data 计算。

持仓台账
--------
panic_positions.json 自动维护: 每笔记录 code/entry_date/entry_price/shares/
stop_price(建仓时按当时 stop_pct 锁定)。出场判定按台账+最新数据计算。

用法
----
    python signal_panic.py                              # 生成次日操作信号
    python signal_panic.py --data a_share_daily_hfq     # 后复权信号(与回测一致)
    python signal_panic.py --list                       # 查看当前持仓与浮盈亏
    python signal_panic.py --buy-fills fb.csv --sell-fills fs.csv  # 回填当日成交
    python signal_panic.py --panic-threshold 50 --capital 2000000
    python signal_panic.py --tg            # 报告推到 Telegram 群(默认只推 Server酱)
    python signal_panic.py --pp           # 加推 PushPlus
    python signal_panic.py --no-sc         # 不推 Server酱

买入成交CSV(fills_buy.csv, 每行): date,code,price[,shares]
   date=成交日(默认最新交易日), price=实际成交价, shares 缺省按每笔 notional 算。
卖出成交CSV(fills_sell.csv, 每行): date,code,price[,reason]
"""
import argparse
import glob
import io
import json
import logging
import multiprocessing
import os
import re
import sys
import time

import numpy as np
import pandas as pd

MAIN_RE = re.compile(r"^(sh\.60|sz\.00)")
SIGNAL_COLS = ["date", "open", "high", "low", "close", "amount", "turn", "isST", "volume"]
VOL_N = 8                      # 量比窗口: 信号日成交量 / 前8日均量(与回测 strategy_bounce 一致)
DEFAULT_LEDGER = "panic_positions.json"


def _safe_write(text: str) -> None:
    """把文本写到 stdout，遇到 GBK 终端编码不了的字符(如 ✓)安全降级，不抛异常。

    Windows 控制台默认 GBK 编码，报告里的特殊符号会触发 UnicodeEncodeError 中断推送流程；
    报告本身会原样推到微信(Server酱/PushPlus 走 UTF-8 不受影响)，这里只是终端显示降级。
    """
    enc = getattr(sys.stdout, "encoding", "") or "utf-8"
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.write(text.encode(enc, errors="replace").decode(enc, errors="replace"))



# ---------------------------------------------------------------------------
# 数据/计算工具
# ---------------------------------------------------------------------------
def build_files(data_dir: str, universe: str, codes_file: str) -> list:
    """按板块/名单构建文件列表(与 strategy_panic 同口径)。"""
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    files = [f for f in files if os.path.basename(f) != "stock_list.csv"]
    if codes_file:
        keep = {ln.strip() for ln in open(codes_file, encoding="utf-8") if ln.strip()}
        files = [f for f in files if os.path.basename(f).rsplit(".", 1)[0] in keep]
    elif universe == "main":
        files = [f for f in files if MAIN_RE.match(os.path.basename(f))]
    return files


def trading_calendar(data_dir: str) -> list:
    """交易日历(以 sh.600000 为基准)。"""
    df = pd.read_csv(os.path.join(data_dir, "sh.600000.csv"), usecols=["date"])
    return [str(d) for d in df["date"].tolist()]


FRAME_CACHE: dict = {}


def load_frame(data_dir: str, code: str) -> pd.DataFrame:
    """读取单只股票 OHLC 数据帧并缓存。失败返回空 DataFrame。"""
    key = (data_dir, code)
    if key not in FRAME_CACHE:
        try:
            df = pd.read_csv(os.path.join(data_dir, code + ".csv"),
                             usecols=["date", "open", "high", "low", "close"])
            df = df.sort_values("date").reset_index(drop=True)
            FRAME_CACHE[key] = df
        except Exception:                               # noqa: BLE001
            FRAME_CACHE[key] = pd.DataFrame()
    return FRAME_CACHE[key]


def scan_one(args):
    """单只股票：判定信号日是否触发个股层入场信号。返回 (code, dict|None)。"""
    path, cfg, signal_date = args
    code = os.path.basename(path).rsplit(".", 1)[0]
    try:
        df = pd.read_csv(path, usecols=SIGNAL_COLS)
    except Exception:                                   # noqa: BLE001
        return code, None
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    need = max(cfg["down_days"], cfg["ma_period"], cfg["atr_period"]) + 5
    if n < need or str(df["date"].iloc[-1]) != signal_date:
        return code, None                       # 数据不足 或 当日停牌/未更新
    if df["isST"].iloc[-1] == 1:
        return code, None
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]

    # 1) 超跌: 近 down_days 日累计跌幅
    ret_n = close.iloc[-1] / close.iloc[-1 - cfg["down_days"]] - 1
    if ret_n > cfg["down_thresh"] or not (close.iloc[-1] > open_.iloc[-1]):
        return code, None                       # 2) 当日收阳企稳

    # 3) ATR% 上限过滤(与 strategy_bounce 同一 EWM 公式)
    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(),
                    (low - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / cfg["atr_period"], adjust=False).mean()
    atr_pct = float(atr.iloc[-1] / close.iloc[-1])
    if atr_pct > cfg["max_atr_pct"]:
        return code, None

    # 流通市值(亿元) = amount*100/turn 近20个有效日中位数(与 filter_by_market_cap 一致)
    valid = (df["turn"] > 0) & (df["amount"] > 0)
    if int(valid.sum()) >= 5:
        mc = float(np.median((df.loc[valid, "amount"] * 100.0
                              / df.loc[valid, "turn"]).tail(20))) / 1e8
    else:
        mc = np.nan

    # 盘后量比 = 信号日成交量 / 前VOL_N日均量(与回测 strategy_bounce 口径一致, 无未来函数)
    vol_n = cfg.get("vol_n", 0)
    vol = pd.to_numeric(df["volume"], errors="coerce").to_numpy()
    vol_ratio = float("nan")
    if vol_n > 0 and n > vol_n and np.all(np.isfinite(vol[-VOL_N:])):
        v0 = float(vol[-1 - VOL_N:-1].mean())
        vol_ratio = float(vol[-1]) / v0 if v0 > 0 else float("nan")

    return code, {"code": code, "close": float(close.iloc[-1]),
                  "ret15d": float(ret_n), "atr_pct": atr_pct, "mc": mc,
                  "vol_ratio": vol_ratio}


def net_round_trip(entry: float, exit_: float, shares: int, cfg: dict) -> float:
    """给定股数的一笔完整往返净收益率(佣金/最低佣金/印花税/滑点)。"""
    buy_value = entry * shares
    sell_value = exit_ * shares
    buy_fee = max(cfg["commission"] * buy_value, cfg["min_commission"]) \
        + cfg["slip"] * buy_value
    sell_fee = max(cfg["commission"] * sell_value, cfg["min_commission"]) \
        + (cfg["stamp"] + cfg["slip"]) * sell_value
    return (sell_value - sell_fee - buy_value - buy_fee) / buy_value


# ---------------------------------------------------------------------------
# 持仓台账
# ---------------------------------------------------------------------------
def load_ledger(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"positions": [], "closed": []}
    data.setdefault("positions", [])
    data.setdefault("closed", [])
    return data


def save_ledger(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_fills(path: str, notional: float, lot: int, default_date: str) -> list:
    """读成交CSV: date,code,price[,shares]。返回行列表。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    rows = []
    for _, r in df.iterrows():
        code = str(r["code"]).strip()
        if not code:
            continue
        price = float(r["price"])
        if "shares" in df.columns and not pd.isna(r["shares"]) \
                and float(r["shares"]) > 0:
            shares = int(r["shares"])
        else:
            shares = int(notional // (price * lot)) * lot
        date = default_date
        if "date" in df.columns and not pd.isna(r["date"]):
            date = str(r["date"]).strip()
        rows.append({"code": code, "price": price, "shares": shares, "date": date})
    return rows


# ---------------------------------------------------------------------------
# 持仓出场判定
# ---------------------------------------------------------------------------
def eval_position(pos: dict, cal_idx: dict, signal_date: str, data_dir: str,
                  real_dir: str, cfg: dict) -> dict:
    """按最新数据判定某持仓的出场动作。

    - MA 反弹用 --data(信号目录，尺度不变，hfq/未复权通用)；
    - 止损比较用真实成交价(--real-data)，因建仓时止损价按真实成交价锁定。
    返回 dict(原字段 + action/cur/bars/note)。
    """
    code = pos["code"]
    df = load_frame(data_dir, code)
    if df is None or len(df) == 0:
        return {**pos, "action": "NO_DATA", "cur": None, "bars": None,
                "note": "无数据"}
    if str(df["date"].iloc[-1]) != signal_date:
        return {**pos, "action": "HOLD", "cur": None, "bars": None,
                "note": "当日停牌/未更新，不可操作"}
    if str(pos.get("entry_date")) == signal_date:
        return {**pos, "action": "HOLD", "cur": float(df["close"].iloc[-1]),
                "bars": 0, "note": "今日刚建仓"}

    bars = cal_idx.get(signal_date, 0) - cal_idx.get(str(pos["entry_date"]), 0)
    close = df["close"]
    ma20 = float(close.rolling(cfg["ma_period"]).mean().iloc[-1])
    c_sig = float(close.iloc[-1])          # --data 收盘(MA 判定用，同口径)
    c, o, l = c_sig, float(df["open"].iloc[-1]), float(df["low"].iloc[-1])
    stop = pos["stop_price"]

    # 止损检查用真实成交价(建仓止损按真实价锁定)；--real-data 与 --data 不同时读取真实帧
    if real_dir != data_dir:
        dfr = load_frame(real_dir, code)
        if len(dfr) > 0 and str(dfr["date"].iloc[-1]) == signal_date:
            o, l = float(dfr["open"].iloc[-1]), float(dfr["low"].iloc[-1])
            c = float(dfr["close"].iloc[-1])          # 显示用真实收盘价

    if o <= stop:                                       # 开盘已破止损
        return {**pos, "action": "SELL_STOP", "cur": c, "bars": bars,
                "note": "开盘已破止损，明日开盘卖出"}
    if l <= stop:                                       # 盘中触及止损
        return {**pos, "action": "SELL_STOP", "cur": c, "bars": bars,
                "note": "盘中已触及止损，明日开盘卖出"}
    if c_sig > ma20:                                    # 反弹到位(MA 用 --data 口径)
        return {**pos, "action": "SELL_MA", "cur": c, "bars": bars,
                "note": "收盘站上MA20，明日开盘卖出"}
    if cfg["time_stop"] > 0 and bars >= cfg["time_stop"]:
        return {**pos, "action": "SELL_TIME", "cur": c, "bars": bars,
                "note": "持有%d根超时，明日开盘卖出" % bars}
    return {**pos, "action": "HOLD", "cur": c, "bars": bars, "note": ""}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="恐慌日超跌反弹——实盘交易信号(盘后生成次日开盘操作清单)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    ap.add_argument("--data", default="a_share_daily",
                    help="信号数据目录(默认 a_share_daily 未复权；回测一致用 a_share_daily_hfq)")
    ap.add_argument("--real-data", default=None,
                    help="真实成交价目录(默认同 --data；--data 为 hfq 时自动用 a_share_daily)")
    ap.add_argument("--positions", default=DEFAULT_LEDGER, help="持仓台账文件")
    ap.add_argument("--list", action="store_true", help="只查看持仓与浮盈亏，不生成信号")
    ap.add_argument("--tg", action="store_true", help="跑完把报告推送到 Telegram 群(tg_config.json)")
    ap.add_argument("--tg-config", default="tg_config.json", help="TG 配置文件(json)")
    ap.add_argument("--pp", dest="pp", action="store_true",
                    help="跑完把报告推送到微信(PushPlus, 默认关; 加 --pp 开)")
    ap.add_argument("--no-pp", dest="pp", action="store_false",
                    help="不推送到微信(PushPlus)")
    ap.add_argument("--pp-config", default="pushplus_config.json", help="PushPlus 配置文件(json)")
    ap.add_argument("--sc", dest="sc", action="store_true", default=True,
                    help="跑完把报告推送到微信(Server酱, 默认开; --no-sc 关)")
    ap.add_argument("--no-sc", dest="sc", action="store_false",
                    help="不推送到微信(Server酱)")
    ap.add_argument("--sc-config", default="serverchan_config.json",
                    help="Server酱 配置文件(json)")
    ap.add_argument("--buy-fills", default=None, help="买入成交CSV(date,code,price[,shares])")
    ap.add_argument("--sell-fills", default=None, help="卖出成交CSV(date,code,price[,reason])")
    ap.add_argument("--universe", default="main", choices=["main", "all"])
    ap.add_argument("--codes-file", default=None)
    # 入场参数
    ap.add_argument("--down-days", type=int, default=15)
    ap.add_argument("--down-thresh", type=float, default=-0.20)
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--max-atr-pct", type=float, default=0.08)
    # 出场参数
    ap.add_argument("--ma-period", type=int, default=20)
    ap.add_argument("--stop-pct", type=float, default=0.08)
    ap.add_argument("--time-stop", type=int, default=25)
    # 恐慌日/市值
    ap.add_argument("--panic-threshold", type=int, default=60)
    ap.add_argument("--min-mc", type=float, default=100.0)
    ap.add_argument("--max-mc", type=float, default=500.0)
    ap.add_argument("--mc-order", default="small", choices=["random", "small", "large"])
    ap.add_argument("--vol-order", default="low", choices=["off", "low", "high"],
                    help="买入排序覆盖为量比: low=低放量优先(默认; 回测最优, 放量越猛胜率越低) | "
                         "high=高放量优先 | off=不用量比(退回市值排序)")
    # 资金/成本(仅用于建议股数与估算收益)
    # 回测定稿: 10万本金 / 每笔2万 / 低放量优先(见回测 trades_panic_v15 10w 配置, +137.9%/1.31)
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--notional", type=float, default=20_000)
    ap.add_argument("--commission", type=float, default=0.00025)
    ap.add_argument("--min-commission", type=float, default=5.0)
    ap.add_argument("--stamp", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.001)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
    cost_cfg = {"commission": args.commission, "min_commission": args.min_commission,
                "stamp": args.stamp, "slip": args.slip}
    cfg = {"down_days": args.down_days, "down_thresh": args.down_thresh,
           "atr_period": args.atr_period, "max_atr_pct": args.max_atr_pct,
           "ma_period": args.ma_period, "stop_pct": args.stop_pct,
           "time_stop": args.time_stop, "notional": args.notional,
           "vol_n": VOL_N if args.vol_order != "off" else 0}

    real_dir = args.real_data or args.data
    if not args.real_data and args.data == "a_share_daily_hfq" \
            and os.path.isdir("a_share_daily"):
        real_dir = "a_share_daily"

    cal = trading_calendar(args.data)
    signal_date = cal[-1]
    cal_idx = {d: i for i, d in enumerate(cal)}

    # 真实成交价缓存(与 --data 目录不同时才读)
    price_cache: dict = {}

    def real_price(code: str, date: str):
        """真实(不复权)收盘价+昨收，用于建议股数/涨停提示；失败返回 None。"""
        key = (real_dir, code)
        if key not in price_cache:
            try:
                df = pd.read_csv(os.path.join(real_dir, code + ".csv"),
                                 usecols=["date", "close", "preclose"])
                df = df.sort_values("date")
                price_cache[key] = df
            except Exception:                            # noqa: BLE001
                price_cache[key] = None
        df = price_cache[key]
        if df is None or len(df) == 0:
            return None
        row = df[df["date"].astype(str) == date]
        if len(row) == 0:
            row = df.tail(1)
        if len(row) == 0:
            return None
        return float(row["close"].iloc[-1]), float(row["preclose"].iloc[-1])

    # 任一推送通道开: 捕获报告文本, 跑完打印到终端并推送
    tg_buf = None
    if args.tg or args.pp or args.sc:
        tg_buf = io.StringIO()
        _real_stdout = sys.stdout
        sys.stdout = tg_buf

    # ---- 1) 回填当日成交(在信号判定之前更新台账) ----
    if args.buy_fills or args.sell_fills:
        ledger = load_ledger(args.positions)
        if args.buy_fills:
            for r in parse_fills(args.buy_fills, args.notional, 100, signal_date):
                if r["shares"] <= 0:
                    print("跳过 %s: 股数为0(价格过高或未指定)" % r["code"])
                    continue
                ledger["positions"].append({
                    "code": r["code"], "entry_date": r["date"],
                    "entry_price": round(r["price"], 4),
                    "shares": r["shares"],
                    "stop_price": round(r["price"] * (1 - args.stop_pct), 4),
                    "stop_pct": args.stop_pct,
                })
                print("已记录买入: %s %s %d股 @ %.3f (止损 %.3f)"
                      % (r["date"], r["code"], r["shares"], r["price"],
                         r["price"] * (1 - args.stop_pct)))
        if args.sell_fills:
            for r in parse_fills(args.sell_fills, args.notional, 100, signal_date):
                hit = None
                for p in ledger["positions"]:
                    if p["code"] == r["code"]:
                        hit = p
                        break
                if hit is None:
                    print("跳过 %s: 台账无此持仓" % r["code"])
                    continue
                gross = r["price"] / hit["entry_price"] - 1
                net = net_round_trip(hit["entry_price"], r["price"], hit["shares"], cost_cfg)
                ledger["positions"].remove(hit)
                ledger["closed"].append({
                    **hit, "exit_date": r["date"], "exit_price": round(r["price"], 4),
                    "gross_ret": round(gross, 6), "net_ret": round(net, 6),
                })
                print("已记录卖出: %s %s @ %.3f  毛利%+.2f%% 净利%+.2f%%"
                      % (r["date"], r["code"], r["price"], gross * 100, net * 100))
        save_ledger(args.positions, ledger)
        print("台账已更新: %s (持仓%d / 历史%d)"
              % (args.positions, len(ledger["positions"]), len(ledger["closed"])))
        print()

    ledger = load_ledger(args.positions)
    print("=" * 78)
    print("恐慌日超跌反弹 | 实盘信号 | 信号日 %s | 次日开盘执行" % signal_date)
    print("数据: %s | 执行价: %s | 恐慌阈值 %d | 市值 [%g, %g] 亿 | 资金 %.0f万 每笔%.0f万"
          % (args.data, real_dir, args.panic_threshold, args.min_mc, args.max_mc,
             args.capital / 1e4, args.notional / 1e4))
    print("=" * 78)

    # ---- 2) 持仓出场判定 ----
    sells, holds = [], []
    for p in ledger["positions"]:
        r = eval_position(p, cal_idx, signal_date, args.data, real_dir, cfg)
        (sells if r["action"].startswith("SELL") else holds).append(r)

    print()
    print("[持仓] 共 %d 个" % len(ledger["positions"]))
    if ledger["positions"]:
        print("  代码        入仓日       入仓价   现价    持K   浮盈亏   状态")
        for r in holds:
            if r["cur"] is None:
                print("  %-11s %s  %.4f   --    --     --      %s"
                      % (r["code"], r["entry_date"], r["entry_price"], r["note"]))
            else:
                pnl = (r["cur"] / r["entry_price"] - 1) * 100
                print("  %-11s %s  %.4f  %.4f  %3d  %+7.2f%%  %s"
                      % (r["code"], r["entry_date"], r["entry_price"], r["cur"],
                         r["bars"], pnl, r["note"]))
        for r in sells:
            pnl = (r["cur"] / r["entry_price"] - 1) * 100
            print("  %-11s %s  %.4f  %.4f  %3d  %+7.2f%%  >>> %s"
                  % (r["code"], r["entry_date"], r["entry_price"], r["cur"],
                     r["bars"], pnl, r["note"]))
    else:
        print("  (空仓)")

    print()
    print("[出场信号 —— 明日开盘卖出] %d 个" % len(sells))
    for r in sells:
        cur = r["cur"] if r["cur"] else r["entry_price"]
        net = net_round_trip(r["entry_price"], cur, r["shares"], cost_cfg)
        print("  %s %-11s 入仓%.3f 现价%.3f 止损%.3f  估算净利%+.2f%%  (%s)"
              % (r["code"], r["action"], r["entry_price"], cur, r["stop_price"],
                 net * 100, r["note"]))
    if not sells:
        print("  (无)")

    if not args.list:
        # ---- 3) 全市场扫描个股信号 + 恐慌日判定 ----
        files = build_files(args.data, args.universe, args.codes_file)
        t0 = time.time()
        tasks = [(f, cfg, signal_date) for f in files]
        sigs = []
        if args.workers > 1:
            with multiprocessing.Pool(processes=args.workers) as pool:
                for _code, s in pool.imap_unordered(scan_one, tasks, chunksize=16):
                    if s is not None:
                        sigs.append(s)
        else:
            for f in files:
                _code, s = scan_one((f, cfg, signal_date))
                if s is not None:
                    sigs.append(s)
        n_sig = len(sigs)
        is_panic = n_sig >= args.panic_threshold
        print()
        print("[全市场扫描] 主板 %d 只 | 个股层信号 %d 个 (用时 %.0fs)"
              % (len(files), n_sig, time.time() - t0))
        print("[恐慌日判定] %s"
              % ("是 [OK] (>= %d 个信号)" % args.panic_threshold if is_panic
                 else "否 (仅 %d 个，未达 %d，不买入)" % (n_sig, args.panic_threshold)))

        # ---- 4) 市值过滤 + 排序 + 买入清单 ----
        print()
        # 默认排序文案(与下方实际排序分支保持一致; 非恐慌日不会买入但提示仍打印)
        order_txt = ("量比从小到大(低放量优先)" if args.vol_order == "low"
                     else "量比从大到小(高放量优先)" if args.vol_order == "high"
                     else "小市值优先" if args.mc_order == "small"
                     else "大市值优先" if args.mc_order == "large"
                     else "随机")
        if not is_panic:
            print("[买入信号] 今日非恐慌日 -> 无买入，保持空仓观望(只处理上面的卖出)")
        else:
            picks = [s for s in sigs if s["mc"] == s["mc"]
                     and args.min_mc <= s["mc"] <= args.max_mc]
            # 量比排序可覆盖市值排序(回测结论: 放量越猛胜率越低, 低放量优先最优)
            if args.vol_order == "low":
                picks.sort(key=lambda s: (s["vol_ratio"], s["mc"]))
            elif args.vol_order == "high":
                picks.sort(key=lambda s: (s["vol_ratio"], s["mc"]), reverse=True)
            elif args.mc_order == "small":
                picks.sort(key=lambda s: s["mc"])
            elif args.mc_order == "large":
                picks.sort(key=lambda s: s["mc"], reverse=True)
            else:
                picks.sort(key=lambda s: s["code"])
            print("[买入信号 —— 次日开盘买入] %d 个 (市值过滤 [%g, %g] 亿，%s)"
                  % (len(picks), args.min_mc, args.max_mc, order_txt))
            rows = []
            for i, s in enumerate(picks, 1):
                rp = real_price(s["code"], signal_date)
                px = rp[0] if rp else s["close"]
                pre = rp[1] if rp else None
                shares = int(args.notional // (px * 100)) * 100
                cost = shares * px
                limit_note = ""
                if rp and pre and pre > 0 and px >= round(pre * 1.1, 2) - 0.001:
                    limit_note = "  !! 今日涨停，明日可能买不进"
                mc_txt = "%.0f" % s["mc"] if s["mc"] == s["mc"] else "--"
                vr_txt = "%.2f" % s["vol_ratio"] if s["vol_ratio"] == s["vol_ratio"] else "--"
                print("  %2d. %-11s 现价%.3f 15日跌%6.1f%% ATR%%%.1f 量比%5s 市值%5s亿  建议%d股 约%.0f元%s"
                      % (i, s["code"], px, s["ret15d"] * 100, s["atr_pct"] * 100,
                         vr_txt, mc_txt, shares, cost, limit_note))
                rows.append({"date": signal_date, "code": s["code"],
                             "close": round(px, 3),
                             "ret15d": round(s["ret15d"], 4),
                             "atr_pct": round(s["atr_pct"], 4),
                             "vol_ratio": round(s["vol_ratio"], 2) if s["vol_ratio"] == s["vol_ratio"] else None,
                             "mc": round(s["mc"], 1) if s["mc"] == s["mc"] else None,
                             "shares": shares, "est_cost": round(cost, 0),
                             "limit_note": limit_note.strip()})
            if rows:
                out = "signals_%s.csv" % signal_date
                pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
                print("  信号文件: %s (次日成交后改价存为 fills_buy.csv 回填)" % out)

        print()
        print("提示: 资金%.0f万 / 每笔%.1f万 -> 最多约 %d 个并发持仓；"
              % (args.capital / 1e4, args.notional / 1e4,
                 max(1, int(args.capital // args.notional))))
        print("      恐慌日信号多于可买数量时按上面排序(%s)依次买入。" % order_txt)
    else:
        print()
        print("(--list 模式，未扫描全市场，无买入信号)")

    # ---- 5) 推送: 还原 stdout，打印报告并推送到各通道 ----
    if args.tg or args.pp or args.sc:
        sys.stdout = _real_stdout
        report = tg_buf.getvalue()
        _safe_write(report)                          # GBK 终端编码不了 ✓ 等, 安全降级
        if args.tg:
            try:
                import tg_notify
                tcfg = tg_notify.load_config(args.tg_config)
                ok = tg_notify.send(report, tcfg, parse_mode="")
                sys.stdout.write("[TG推送] %s\n" % ("成功 OK" if ok else "失败"))
            except Exception as exc:                    # noqa: BLE001
                sys.stdout.write("[TG推送] 失败: %s\n" % exc)
        if args.pp:
            try:
                import pushplus_notify
                pcfg = pushplus_notify.load_config(args.pp_config)
                ok = pushplus_notify.send(report, pcfg, title="恐慌超跌信号")
                sys.stdout.write("[PushPlus] %s\n" % ("成功 OK" if ok else "失败"))
            except Exception as exc:                    # noqa: BLE001
                sys.stdout.write("[PushPlus] 失败: %s\n" % exc)
        if args.sc:
            try:
                import serverchan_notify
                scfg = serverchan_notify.load_config(args.sc_config)
                ok = serverchan_notify.send(report, scfg, title="恐慌超跌信号")
                sys.stdout.write("[Server酱] %s\n" % ("成功 OK" if ok else "失败"))
            except Exception as exc:                    # noqa: BLE001
                sys.stdout.write("[Server酱] 失败: %s\n" % exc)


if __name__ == "__main__":
    main()


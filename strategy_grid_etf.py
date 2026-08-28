# -*- coding: utf-8 -*-
"""
strategy_grid_etf.py — 沪深300估值加权网格(非纯马丁)
=====================================================

把纯马丁"亏损翻倍加注"改造成有锚、有界的摊薄网格，与恐慌策略对照。

锚: 沪深300收盘价的滚动 quantile_window(5)年分位数(替代PE, 等价均值回归信号)。
  - 价格越跌, 分位越低 → 加仓份额越大(线性阶梯, 非翻倍, 控制膨胀)
  - 价格回到高位 → 止盈
  - 最多 max_grids(6) 档补仓, 资金分档预留, 用完躺平(不融资, 不无限加倍)
标的用 ETF(510300) 模拟: ETF价格 ≈ 指数点 / price_scale(默认1000),
  1手=100份, 与实盘510300对齐(指数4000点→ETF约4元→1手400元, 小资金可做)。
  分位数仍用指数点计算(等价), 交易用ETF价格整手。

对照基线: 同期一次性买入持有(等额资金投入沪深300ETF)。

成本: 佣金万2.5/边 min5元, 印花税卖千0.5, 滑点0.1%/边(与回测口径一致)。

用法:
    python strategy_grid_etf.py                       # 默认 10万 沪深300
    python strategy_grid_etf.py --capital 200000 --quantile-window 5
"""

import argparse
import os

import numpy as np
import pandas as pd


def load_index(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    for c in ("close", "high", "low", "open"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return df


def backtest_grid(df: pd.DataFrame, capital: float, n_grids: int,
                  quantile_window: int, comm: float, minc: float,
                  stamp: float, slip: float,
                  buy_levels, sell_pct: float, price_scale: float = 1000.0) -> dict:
    """
    阶梯网格:
      buy_levels = [(分位上界, 份数)] 从低到高, 如 [(0.15,4),(0.30,2),(0.50,1)]
      价格跌破某档分位 → 按该档份数建仓(整手100股等价为"份")
      全部仓位在 价格回升到 sell_quantile分位 或 单档+sell_pct 时止盈
    每份名义本金 = capital / (n_grids * max_份额倍数) 预留, 用完不再加
    price_scale: 指数点→ETF价格除数(510300≈1000)。分位用指数点, 交易用ETF价。
    """
    # 滚动分位(用截至当日的历史, 无未来函数; 窗口不足时用全部历史)
    lookback = 242 * quantile_window
    q = df["close"].rolling(lookback, min_periods=60).rank(pct=True)
    df = df.assign(q=q)
    # ETF价格(用于整手交易) = 指数点 / scale
    df["etf"] = df["close"] / price_scale

    unit = capital / n_grids            # 每档预留资金
    cash = capital
    # 持仓: list of {shares, cost_basis, lots}
    positions = []   # 每档建仓记录
    total_shares = 0
    total_cost = 0.0
    nav_series = []
    # 建仓状态: 当前已建到第几档(避免同档重复加)
    grid_idx = 0
    # buy_levels 按分位从低到高排, 越低越先触发
    levels = sorted(buy_levels, key=lambda x: x[0])  # [(q_thr, lots), ...]

    def buy_cost(shares, price):
        gross = shares * price
        fee = max(minc, gross * comm) + gross * slip
        return gross + fee

    def sell_proceeds(shares, price):
        gross = shares * price
        fee = max(minc, gross * comm) + gross * stamp + gross * slip
        return gross - fee

    trades = []  # 交易记录
    for i, row in df.iterrows():
        px = float(row["etf"])                       # ETF价(交易用)
        idx = float(row["close"])                    # 指数点(仅打印)
        qq = float(row["q"]) if pd.notna(row["q"]) else np.nan
        # ---- 建仓: 分位跌破下一档 ----
        if pd.notna(qq) and grid_idx < len(levels):
            q_thr, lots = levels[grid_idx]
            if qq <= q_thr:
                # 份数: 每份买 unit 价值, 整手对齐(ETF 1手=100份)
                shares = int(unit * lots / (px * 100)) * 100
                if shares > 0:
                    need = buy_cost(shares, px)
                    if cash >= need:
                        cash -= need
                        total_shares += shares
                        total_cost += need
                        positions.append({"shares": shares, "price": px,
                                          "lots": lots, "date": row["date"]})
                        trades.append({"date": row["date"], "type": "BUY",
                                       "price": px, "idx": idx, "shares": shares,
                                       "q": round(qq, 3), "grid": grid_idx})
                        grid_idx += 1
        # ---- 止盈: 分位回升到高位 或 整体浮盈达 sell_pct ----
        if total_shares > 0 and pd.notna(qq):
            avg_cost = total_cost / total_shares if total_shares else px
            hit_q = qq >= 0.70
            hit_pct = px / avg_cost - 1 >= sell_pct if total_shares else False
            if hit_q or hit_pct:
                proceeds = sell_proceeds(total_shares, px)
                ret = (proceeds - total_cost) / total_cost if total_cost else 0
                trades.append({"date": row["date"], "type": "SELL", "price": px,
                               "idx": idx, "shares": total_shares, "q": round(qq, 3),
                               "ret": round(ret, 4), "reason": "q>=0.70" if hit_q else f"+{sell_pct:.0%}"})
                cash += proceeds
                total_shares = 0
                total_cost = 0.0
                positions = []
                grid_idx = 0   # 重置, 下一轮重新从低档建仓
        # ---- 盯市 ----
        nav = cash + total_shares * px
        nav_series.append({"date": row["date"], "nav": nav,
                            "q": qq, "n_pos": len(positions)})

    nav_df = pd.DataFrame(nav_series)
    nav_df = nav_df.dropna(subset=["nav"])
    # 最后强制平仓统计(期末市值)
    final_nav = nav_df["nav"].iloc[-1] if len(nav_df) else capital
    navv = nav_df["nav"].to_numpy()
    peak = np.maximum.accumulate(navv)
    mdd = float(((peak - navv) / peak).max()) if len(navv) else 0.0
    yrs = (nav_df["date"].iloc[-1] - nav_df["date"].iloc[0]).days / 365.25 if len(nav_df) else 0
    total_ret = final_nav / capital - 1
    cagr = (final_nav / capital) ** (1 / yrs) - 1 if yrs > 0 else 0
    r = np.diff(navv) / navv[:-1]
    sharpe = float(np.mean(r) / np.std(r) * np.sqrt(242)) if np.std(r) > 0 else 0.0
    tr = pd.DataFrame(trades)
    # 交易层统计(每轮止盈的ret)
    sells = tr[tr["type"] == "SELL"] if not tr.empty else tr
    win_rate = (sells["ret"] > 0).mean() if len(sells) else 0.0
    return {"nav_df": nav_df, "trades": tr, "final_nav": final_nav,
            "total_ret": total_ret, "cagr": cagr, "mdd": mdd, "sharpe": sharpe,
            "n_rounds": len(sells), "win_rate": win_rate,
            "avg_round_ret": sells["ret"].mean() if len(sells) else 0.0}


def buy_and_hold(df: pd.DataFrame, capital: float, comm: float,
                 minc: float, slip: float, price_scale: float = 1000.0) -> dict:
    """一次性买入持有对照(同期等额资金全仓沪深300ETF, 期末市价)。"""
    etf = df["close"].to_numpy() / price_scale
    px0 = float(etf[0])
    shares = int(capital / (px0 * 100)) * 100
    cost = shares * px0 + max(minc, shares * px0 * comm) + shares * px0 * slip
    cash = capital - cost
    nav = cash + shares * etf
    final = cash + shares * float(etf[-1])
    peak = np.maximum.accumulate(nav)
    mdd = float(((peak - nav) / peak).max())
    yrs = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    r = np.diff(nav) / nav[:-1]
    sharpe = float(np.mean(r) / np.std(r) * np.sqrt(242)) if np.std(r) > 0 else 0.0
    return {"final_nav": final, "total_ret": final / capital - 1,
            "cagr": (final / capital) ** (1 / yrs) - 1 if yrs > 0 else 0,
            "mdd": mdd, "sharpe": sharpe, "shares": shares}


def main() -> None:
    ap = argparse.ArgumentParser(description="沪深300估值加权网格(非纯马丁)回测")
    ap.add_argument("--data", default="a_share_daily_hfq/sh.000300.csv")
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--quantile-window", type=int, default=5, help="分位回看年数(默认5)")
    ap.add_argument("--n-grids", type=int, default=6, help="最大补仓档数(默认6)")
    ap.add_argument("--price-scale", type=float, default=1000.0,
                    help="指数点→ETF价除数(510300≈1000; 默认1000)")
    ap.add_argument("--sell-pct", type=float, default=0.15, help="整体浮盈止盈阈值(默认15%)")
    ap.add_argument("--commission", type=float, default=0.00025)
    ap.add_argument("--min-commission", type=float, default=5.0)
    ap.add_argument("--stamp", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.001)
    ap.add_argument("--nav-out", default="trades_grid_hs300_nav.csv")
    args = ap.parse_args()

    df = load_index(args.data)
    print(f"沪深300: {len(df)}根日线  {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    print(f"资金 {args.capital/1e4:.0f}万 | 网格档 {args.n_grids} | 分位窗 {args.quantile_window}年 | 止盈+{args.sell_pct:.0%}或q>=0.70")
    print("=" * 64)

    # 阶梯: 越低分位加越多(线性非翻倍, 控膨胀)
    # q<=0.50 买1份, q<=0.30 加2份, q<=0.15 加3份 (共3档6份, 对应n_grids=6)
    buy_levels = [(0.50, 1), (0.30, 2), (0.15, 3)]

    g = backtest_grid(df, args.capital, args.n_grids, args.quantile_window,
                      args.commission, args.min_commission, args.stamp, args.slip,
                      buy_levels, args.sell_pct, args.price_scale)
    bh = buy_and_hold(df, args.capital, args.commission, args.min_commission,
                      args.slip, args.price_scale)

    print("【估值加权网格】")
    print(f"  期末净值: {g['final_nav']/1e4:.1f}万   总收益 {g['total_ret']:+.1%}   CAGR {g['cagr']:+.1%}")
    print(f"  最大回撤: {g['mdd']:.1%}   夏普 {g['sharpe']:.2f}")
    print(f"  完成轮数: {g['n_rounds']}   胜率 {g['win_rate']:.0%}   单轮均收益 {g['avg_round_ret']:+.1%}")
    print()
    print("【一次性买入持有(对照)】")
    print(f"  期末净值: {bh['final_nav']/1e4:.1f}万   总收益 {bh['total_ret']:+.1%}   CAGR {bh['cagr']:+.1%}")
    print(f"  最大回撤: {bh['mdd']:.1%}   夏普 {bh['sharpe']:.2f}")
    print()
    print("-" * 64)
    print(f"网格 vs 持有: 收益 {g['total_ret']:+.1%} vs {bh['total_ret']:+.1%}  "
          f"| 回撤 {g['mdd']:.1%} vs {bh['mdd']:.1%}  | 夏普 {g['sharpe']:.2f} vs {bh['sharpe']:.2f}")
    if g["trades"] is not None and not g["trades"].empty:
        print()
        print("交易明细:")
        for _, r in g["trades"].iterrows():
            if r["type"] == "BUY":
                print(f"  {r['date'].date()} BUY  指数{r['idx']:.0f}/ETF{r['price']:.3f}  "
                      f"{int(r['shares'])}份  q={r['q']}  档{r['grid']}")
            else:
                print(f"  {r['date'].date()} SELL 指数{r['idx']:.0f}/ETF{r['price']:.3f}  "
                      f"{int(r['shares'])}份  q={r['q']}  收益{r['ret']:+.1%}  [{r['reason']}]")

    g["nav_df"].assign(nav=lambda x: x["nav"]).to_csv(args.nav_out, index=False, encoding="utf-8-sig")
    print(f"\n净值曲线: {args.nav_out}")


if __name__ == "__main__":
    main()

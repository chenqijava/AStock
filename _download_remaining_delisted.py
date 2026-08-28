# -*- coding: utf-8 -*-
"""下载剩余退市股后复权日线数据"""
import baostock as bs
import os
import pandas as pd
import time

DELIST_TXT = "delisted_codes.txt"
OUT_DIR = "a_share_delisted_hfq"
START_DATE = "2016-01-01"
END_DATE = "2026-08-31"

os.makedirs(OUT_DIR, exist_ok=True)

# 读取退市股列表
with open(DELIST_TXT, "r") as f:
    all_codes = [line.strip() for line in f if line.strip()]

# 过滤已下载的
to_download = []
for code in all_codes:
    path = os.path.join(OUT_DIR, code + ".csv")
    if not (os.path.exists(path) and os.path.getsize(path) > 100):
        to_download.append(code)

print(f"总退市股: {len(all_codes)}, 已下载: {len(all_codes)-len(to_download)}, 待下载: {len(to_download)}")

lg = bs.login()
print(f"baostock login: {lg.error_msg}")

downloaded = 0
failed = []
for i, code in enumerate(to_download):
    out_path = os.path.join(OUT_DIR, code + ".csv")
    
    rs = bs.query_history_k_data_plus(
        code,
        "date,code,open,high,low,close,volume,amount,turn,isST",
        start_date=START_DATE, end_date=END_DATE,
        frequency="d", adjustflag="2"
    )
    if rs.error_code != "0":
        failed.append(code)
        continue
    
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    
    if len(rows) < 30:
        failed.append(code)
        continue
    
    df = pd.DataFrame(rows, columns=rs.fields)
    for col in ["open", "high", "low", "close", "volume", "amount", "turn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["isST"] = df["isST"].apply(lambda x: 1 if str(x) == "1" else 0)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    downloaded += 1
    
    if (i + 1) % 10 == 0:
        print(f"  进度: [{i+1}/{len(to_download)}] 成功{downloaded} 失败{len(failed)}")

bs.logout()
print(f"\n完成: 新下载 {downloaded} 只, 失败 {len(failed)} 只")
print(f"失败列表: {failed[:20]}")

# 统计总数
total = len([f for f in os.listdir(OUT_DIR) if f.endswith(".csv")])
print(f"a_share_delisted_hfq 目录总文件数: {total}")

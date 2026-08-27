# -*- coding: utf-8 -*-
"""
tg_notify.py — 推送消息到 Telegram 群(Bot API)
================================================

配置存于 tg_config.json(默认，可用 --config 指定):
    {
      "token":   "123456:ABC...",          # bot token
      "chat_id": "-1001234567890",         # 群 chat_id(负号开头)或用户ID
      "proxy":   "http://127.0.0.1:10808"  # 可选；国内直连 api.telegram.org 被墙时需要
    }

用法
----
    python tg_notify.py --test                # 发一条测试消息验证配置
    python tg_notify.py "任意文本"            # 发纯文本
    python tg_notify.py --html "<b>加粗</b>"  # 发 HTML(parse_mode=HTML)
    python tg_notify.py --get-chat            # 列出 bot 已加入的群(找 chat_id)
    python tg_notify.py --file report.txt     # 发送文件内容

安全提示: tg_config.json 含 token，勿提交到公开仓库。
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

DEFAULT_CONFIG = "tg_config.json"
API = "https://api.telegram.org/bot{token}/{method}"


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        sys.stderr.write("未找到配置文件: %s\n" % path)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if not cfg.get("token"):
        sys.stderr.write("配置文件缺少 token\n")
        sys.exit(1)
    return cfg


def _opener(cfg: dict):
    if cfg.get("proxy"):
        ph = urllib.request.ProxyHandler(
            {"http": cfg["proxy"], "https": cfg["proxy"]})
        return urllib.request.build_opener(ph)
    return urllib.request.build_opener()


def api_call(cfg: dict, method: str, data: dict = None) -> dict:
    """调用 Bot API。data 为表单字段。失败/非 ok 返回 {'ok': False, 'error': ...}。"""
    url = API.format(token=cfg["token"], method=method)
    try:
        body = urllib.parse.urlencode(data).encode() if data else None
        req = urllib.request.Request(url, data=body, method="POST" if body else "GET")
        with _opener(cfg).open(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as exc:                            # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def send(text: str, cfg: dict, parse_mode: str = "HTML") -> bool:
    """发送消息到群。返回是否成功。"""
    if not cfg.get("chat_id"):
        sys.stderr.write("配置文件缺少 chat_id(用 --get-chat 找群，或填群ID)\n")
        return False
    if parse_mode:
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    data = {"chat_id": cfg["chat_id"], "text": text}
    if parse_mode:
        data["parse_mode"] = "HTML"
    res = api_call(cfg, "sendMessage", data)
    if not res.get("ok"):
        sys.stderr.write("发送失败: %s\n" % res.get("error", res.get("description", res)))
    return bool(res.get("ok"))


def get_updates(cfg: dict) -> dict:
    """取 bot 收到的更新(含群信息，用于找 chat_id)。"""
    res = api_call(cfg, "getUpdates")
    if not res.get("ok"):
        sys.stderr.write("getUpdates 失败: %s\n" % res.get("error", res.get("description", res)))
        return {"ok": False, "result": []}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description="推送消息到 Telegram 群",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="配置文件(json)")
    ap.add_argument("--file", default=None, help="发送该文件内容")
    ap.add_argument("--html", default=None, help="发送 HTML 文本(parse_mode=HTML)")
    ap.add_argument("--test", action="store_true", help="发一条测试消息")
    ap.add_argument("--get-chat", action="store_true", help="列出 bot 已加入的群(找 chat_id)")
    ap.add_argument("text", nargs="*", help="要发送的纯文本")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.get_chat:
        up = get_updates(cfg)
        seen = {}
        for u in up.get("result", []):
            chat = (u.get("message") or u.get("edited_message") or {})
            c = chat.get("chat", {})
            if c.get("id") is not None:
                seen[c["id"]] = (c.get("title") or c.get("username") or "?" , c.get("type"))
        if not seen:
            print("未发现更新。请先把 bot 拉进目标群并在群里发一条消息(或@bot)，再运行本命令。")
        else:
            print("找到的 chat_id (群 id 以 -100 开头):")
            for cid, (name, typ) in seen.items():
                print("  %s  %s  (%s)" % (cid, name, typ))
        return

    if args.test:
        ok = send("✅ 测试消息: 恐慌超跌信号机器人已就绪", cfg, parse_mode="")
        print("测试消息发送%s" % ("成功" if ok else "失败"))
        return

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    elif args.html is not None:
        text, pm = args.html, "HTML"
    else:
        text, pm = " ".join(args.text), ""
    if not text.strip():
        sys.stderr.write("无内容可发送\n")
        sys.exit(1)
    send(text, cfg, pm)


if __name__ == "__main__":
    main()

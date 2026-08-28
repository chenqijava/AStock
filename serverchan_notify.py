# -*- coding: utf-8 -*-
"""
serverchan_notify.py — 推送消息到微信(Server酱)
=================================================

Server酱(sct.ftqq.com) 是方糖团队提供的微信推送服务: 拿一个 sendkey,
通过 HTTP 接口把消息推到绑定微信。与 PushPlus 互为备选通道。

Turbo 版接口(sctapi): POST https://sctapi.ftqq.com/<sendkey>.send
  body: title=标题&desp=正文(支持 Markdown)
  成功返回: {"code": 0, "message": "...", "data": {"pushid": "..."}}

配置存于 serverchan_config.json(默认, 可用 --config 指定):
    {
      "sendkey": "SCT123456abcdef..."
    }

用法
----
    python serverchan_notify.py --test                  # 发一条测试消息验证 sendkey
    python serverchan_notify.py "任意文本"               # 发纯文本
    python serverchan_notify.py --title 标题 "正文"      # 带标题
    python serverchan_notify.py --file report.txt       # 发送文件内容
    python serverchan_notify.py --md "## 标题\n正文"     # 发 Markdown

安全提示: serverchan_config.json 含 sendkey, 勿提交到公开仓库(已加入 .gitignore)。
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

DEFAULT_CONFIG = "serverchan_config.json"
API = "https://sctapi.ftqq.com/{sendkey}.send"


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        sys.stderr.write("未找到配置文件: %s\n" % path)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if not cfg.get("sendkey"):
        sys.stderr.write("配置文件缺少 sendkey(去 https://sct.ftqq.com/ 获取)\n")
        sys.exit(1)
    return cfg


def send(text: str, cfg: dict, title: str = "信号") -> bool:
    """发送消息到微信。返回是否成功。

    text:  消息正文(Server酱支持 Markdown; Turbo 版单条上限 ~64KB)
    title: 消息标题(必填, 必须有值)
    """
    if not title.strip():
        title = "信号"
    # 正文超长截断(Server酱建议 < 32KB, 这里留足 32KB)
    if len(text) > 32000:
        text = text[:32000] + "\n...(正文过长已截断)"
    payload = urllib.parse.urlencode(
        {"title": title, "desp": text}).encode("utf-8")
    try:
        api = API.format(sendkey=cfg["sendkey"])
        req = urllib.request.Request(api, data=payload, method="POST",
                                     headers={"Content-Type":
                                              "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=30) as r:
            res = json.loads(r.read().decode("utf-8"))
    except Exception as exc:                            # noqa: BLE001
        sys.stderr.write("Server酱 请求失败: %s\n" % exc)
        return False
    if res.get("code") != 0:
        sys.stderr.write("Server酱 推送失败: %s\n" % res.get("message", res))
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="推送消息到微信(Server酱)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="配置文件(json)")
    ap.add_argument("--title", default="信号", help="消息标题(默认'信号')")
    ap.add_argument("--file", default=None, help="发送该文件内容")
    ap.add_argument("--md", default=None, help="发送 Markdown 文本(desp)")
    ap.add_argument("--test", action="store_true", help="发一条测试消息")
    ap.add_argument("text", nargs="*", help="要发送的纯文本")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.test:
        ok = send("✅ 测试消息: Server酱 微信推送已就绪", cfg, title="测试")
        print("测试消息发送%s" % ("成功" if ok else "失败"))
        return

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    elif args.md is not None:
        text = args.md
    else:
        text = " ".join(args.text)
    if not text.strip():
        sys.stderr.write("无内容可发送\n")
        sys.exit(1)
    send(text, cfg, title=args.title)


if __name__ == "__main__":
    main()

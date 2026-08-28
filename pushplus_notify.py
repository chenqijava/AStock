# -*- coding: utf-8 -*-
"""
pushplus_notify.py — 推送消息到微信(PushPlus)
============================================

PushPlus 是一个把消息转发到微信的推送服务, 只需一个 token 即可
(扫码关注公众号绑定后, 在 https://www.pushplus.plus/ 获取 token)。
无需代理, 国内直连, 无需自建 bot, 适合不便使用 Telegram 的场景。

配置存于 pushplus_config.json(默认, 可用 --config 指定):
    {
      "token": "your_pushplus_token",
      "topic": "可选, 群组编码(一对一推送留空即可)"
    }

API
----
    POST http://www.pushplus.plus/send
    body(json): {"token": ..., "title": ..., "content": ..., "template": "txt", "topic": "(可选)"}
    成功返回: {"code": 200, "msg": "请求成功", "data": "..."}

用法
----
    python pushplus_notify.py --test                 # 发一条测试消息验证 token
    python pushplus_notify.py "任意文本"              # 发纯文本
    python pushplus_notify.py --title 标题 "正文"     # 带标题
    python pushplus_notify.py --file report.txt      # 发送文件内容
    python pushplus_notify.py --html "<b>加粗</b>"    # 发 HTML(template=html)

安全提示: pushplus_config.json 含 token, 勿提交到公开仓库(已加入 .gitignore)。
"""
import argparse
import json
import os
import sys
import urllib.request

DEFAULT_CONFIG = "pushplus_config.json"
API = "http://www.pushplus.plus/send"


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        sys.stderr.write("未找到配置文件: %s\n" % path)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if not cfg.get("token"):
        sys.stderr.write("配置文件缺少 token(去 https://www.pushplus.plus/ 获取)\n")
        sys.exit(1)
    return cfg


def send(text: str, cfg: dict, title: str = "信号",
         template: str = "txt") -> bool:
    """发送消息到微信。返回是否成功。

    text:    消息正文
    title:   消息标题(微信卡片上显示)
    template: txt(默认纯文本) / html / json / markdown
    """
    payload = {
        "token": cfg["token"],
        "title": title,
        "content": text,
        "template": template,
    }
    if cfg.get("topic"):
        payload["topic"] = cfg["topic"]
    try:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            API, data=body, method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            res = json.loads(r.read().decode("utf-8"))
    except Exception as exc:                            # noqa: BLE001
        sys.stderr.write("PushPlus 请求失败: %s\n" % exc)
        return False
    if res.get("code") != 200:
        sys.stderr.write("PushPlus 推送失败: %s\n" % res.get("msg", res))
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="推送消息到微信(PushPlus)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n")[1] if "用法\n" in __doc__ else "",
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="配置文件(json)")
    ap.add_argument("--title", default="信号", help="消息标题(默认'信号')")
    ap.add_argument("--file", default=None, help="发送该文件内容")
    ap.add_argument("--html", default=None,
                    help="发送 HTML 文本(template=html)")
    ap.add_argument("--test", action="store_true", help="发一条测试消息")
    ap.add_argument("text", nargs="*", help="要发送的纯文本")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.test:
        ok = send("✅ 测试消息: PushPlus 微信推送已就绪", cfg,
                  title="测试", template="txt")
        print("测试消息发送%s" % ("成功" if ok else "失败"))
        return

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
        tmpl = "txt"
    elif args.html is not None:
        text, tmpl = args.html, "html"
    else:
        text, tmpl = " ".join(args.text), "txt"
    if not text.strip():
        sys.stderr.write("无内容可发送\n")
        sys.exit(1)
    send(text, cfg, title=args.title, template=tmpl)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
在自己電腦上執行 XRD Studio（自己使用時建議用這個方式）。
執行：python run_local.py   → 自動開啟 http://127.0.0.1:8000/
結束：在這個視窗按 Ctrl+C（Spyder 可按 Console 的停止鍵）。

和放在 GitHub Pages 上的差別：「鑑定」分頁可以直接在工具內顯示 webmineral 的查詢結果。
webmineral 不允許其他網頁從瀏覽器直接讀取它的結果，所以由這支程式在本機代為查詢
（只轉送 webmineral X-Ray 搜尋頁，一次查詢等同你自己在網頁上按一次 Submit）。
需要網路（第一次會從 jsDelivr 下載瀏覽器用的 Python 執行環境，之後由瀏覽器快取）。
"""
import functools
import http.server
import json
import os
import re
import threading
import urllib.parse
import urllib.request
import webbrowser

PORT = 8000
HERE = os.path.dirname(os.path.abspath(__file__))
WM_URL = "https://webmineral.com/MySQL/xray.php"
NUM = re.compile(r"^\d{0,3}(\.\d{1,6})?$")


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path != "/wm":
            return super().do_GET()
        q = urllib.parse.parse_qs(u.query)
        if "ping" in q:
            return self._send(200, json.dumps({"ok": True}).encode(), "application/json")
        params = {"lambda": "1.54056 - CuKa1", "chem": "", "submit": "Submit"}
        for k in ("ed1", "ed2", "ed3", "minmax", "st"):
            v = (q.get(k) or [""])[0].strip()
            if v and not NUM.match(v):
                return self._send(400, b"bad parameter", "text/plain")
            params[k] = v
        req = urllib.request.Request(WM_URL + "?" + urllib.parse.urlencode(params),
                                     headers={"User-Agent": "Mozilla/5.0 (XRD Studio personal use)"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
        except Exception as e:
            return self._send(502, ("webmineral 無法連線：%s" % e).encode("utf-8"), "text/plain; charset=utf-8")
        self._send(200, body, "text/html; charset=windows-1252")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


Handler.extensions_map.update({".js": "text/javascript", ".py": "text/plain; charset=utf-8", ".gz": "application/gzip"})
server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), functools.partial(Handler, directory=HERE))
url = "http://127.0.0.1:%d/" % PORT
print("XRD Studio:", url, "(Ctrl+C to stop)")
threading.Timer(0.8, lambda: webbrowser.open(url)).start()
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    server.server_close()

# -*- coding: utf-8 -*-
"""
在自己電腦上預覽 XRD Studio（上傳到 GitHub Pages 之前先測試用）。
執行：python run_local.py   → 自動開啟 http://127.0.0.1:8000/
需要網路（第一次會從 jsDelivr 下載瀏覽器用的 Python 執行環境，之後由瀏覽器快取）。
結束：在這個視窗按 Ctrl+C（Spyder 可按 Console 的停止鍵）。
"""
import functools
import http.server
import os
import threading
import webbrowser

PORT = 8000
here = os.path.dirname(os.path.abspath(__file__))
Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=here)
server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
url = "http://127.0.0.1:%d/" % PORT
print("XRD Studio local preview:", url, "(Ctrl+C to stop)")
threading.Timer(0.8, lambda: webbrowser.open(url)).start()
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    server.server_close()

"""标准库 HTTP 服务：无第三方依赖即可运行看板与 API。

使用 ``ThreadingHTTPServer``，每个请求交给 ``Api.dispatch`` 处理，
与 FastAPI 版本共享同一套业务逻辑。适合内网离线部署或 CI 冒烟。
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .api import Api, Response
from .config import Config


def make_handler(api: Api, quiet: bool = True):
    class Handler(BaseHTTPRequestHandler):
        server_version = "kb-rag-eval/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            if not quiet:
                sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def handle_one_request(self) -> None:
            # 浏览器会主动关闭空闲的 keep-alive 连接，属于正常现象，不应打印堆栈
            try:
                super().handle_one_request()
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                self.close_connection = True

        def _handle(self, method: str) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            try:
                response: Response = api.dispatch(method, self.path, query, body)
            except Exception as exc:  # 兜底，防止连接被直接断开
                response = Response.json({"error": f"{type(exc).__name__}: {exc}"}, status=500)
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in response.headers.items():
                self.send_header(key, value)
            self.end_headers()
            if method != "HEAD":
                self.wfile.write(response.body)

        def do_GET(self) -> None:  # noqa: N802
            self._handle("GET")

        def do_HEAD(self) -> None:  # noqa: N802
            self._handle("HEAD")

        def do_POST(self) -> None:  # noqa: N802
            self._handle("POST")

        def do_DELETE(self) -> None:  # noqa: N802
            self._handle("DELETE")

    return Handler


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8000, quiet: bool = True) -> None:
    api = Api(cfg)
    handler = make_handler(api, quiet=quiet)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    banner = f"""
┌──────────────────────────────────────────────────────────┐
│  RAG 评测看板已启动                                      │
│  http://{host}:{port}
│  模式：{cfg.generate.provider}/{cfg.embedding.provider}/{cfg.store.provider}
│  数据库：{cfg.project.db_path}
└──────────────────────────────────────────────────────────┘
"""
    print(banner)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭…")
    finally:
        httpd.server_close()


def serve_in_thread(cfg: Config, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    """测试用：后台线程启动，返回 (server, thread, url)。"""
    api = Api(cfg)
    httpd = ThreadingHTTPServer((host, port), make_handler(api, quiet=True))
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    actual_port = httpd.server_address[1]
    return httpd, thread, f"http://{host}:{actual_port}"

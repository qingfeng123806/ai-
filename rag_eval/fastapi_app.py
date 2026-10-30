"""FastAPI 应用：与标准库服务器共享同一套 ``Api.dispatch`` 业务逻辑。

安装 fastapi + uvicorn 后可直接：``uvicorn rag_eval.fastapi_app:app --port 8000``，
或使用 CLI 的 ``serve --framework fastapi``。未安装 fastapi 时本模块导入会失败，
CLI 会自动回退到标准库服务器（rag_eval/dashboard.py）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import __version__
from .api import Api
from .config import Config


def create_app(config: Config | None = None) -> Any:
    from fastapi import FastAPI, Request
    from fastapi.responses import Response as FastAPIResponse

    cfg = config or Config.load(os.environ.get("RAG_EVAL_CONFIG", "configs/baseline.toml"))
    api = Api(cfg)
    app = FastAPI(
        title="RAG 自动化评测系统",
        version=__version__,
        description="检索质量 / 生成质量 / 安全合规 / 性能延迟的统一量化与 A/B 对比",
    )
    app.state.api = api

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD"])
    async def catch_all(request: Request, full_path: str) -> FastAPIResponse:
        query: dict[str, list[str]] = {}
        for key, value in request.query_params.multi_items():
            query.setdefault(key, []).append(value)
        body = await request.body()
        raw_path = "/" + full_path
        if request.url.query:
            raw_path += "?" + request.url.query
        response = api.dispatch(request.method, raw_path, query, body)
        return FastAPIResponse(
            content=response.body,
            status_code=response.status,
            media_type=response.content_type,
            headers=response.headers,
        )

    return app


def main() -> None:
    import uvicorn

    config_path = os.environ.get("RAG_EVAL_CONFIG", "configs/baseline.toml")
    port = int(os.environ.get("RAG_EVAL_PORT", "8000"))
    uvicorn.run(create_app(Config.load(config_path)), host="0.0.0.0", port=port)


try:  # 有 fastapi 时提供模块级 app，便于 uvicorn 直接引用
    app = create_app()
except Exception:  # pragma: no cover - 离线环境无 fastapi / 无配置文件
    app = None


__all__ = ["create_app", "main", "app"]

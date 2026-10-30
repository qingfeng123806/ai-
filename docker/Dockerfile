# syntax=docker/dockerfile:1
# kb-rag-eval 运行镜像：Python 3.12 slim + 多阶段构建 + 非 root 用户。
# 构建上下文必须是仓库根目录（Dockerfile 需要 COPY rag_eval / configs / data）：
#   docker build -f docker/Dockerfile -t kb-rag-eval:1.0.0 .

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

# 非 root 运行：固定 uid/gid，便于挂载卷的权限规划
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY rag_eval ./rag_eval
COPY configs ./configs
COPY data ./data

# 运行期写入目录（建议以卷挂载覆盖，便于保留多次评测结果）
RUN mkdir -p /app/artifacts && chown -R app:app /app

USER app

EXPOSE 8000

# 挂载点：语料、派生产物、配置
VOLUME ["/app/data", "/app/artifacts", "/app/configs"]

# 用标准库 urllib 探活，不引入 curl，保持镜像最小
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status == 200 else 1)"

ENTRYPOINT ["python", "-m", "rag_eval", "serve", "--host", "0.0.0.0", "--port", "8000"]

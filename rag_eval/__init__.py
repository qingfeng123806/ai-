"""RAG 自动化评测系统（kb-rag-eval）。

面向知识库问答的检索增强生成评测平台：把检索质量、生成质量、安全合规与性能
延迟统一量化，并支持切片策略 / TopK / 重排序 / Prompt / 模型的 A/B 对比。

本包的核心链路只依赖 Python 标准库，第三方能力（BGE、Chroma、Milvus、
RAGAs、DeepEval、FastAPI、OpenAI 兼容接口）以可选适配器方式接入，
缺失时自动降级到内置的确定性实现，因此可以在离线 / CI 环境中完整跑通。
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]

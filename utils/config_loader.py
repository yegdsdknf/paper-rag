import yaml
import os

from utils.console import ensure_utf8_console

_config_cache = None    # 模块级缓存

ensure_utf8_console()

def load_config(config_path: str = "./config.yaml") -> dict:
    """加载 YAML 配置文件（带缓存，同一进程只读一次）"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在：{config_path}")
        print("使用默认配置...")
        _config_cache = get_default_config()
        return _config_cache

    with open(config_path, "r", encoding="utf-8") as f:
        _config_cache = yaml.safe_load(f)
    print(f"✅ 配置文件已加载：{config_path}")
    return _config_cache

def get_default_config() -> dict:
    """返回默认配置（Phase 1 的兜底值，config.yaml 不存在时使用）"""
    return {
        "persist_directory": "./chroma_db",
        "k": 6,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "separators": ["\n\n", "\n", "。", ".", " ", ""],
        "embedding_model": "BAAI/bge-m3",
        "llm_model": "deepseek-r1:8b",
        "temperature": 0.1,
        "default_vector_weight": 0.5,
        "default_bm25_weight": 0.5,
        "skip_pages": {},
        "semantic_similarity_threshold": 0.7,
    }

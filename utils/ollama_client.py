from langchain_ollama import ChatOllama

from utils.config_loader import load_config


_config = load_config()
DEFAULT_NUM_CTX = _config.get("llm_num_ctx", 4096)
DEFAULT_NUM_PREDICT = _config.get("llm_num_predict", 1024)


def create_chat_ollama(
    model: str,
    temperature: float,
    num_ctx: int | None = None,
    num_predict: int | None = None,
) -> ChatOllama:
    """创建统一配置的 Ollama 聊天模型实例。"""
    return ChatOllama(
        model=model,
        temperature=temperature,
        num_ctx=num_ctx if num_ctx is not None else DEFAULT_NUM_CTX,
        num_predict=num_predict if num_predict is not None else DEFAULT_NUM_PREDICT,
    )

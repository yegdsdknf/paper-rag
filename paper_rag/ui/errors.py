from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


ErrorSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class FriendlyError:
    title: str
    message: str
    suggestions: list[str] = field(default_factory=list)
    details: str = ""
    show_doctor_hint: bool = True
    severity: ErrorSeverity = "error"


def _setting(settings: Mapping[str, Any] | None, key: str, default: str = "") -> str:
    if not settings:
        return default
    value = settings.get(key, default)
    return str(value) if value is not None else default


def format_runtime_error(error: Exception, settings: Mapping[str, Any] | None = None) -> FriendlyError:
    details = f"{type(error).__name__}: {error}"
    lowered = str(error).lower()

    if any(signal in lowered for signal in ["connection refused", "connect timeout", "read timed out", "ollama service"]):
        return FriendlyError(
            title="无法连接到 Ollama 服务",
            message="本地 LLM 服务当前不可用，问答生成无法继续。",
            suggestions=[
                "运行 ollama serve 启动 Ollama 服务。",
                "确认端口 11434 没有被防火墙或其他程序阻断。",
                "运行 python main.py doctor 查看完整环境诊断。",
            ],
            details=details,
            show_doctor_hint=True,
        )

    if any(signal in lowered for signal in ["local model not found", "hf_hub_offline", "cannot find", "does not appear to have a file named"]):
        return FriendlyError(
            title="本地 Embedding 模型不可用",
            message="项目当前按离线方式加载 embedding 模型，但本地模型文件或缓存不可用。",
            suggestions=[
                "确认 embedding 模型已提前下载到本机。",
                "或将 config.yaml 中的 embedding_model 改成可用的本地模型路径。",
                "运行 python main.py doctor 查看模型检查详情。",
            ],
            details=details,
            show_doctor_hint=True,
        )

    if "model not found" in lowered:
        model = _setting(settings, "llm_model", "<model>")
        return FriendlyError(
            title="Ollama 模型未下载",
            message=f"配置中的 LLM 模型不可用：{model}",
            suggestions=[
                f"运行 ollama pull {model} 下载模型。",
                "确认 config.yaml 中的 llm_model 拼写正确。",
            ],
            details=details,
            show_doctor_hint=False,
        )

    if any(signal in lowered for signal in ["collection is empty", "collection", "persist_directory", "no such table", "does not exist"]):
        return FriendlyError(
            title="向量库未构建或为空",
            message="当前知识库不可用，系统无法从论文中检索证据。",
            suggestions=[
                "确认 config.yaml 中的 persist_directory 和 collection_name 是否正确。",
                "运行 python main.py build 构建或刷新向量库。",
                "运行 python main.py doctor 检查 Chroma collection 状态。",
            ],
            details=details,
            show_doctor_hint=True,
        )

    if "rerank unavailable" in lowered or "reranker" in lowered:
        return FriendlyError(
            title="精排模型不可用",
            message="精排模型当前不可用，系统已使用原始检索顺序继续运行。",
            suggestions=[
                "确认 config.yaml 中的 reranker_model 路径是否存在。",
                "如果暂时不需要精排，可关闭 enable_rerank。",
            ],
            details=details,
            show_doctor_hint=False,
            severity="warning",
        )

    return FriendlyError(
        title="运行时错误",
        message="系统运行过程中出现未分类错误。",
        suggestions=[
            "运行 python main.py doctor 查看环境诊断。",
            "如果问题持续出现，请展开技术详情查看原始错误。",
        ],
        details=details,
        show_doctor_hint=True,
    )


def render_streamlit_error(st_module: Any, friendly_error: FriendlyError) -> None:
    if friendly_error.severity == "warning":
        st_module.warning(friendly_error.title)
    else:
        st_module.error(friendly_error.title)

    st_module.markdown(friendly_error.message)
    if friendly_error.suggestions:
        st_module.markdown("**建议下一步：**")
        for suggestion in friendly_error.suggestions:
            st_module.markdown(f"- {suggestion}")
    if friendly_error.show_doctor_hint and not any("python main.py doctor" in item for item in friendly_error.suggestions):
        st_module.markdown("- 运行 python main.py doctor 查看完整环境诊断。")
    if friendly_error.details:
        with st_module.expander("技术详情"):
            st_module.markdown(f"`{friendly_error.details}`")

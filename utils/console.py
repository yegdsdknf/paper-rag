import os
import sys


def configure_runtime_env() -> None:
    """统一配置运行时环境，避免 Hugging Face 默认联网和过多告警。"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def ensure_utf8_console() -> None:
    """尽量把当前进程的标准输出/错误切到 UTF-8。"""
    configure_runtime_env()
    if os.name == "nt":
        # 让当前 Python 进程优先使用 UTF-8 输出，避免 emoji 在 Windows 终端里报编码错误。
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        try:
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

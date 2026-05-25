import os
import sys


def ensure_utf8_console() -> None:
    """尽量把当前进程的标准输出/错误切到 UTF-8。"""
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

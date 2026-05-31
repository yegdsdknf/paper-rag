from __future__ import annotations

from typing import Any, Mapping


def get_setting(settings: Any, key: str, default: Any = None) -> Any:
    """兼容 dict 与 typed settings，便于包内模块渐进迁移配置对象。"""
    if settings is None:
        return default
    if isinstance(settings, Mapping):
        return settings.get(key, default)
    return getattr(settings, key, default)

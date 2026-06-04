from __future__ import annotations

from typing import Any, Callable, MutableMapping


LLMCacheKey = tuple[str, float, int, int]


def select_embedding_device(torch_module: Any | None) -> str:
    if torch_module is not None and torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def get_cached_llm(
    cache: MutableMapping[LLMCacheKey, Any | None],
    create_llm_fn: Callable[..., Any],
    *,
    model: str,
    temperature: float,
    num_ctx: int,
    num_predict: int,
    ping_prompt: str = "ping",
    on_success: Callable[[str], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> Any | None:
    key = (model, temperature, num_ctx, num_predict)
    if key in cache:
        return cache[key]

    llm = create_llm_fn(
        model=model,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )
    try:
        llm.invoke(ping_prompt)
        if on_success is not None:
            on_success(model)
        cache[key] = llm
    except Exception as exc:
        if on_error is not None:
            on_error(exc)
        cache[key] = None
    return cache[key]

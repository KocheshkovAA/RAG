"""Учёт токенов через общий callback-handler на весь запрос."""

from langchain_core.callbacks import UsageMetadataCallbackHandler


def new_usage_handler() -> UsageMetadataCallbackHandler:
    return UsageMetadataCallbackHandler()


def summarize_usage(handler: UsageMetadataCallbackHandler) -> dict:
    by_model = {model: dict(usage) for model, usage in handler.usage_metadata.items()}
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for usage in by_model.values():
        for key in total:
            total[key] += usage.get(key, 0) or 0
    return {"total": total, "by_model": by_model}

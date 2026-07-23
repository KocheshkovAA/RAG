"""Общая часть для логирования конфига eval-прогонов (не Postgres — обычный
JSON-сайдкар рядом с результатами). Без этого через полгода нечем объяснить,
почему цифры изменились: сменили промпт? модель? датасет?
Используется и evaluate_routes.py, и evaluate_retrieval.py."""

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def write_manifest(results_dir: Path, run_id: str, **fields) -> Path:
    """Пишет app/eval/results/run_<run_id>.json с run_id/started_at/git_commit
    плюс произвольные дополнительные поля (dataset_path, roles, summaries и т.д.
    — разные у разных eval-скриптов, поэтому просто **fields)."""
    manifest = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        **fields,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = results_dir / f"run_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path

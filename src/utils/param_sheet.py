from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any


TEMPLATE_ROWS = [
    {"key": "websearch_enabled", "value": "true", "enabled": "1", "notes": "Enable/disable web search"},
    {"key": "websearch_top_n", "value": "5", "enabled": "1", "notes": "How many top candidates to enrich"},
    {"key": "top_k_bayes", "value": "20", "enabled": "1", "notes": "Bayes top-k candidates"},
    {"key": "top_k_screen", "value": "10", "enabled": "1", "notes": "AI screening top-k"},
    {"key": "db_candidate_workers", "value": "8", "enabled": "1", "notes": "Parallel workers across candidate dedup"},
    {"key": "samples", "value": "100", "enabled": "1", "notes": "Bayesian sample size"},
    {"key": "n_structures", "value": "5", "enabled": "1", "notes": "Structures generated per material"},
    {"key": "relax_timeout_sec", "value": "120", "enabled": "1", "notes": "Relaxation timeout per task"},
    {"key": "skip_doc_update", "value": "false", "enabled": "1", "notes": "Skip theory update step"},
    {"key": "agentos_default_iterations", "value": "3", "enabled": "1", "notes": "Default iterations for AgentOS"},
]


def ensure_param_sheet(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        return p

    with p.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "value", "enabled", "notes"])
        writer.writeheader()
        writer.writerows(TEMPLATE_ROWS)
    return p


def _parse_value(raw: str, default: Any) -> Any:
    text = (raw or "").strip()
    if text == "":
        return default

    if isinstance(default, bool):
        return text.lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(text))
    if isinstance(default, float):
        return float(text)
    if isinstance(default, list):
        # Accept JSON array or Python literal list, fallback to comma-separated.
        try:
            value = json.loads(text)
            if isinstance(value, list):
                return value
        except Exception:
            pass
        try:
            value = ast.literal_eval(text)
            if isinstance(value, list):
                return value
        except Exception:
            pass
        return [item.strip() for item in text.split(",") if item.strip()]
    if default is None:
        low = text.lower()
        if low in {"none", "null"}:
            return None
    return text


def load_param_overrides(path: str | Path, base_config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    p = Path(path)
    if not p.exists():
        return {}, []

    overrides: dict[str, Any] = {}
    warnings: list[str] = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"key", "value"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            warnings.append(f"Invalid param sheet header in {p}. Required: key,value")
            return {}, warnings

        for row in reader:
            key = (row.get("key") or "").strip()
            if not key:
                continue
            if key not in base_config:
                warnings.append(f"Unknown config key skipped: {key}")
                continue
            try:
                overrides[key] = _parse_value(row.get("value") or "", base_config[key])
            except Exception as exc:
                warnings.append(f"Failed to parse key={key}: {exc}")
    return overrides, warnings


def load_param_prefill(path: str | Path, base_config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Load all known key-values from CSV for UI prefill (ignores enabled column)."""
    p = Path(path)
    if not p.exists():
        return {}, []

    values: dict[str, Any] = {}
    warnings: list[str] = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"key", "value"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            warnings.append(f"Invalid param sheet header in {p}. Required: key,value")
            return {}, warnings

        for row in reader:
            key = (row.get("key") or "").strip()
            if not key:
                continue
            if key not in base_config:
                continue
            try:
                values[key] = _parse_value(row.get("value") or "", base_config[key])
            except Exception as exc:
                warnings.append(f"Failed to parse key={key}: {exc}")
    return values, warnings


def _stringify_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return ",".join(str(v) for v in value)
    return str(value)


def persist_param_values(
    path: str | Path,
    values: dict[str, Any],
    keys: list[str] | None = None,
    enable_for_new_keys: bool = False,
) -> tuple[int, list[str]]:
    p = ensure_param_sheet(path)
    target_keys = set(keys) if keys else set(values.keys())
    warnings: list[str] = []

    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or ["key", "value", "enabled", "notes"]

    existing_keys = set()
    updated = 0
    for row in rows:
        key = (row.get("key") or "").strip()
        if not key:
            continue
        existing_keys.add(key)
        if key not in target_keys or key not in values:
            continue
        row["value"] = _stringify_value(values.get(key))
        updated += 1

    for key in sorted(target_keys):
        if key in existing_keys or key not in values:
            continue
        rows.append(
            {
                "key": key,
                "value": _stringify_value(values.get(key)),
                "enabled": "1" if enable_for_new_keys else "0",
                "notes": "Auto-added by runtime memory",
            }
        )
        updated += 1

    try:
        with p.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:
        warnings.append(f"Failed to persist param sheet {p}: {exc}")
        return 0, warnings

    return updated, warnings

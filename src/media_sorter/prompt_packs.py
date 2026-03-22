from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml

from .config import DEFAULT_LEVEL_PROMPTS, DEFAULT_SUBJECT_PROMPTS


PromptMap = dict[str, str]


@dataclass(slots=True)
class PromptPack:
    level_prompts: PromptMap | None = None
    subject_prompts: PromptMap | None = None


def load_prompt_pack(path: Path | str, kind: str | None = None) -> PromptMap | PromptPack:
    payload = _load_mapping_file(path)
    if kind is None:
        return normalize_prompt_pack(payload)
    return normalize_prompt_payload(payload, kind=kind)


def merge_prompt_overrides(base: PromptMap, override: PromptMap | None) -> PromptMap:
    merged = dict(base)
    if override:
        merged.update({str(k): str(v) for k, v in override.items()})
    return merged


def resolve_level_prompts(
    *,
    inline_json: str | None = None,
    prompts_path: Path | str | None = None,
) -> PromptMap:
    prompts = dict(DEFAULT_LEVEL_PROMPTS)
    if prompts_path:
        prompts = merge_prompt_overrides(prompts, load_prompt_pack(prompts_path, kind="category"))
    if inline_json:
        prompts = merge_prompt_overrides(prompts, _coerce_prompt_map(json.loads(inline_json), "--prompts"))
    return prompts


def resolve_subject_prompts(
    *,
    inline_json: str | None = None,
    prompts_path: Path | str | None = None,
) -> PromptMap:
    prompts = dict(DEFAULT_SUBJECT_PROMPTS)
    if prompts_path:
        prompts = merge_prompt_overrides(prompts, load_prompt_pack(prompts_path, kind="subject"))
    if inline_json:
        prompts = merge_prompt_overrides(
            prompts, _coerce_prompt_map(json.loads(inline_json), "--subject-prompts")
        )
    return prompts


def normalize_prompt_pack(payload: Any) -> PromptPack:
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Prompt pack must be a non-empty object")

    level_prompts: PromptMap | None = None
    subject_prompts: PromptMap | None = None

    if any(key in payload for key in ("level_prompts", "category_prompts", "prompts")):
        category_candidate = (
            payload.get("level_prompts")
            or payload.get("category_prompts")
            or payload.get("prompts")
        )
        if category_candidate is not None:
            level_prompts = _coerce_prompt_map(category_candidate, "category prompt pack")

    if any(key in payload for key in ("subject_prompts", "subjects")):
        subject_candidate = payload.get("subject_prompts") or payload.get("subjects")
        if subject_candidate is not None:
            subject_prompts = _coerce_prompt_map(subject_candidate, "subject prompt pack")

    if level_prompts is None and subject_prompts is None:
        # Backward-compatible single-map packs default to category prompts.
        level_prompts = _coerce_prompt_map(payload, "category prompt pack")

    return PromptPack(level_prompts=level_prompts, subject_prompts=subject_prompts)


def normalize_prompt_payload(payload: Any, kind: str) -> PromptMap:
    if kind not in {"category", "subject"}:
        raise ValueError("kind must be 'category' or 'subject'")

    pack = normalize_prompt_pack(payload)
    if kind == "category":
        if not pack.level_prompts:
            raise ValueError("Prompt pack does not contain category prompts")
        return pack.level_prompts

    if not pack.subject_prompts:
        raise ValueError("Prompt pack does not contain subject prompts")
    return pack.subject_prompts


def _load_mapping_file(path: Path | str) -> Any:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise ValueError(f"Prompt pack file not found: {prompt_path}")

    raw = prompt_path.read_text(encoding="utf-8")
    suffix = prompt_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(raw)
    if suffix == ".json":
        return json.loads(raw)
    raise ValueError("Prompt pack files must use .json, .yaml, or .yml")


def _coerce_prompt_map(value: Any, source: str) -> PromptMap:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{source} must be a non-empty object")
    return {str(k): str(v) for k, v in value.items()}

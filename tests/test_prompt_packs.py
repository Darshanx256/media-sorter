from __future__ import annotations

import json
from pathlib import Path

from media_sorter.prompt_packs import (
    PromptPack,
    load_prompt_pack,
    resolve_level_prompts,
    resolve_subject_prompts,
)


def test_load_prompt_pack_supports_unified_yaml_pack(tmp_path: Path) -> None:
    pack_path = tmp_path / "prompts.yaml"
    pack_path.write_text(
        "level_prompts:\n"
        "  portrait: a portrait photo\n"
        "subject_prompts:\n"
        "  person: a person\n"
        "  pet: a pet\n"
        "  other: other\n",
        encoding="utf-8",
    )

    pack = load_prompt_pack(pack_path)
    assert isinstance(pack, PromptPack)
    assert pack.level_prompts == {"portrait": "a portrait photo"}
    assert pack.subject_prompts == {"person": "a person", "pet": "a pet", "other": "other"}


def test_resolve_level_prompts_prefers_inline_over_file_over_defaults(tmp_path: Path) -> None:
    pack_path = tmp_path / "categories.json"
    pack_path.write_text(json.dumps({"portrait": "file portrait", "travel": "file travel"}), encoding="utf-8")

    prompts = resolve_level_prompts(
        prompts_path=pack_path,
        inline_json=json.dumps({"portrait": "inline portrait"}),
    )

    assert prompts["portrait"] == "inline portrait"
    assert prompts["travel"] == "file travel"
    assert "document" in prompts


def test_resolve_subject_prompts_supports_subject_only_pack(tmp_path: Path) -> None:
    pack_path = tmp_path / "subjects.yaml"
    pack_path.write_text(
        "subject_prompts:\n"
        "  person: a human being\n"
        "  pet: a domestic pet\n"
        "  other: a non-person scene\n",
        encoding="utf-8",
    )

    prompts = resolve_subject_prompts(prompts_path=pack_path)
    assert prompts["person"] == "a human being"
    assert prompts["pet"] == "a domestic pet"

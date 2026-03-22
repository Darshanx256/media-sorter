from __future__ import annotations

from media_sorter.cli import build_doctor_parser, build_finalize_parser, build_parser


def test_main_help_mentions_prompt_pack_files() -> None:
    help_text = build_parser().format_help()
    assert "--prompts-file" in help_text
    assert "--subject-prompts-file" in help_text


def test_finalize_help_mentions_image_first_bundle() -> None:
    help_text = build_finalize_parser().format_help()
    assert "image-first" in help_text
    assert "--prompts-file" in help_text


def test_doctor_help_mentions_environment_checks() -> None:
    help_text = build_doctor_parser().format_help()
    assert "--expect-video" in help_text
    assert "--expect-finalize" in help_text

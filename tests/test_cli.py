from __future__ import annotations

import os

os.environ.setdefault("TERMINAL_WIDTH", "200")

from typer.testing import CliRunner

from media_sorter.cli import app


def test_main_help_mentions_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Commands" in result.stdout
    assert "analyze" in result.stdout


def test_analyze_help_mentions_prompt_pack_files() -> None:
    result = CliRunner().invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "--prompts-file" in result.stdout
    assert "--subject-prompts-file" in result.stdout


def test_finalize_help_mentions_image_first_bundle() -> None:
    result = CliRunner().invoke(app, ["finalize", "--help"])
    assert result.exit_code == 0
    assert "image-first" in result.stdout
    assert "--prompts-file" in result.stdout


def test_doctor_help_mentions_environment_checks() -> None:
    result = CliRunner().invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "--expect-video" in result.stdout
    assert "--expect-finalize" in result.stdout

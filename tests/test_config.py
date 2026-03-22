from __future__ import annotations

from pathlib import Path

import pytest

from media_sorter.config import DEFAULT_LEVEL_PROMPTS, SorterConfig


def test_default_level_prompts_are_real_gallery_categories() -> None:
    assert "portrait" in DEFAULT_LEVEL_PROMPTS
    assert "document" in DEFAULT_LEVEL_PROMPTS
    assert "screenshot" in DEFAULT_LEVEL_PROMPTS


def test_analysis_only_mode_does_not_require_output_dir(tmp_path: Path) -> None:
    config = SorterConfig(source_dir=tmp_path, output_dir=None, copy_mode="none")
    assert config.output_dir is None


def test_copy_mode_requires_output_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_dir is required"):
        SorterConfig(source_dir=tmp_path, output_dir=None, copy_mode="copy")


def test_invalid_copy_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="copy_mode must be"):
        SorterConfig(source_dir=tmp_path, output_dir=tmp_path / "out", copy_mode="zip")


def test_invalid_confidence_range_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_category_confidence"):
        SorterConfig(
            source_dir=tmp_path,
            output_dir=None,
            min_category_confidence=2.0,
        )

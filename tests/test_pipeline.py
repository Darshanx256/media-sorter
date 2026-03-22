from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from media_sorter.config import SorterConfig
from media_sorter.pipeline import MediaAnalyzer, MediaSorter

from conftest import FakeClassifier


def test_analyzer_smoke_writes_manifest(source_dir: Path) -> None:
    config = SorterConfig(
        source_dir=source_dir,
        output_dir=None,
        write_manifest=True,
        manifest_format="jsonl",
    )
    analyzer = MediaAnalyzer(config, classifier=FakeClassifier(config))
    stats = analyzer.run()

    assert stats.total_seen == 1
    assert analyzer.records[0].status == "ok"
    assert analyzer.records[0].metadata.width == 64
    assert analyzer.manifest_output_path is not None and analyzer.manifest_output_path.exists()


def test_index_update_skips_unchanged_file(source_dir: Path) -> None:
    config = SorterConfig(
        source_dir=source_dir,
        output_dir=None,
        enable_index=True,
        index_mode="update",
    )
    MediaAnalyzer(config, classifier=FakeClassifier(config)).run()
    second = MediaAnalyzer(config, classifier=FakeClassifier(config))
    stats = second.run()

    assert stats.total_seen == 1
    assert stats.skipped == 1
    assert [record.status for record in second.records] == ["skipped"]


def test_sorter_copy_mode_writes_output_and_manifest(source_dir: Path, sample_image: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "sorted"
    config = SorterConfig(
        source_dir=source_dir,
        output_dir=output_dir,
        copy_mode="copy",
        write_manifest=True,
        manifest_format="json",
    )
    sorter = MediaSorter(config, classifier=FakeClassifier(config))
    stats = sorter.run()

    expected_output = output_dir / next(iter(config.level_prompts)) / sample_image.name
    assert stats.total_seen == 1
    assert expected_output.exists()
    assert sorter.manifest_output_path is not None and sorter.manifest_output_path.exists()
    assert sorter.records[0].route_label == next(iter(config.level_prompts))


def test_video_missing_dependency_yields_error_record(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"not-a-real-video")
    config = SorterConfig(source_dir=tmp_path, output_dir=None)
    analyzer = MediaAnalyzer(config, classifier=FakeClassifier(config))

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cv2":
            raise ImportError("No module named 'cv2'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    record = analyzer.analyze_file(video_path)

    assert record.status == "error"
    assert "opencv-python-headless" in (record.error or "")


def test_face_missing_dependency_raises_clear_error(monkeypatch, source_dir: Path) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cv2":
            raise ImportError("No module named 'cv2'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    config = SorterConfig(source_dir=source_dir, output_dir=None, enable_face_sorting=True)
    with pytest.raises(RuntimeError, match="opencv-python-headless"):
        MediaAnalyzer(config, classifier=FakeClassifier(config))


@pytest.mark.optional
def test_video_sampling_smoke_when_cv2_is_available(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    video_path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (32, 32),
    )
    try:
        import numpy as np

        for value in (30, 120, 220):
            frame = np.full((32, 32, 3), value, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    if not video_path.exists() or video_path.stat().st_size == 0:
        pytest.skip("cv2 video writer codec unavailable in this environment")

    config = SorterConfig(source_dir=tmp_path, output_dir=None, max_video_frames=2)
    analyzer = MediaAnalyzer(config, classifier=FakeClassifier(config))
    record = analyzer.analyze_file(video_path)

    assert record.status == "ok"
    assert record.media_type == "video"
    assert len(record.sampled_frames) >= 1


@pytest.mark.optional
def test_face_sorting_smoke_when_cv2_is_available(source_dir: Path) -> None:
    pytest.importorskip("cv2")
    config = SorterConfig(source_dir=source_dir, output_dir=None, enable_face_sorting=True)
    analyzer = MediaAnalyzer(config, classifier=FakeClassifier(config))
    record = analyzer.analyze_file(source_dir / "sample.jpg")

    assert record.status == "ok"
    assert record.face_identity is None or isinstance(record.face_identity, str)

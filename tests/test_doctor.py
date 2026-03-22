from __future__ import annotations

from media_sorter.doctor import collect_dependency_status, render_doctor_report


def test_collect_dependency_status_covers_expected_modules() -> None:
    names = {status.name for status in collect_dependency_status()}
    assert {"torch", "open_clip", "onnx", "onnxruntime", "cv2"} <= names


def test_render_doctor_report_includes_feature_sections() -> None:
    report = render_doctor_report(expect_video=True, expect_finalize=True)
    assert "feature_readiness:" in report
    assert "ml_backend:" in report
    assert "finalize_bundle:" in report
    assert "video_runtime:" in report

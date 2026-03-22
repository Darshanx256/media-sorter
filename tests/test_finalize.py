from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from media_sorter.config import SorterConfig
from media_sorter.finalize import BundleFinalizer, BundleValidationResult


def test_finalize_generation_can_be_unit_tested_without_real_export(monkeypatch, tmp_path: Path) -> None:
    class DummyClassifier:
        preprocess = object()
        model = object()

        def __init__(self, config) -> None:
            self.config = config

    monkeypatch.setattr("media_sorter.finalize.MediaClassifier", DummyClassifier)
    monkeypatch.setattr(BundleFinalizer, "_ensure_export_deps", lambda self: None)
    monkeypatch.setattr(
        BundleFinalizer,
        "_extract_preprocess_config",
        lambda self, preprocess: {
            "resize_size": 224,
            "crop_size": 224,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
    )
    monkeypatch.setattr(
        BundleFinalizer,
        "_build_text_embeddings",
        lambda self, classifier: {
            "subject": np.ones((3, 4), dtype=np.float32),
            "count": np.ones((3, 4), dtype=np.float32),
            "category": np.ones((len(self.config.level_prompts), 4), dtype=np.float32),
        },
    )
    monkeypatch.setattr(BundleFinalizer, "_export_image_encoder", lambda self, classifier, path, cfg: path.write_text("onnx"))
    monkeypatch.setattr(BundleFinalizer, "_quantize_model", lambda self, src, dst: dst.write_text("quantized"))
    monkeypatch.setattr(
        BundleFinalizer,
        "validate_bundle",
        lambda self, artifacts, check_runner=True: BundleValidationResult(
            ok=True,
            checked_files=[],
            runner_checked=check_runner,
            errors=[],
        ),
    )

    config = SorterConfig(source_dir=tmp_path, output_dir=None)
    artifacts = BundleFinalizer(config).finalize(tmp_path / "bundle")
    assert artifacts.model_path.exists()
    assert artifacts.quantized_model_path is not None
    assert artifacts.config_path.exists()
    assert artifacts.embeddings_path.exists()
    assert artifacts.runner_path.exists()


@pytest.mark.ml
def test_finalize_real_smoke_when_deps_are_available(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("open_clip")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    config = SorterConfig(source_dir=tmp_path, output_dir=None, device="cpu")
    bundle_dir = tmp_path / "bundle"
    artifacts = BundleFinalizer(config).finalize(bundle_dir)

    assert artifacts.model_path.exists()
    assert artifacts.config_path.exists()
    assert artifacts.embeddings_path.exists()
    assert artifacts.requirements_path.exists()
    assert artifacts.quantized_model_path is not None and artifacts.quantized_model_path.exists()
    assert artifacts.runner_path.exists()

    config_payload = json.loads(artifacts.config_path.read_text(encoding="utf-8"))
    assert config_payload["features"]["video_analysis"] is False
    assert config_payload["features"]["face_sorting"] is False

    smoke_image = tmp_path / "runner_smoke.png"
    Image.new("RGB", (256, 256), color=(100, 140, 210)).save(smoke_image)
    completed = subprocess.run(
        [sys.executable, str(artifacts.runner_path), str(smoke_image)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert "subject_scores" in payload
    assert "category_scores" in payload

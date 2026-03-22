from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from media_sorter.config import SorterConfig
from media_sorter.core import Prediction


class FakeClassifier:
    def __init__(self, config: SorterConfig) -> None:
        self.config = config

    def predict_image(self, file_path: Path) -> Prediction:
        return self._prediction(file_path)

    def predict_pil(self, image: Image.Image) -> Prediction:
        return self._prediction(None)

    def embedding(self, image: Image.Image) -> np.ndarray:
        vec = np.array([1.0, 0.5, 0.25], dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def _prediction(self, file_path: Path | None) -> Prediction:
        categories = list(self.config.level_prompts)
        category_scores = {label: 0.01 for label in categories}
        first = categories[0]
        category_scores[first] = 0.91
        return Prediction(
            file_path=file_path,
            subject="person",
            subject_confidence=0.92,
            subject_scores={"person": 0.92, "pet": 0.04, "other": 0.04},
            is_solo_person=True,
            solo_label=self.config.count_labels[0],
            solo_confidence=0.88,
            count_scores={
                self.config.count_labels[0]: 0.88,
                self.config.count_labels[1]: 0.05,
                self.config.count_labels[2]: 0.07,
            },
            category=first,
            category_confidence=0.91,
            category_scores=category_scores,
        )


@pytest.fixture()
def source_dir(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (64, 48), color=(128, 180, 220)).save(source / "sample.jpg")
    return source


@pytest.fixture()
def sample_image(source_dir: Path) -> Path:
    return source_dir / "sample.jpg"


@pytest.fixture()
def fake_classifier(source_dir: Path) -> FakeClassifier:
    config = SorterConfig(source_dir=source_dir, output_dir=None)
    return FakeClassifier(config)

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import sys

import numpy as np
from PIL import Image

from .config import SorterConfig


@dataclass(slots=True)
class Prediction:
    file_path: Path | None
    subject: str
    subject_confidence: float
    subject_scores: dict[str, float]
    is_solo_person: bool
    solo_label: str
    solo_confidence: float
    count_scores: dict[str, float]
    category: str | None
    category_confidence: float
    category_scores: dict[str, float]


class MediaClassifier:
    def __init__(self, config: SorterConfig) -> None:
        self._backend = ""
        try:
            import torch
            self._torch = torch

            # Preferred backend: open-clip-torch (avoids deprecated pkg_resources path).
            try:
                import open_clip

                model_name = self._normalize_open_clip_model_name(config.model_name)
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name,
                    pretrained="openai",
                    device=config.device,
                    force_quick_gelu=True,
                )
                self.model.eval()
                self._tokenize = open_clip.get_tokenizer(model_name)
                self._backend = "open_clip"
            except (ImportError, ModuleNotFoundError):
                # Fallback to clip-anytorch only when open_clip is genuinely
                # not installed. Any other error (OOM, download failure, etc.)
                # should surface to the user rather than silently switch backends.
                import clip

                self.model, self.preprocess = clip.load(config.model_name, device=config.device)
                self._tokenize = clip.tokenize
                self._backend = "clip_anytorch"
        except Exception as exc:
            raise RuntimeError(self._build_ml_setup_error_message(exc, stage="import")) from exc

        self.config = config

        self._subject_labels = list(config.subject_prompts.keys())
        self._subject_tokens = self._tokenize([config.subject_prompts[label] for label in self._subject_labels]).to(
            config.device
        )

        self._count_tokens = self._tokenize(list(config.count_labels)).to(config.device)
        self._category_labels = list(config.level_prompts.keys())
        self._category_tokens = self._tokenize([config.level_prompts[label] for label in self._category_labels]).to(
            config.device
        )

    @property
    def category_labels(self) -> list[str]:
        return list(self._category_labels)

    def score_image_prompts(self, file_path: Path, prompts: dict[str, str]) -> dict[str, float]:
        with Image.open(file_path) as image:
            rgb = image.convert("RGB")
        return self.score_pil_prompts(rgb, prompts)

    def score_pil_prompts(self, image: Image.Image, prompts: dict[str, str]) -> dict[str, float]:
        if not prompts:
            return {}

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.config.device)
        labels = list(prompts.keys())
        tokens = self._tokenize([prompts[label] for label in labels]).to(self.config.device)
        with self._torch.no_grad():
            logits = self._image_text_logits(image_tensor, tokens)
            probs = logits.softmax(dim=-1).cpu().numpy()[0]
        return {label: float(probs[idx]) for idx, label in enumerate(labels)}

    def predict_image(self, file_path: Path) -> Prediction:
        with Image.open(file_path) as image:
            rgb = image.convert("RGB")
        prediction = self.predict_pil(rgb)
        prediction.file_path = file_path
        return prediction

    def predict_pil(self, image: Image.Image) -> Prediction:
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.config.device)

        with self._torch.no_grad():
            subject_logits = self._image_text_logits(image_tensor, self._subject_tokens)
            subject_probs = subject_logits.softmax(dim=-1).cpu().numpy()[0]
            subject_scores = {
                label: float(subject_probs[idx]) for idx, label in enumerate(self._subject_labels)
            }
            subject_idx = int(np.argmax(subject_probs))
            subject = self._subject_labels[subject_idx]
            subject_conf = float(subject_probs[subject_idx])
            person_idx = self._subject_labels.index("person") if "person" in self._subject_labels else None
            person_prob = float(subject_probs[person_idx]) if person_idx is not None else 0.0

            count_logits = self._image_text_logits(image_tensor, self._count_tokens)
            count_probs = count_logits.softmax(dim=-1).cpu().numpy()[0]
            count_scores = {
                label: float(count_probs[idx]) for idx, label in enumerate(self.config.count_labels)
            }

            solo_idx = int(np.argmax(count_probs))
            solo_label = self.config.count_labels[solo_idx]
            solo_prob = float(count_probs[0])
            non_solo_prob = float(max(count_probs[1], count_probs[2]))
            is_solo = (
                subject == "person"
                and person_prob >= self.config.min_person_confidence
                and solo_prob >= self.config.min_solo_confidence
                and (solo_prob - non_solo_prob) >= self.config.min_solo_margin
            )

            cat_logits = self._image_text_logits(image_tensor, self._category_tokens)
            cat_probs = cat_logits.softmax(dim=-1).cpu().numpy()[0]
            category_scores = {
                label: float(cat_probs[idx]) for idx, label in enumerate(self._category_labels)
            }
            cat_idx = int(np.argmax(cat_probs))
            category_label = self._category_labels[cat_idx]
            category_conf = float(cat_probs[cat_idx])

        return Prediction(
            file_path=None,
            subject=subject,
            subject_confidence=subject_conf,
            subject_scores=subject_scores,
            is_solo_person=is_solo,
            solo_label=solo_label,
            solo_confidence=solo_prob,
            count_scores=count_scores,
            category=category_label,
            category_confidence=category_conf,
            category_scores=category_scores,
        )

    def embedding(self, image: Image.Image) -> np.ndarray:
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.config.device)
        with self._torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy()

    def _image_text_logits(self, image_tensor, text_tokens):
        if self._backend == "clip_anytorch":
            logits, _ = self.model(image_tensor, text_tokens)
            return logits

        image_features = self.model.encode_image(image_tensor)
        text_features = self.model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return 100.0 * image_features @ text_features.T

    @staticmethod
    def _normalize_open_clip_model_name(model_name: str) -> str:
        mapped = {
            "ViT-B/32": "ViT-B-32",
            "ViT-B/16": "ViT-B-16",
            "ViT-L/14": "ViT-L-14",
        }
        return mapped.get(model_name, model_name.replace("/", "-"))

    @staticmethod
    def _build_ml_setup_error_message(exc: Exception, stage: str, device: str | None = None) -> str:
        missing_name = getattr(exc, "name", "") or "unknown module"
        exc_name = type(exc).__name__
        exc_text = str(exc)
        full_text = f"{exc_name}: {exc_text}"
        python_bin = shlex.quote(sys.executable)
        pip_cmd = f"{python_bin} -m pip"
        venv = os.environ.get("VIRTUAL_ENV")
        venv_hint = (
            f"Active venv: {venv}\n"
            if venv
            else "No active VIRTUAL_ENV detected. Ensure you run with your intended venv python.\n"
        )
        base = (
            "ML dependencies are not ready.\n"
            f"Failure stage: {stage}\n"
            f"Detected error: {full_text}\n"
            f"Missing import name (if any): {missing_name}\n\n"
            f"Python in use: {sys.executable}\n"
            + venv_hint
            + "\n"
            "Try these checks first:\n"
            f"1) {python_bin} -c \"import sys; print(sys.version); print(sys.executable)\"\n"
            f"2) {pip_cmd} show torch torchvision clip-anytorch setuptools\n"
        )

        if missing_name == "pkg_resources" or "pkg_resources" in exc_text:
            return (
                base
                + "\nDetected missing 'pkg_resources' from the clip-anytorch path.\n"
                "Preferred fix is to install open-clip-torch and avoid clip-anytorch entirely.\n"
                "Run:\n"
                f"  {pip_cmd} install open-clip-torch\n"
                "Then verify:\n"
                f"  {python_bin} -c \"import open_clip, torch; print('ok')\"\n"
            )

        if "operator torchvision::nms does not exist" in exc_text:
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            py_version_note = (
                "\nDetected Python >= 3.13. Many torch/torchvision wheels may be unavailable or unstable here.\n"
                "If reinstall below still fails, create a Python 3.11/3.12 venv and install there.\n"
                if sys.version_info >= (3, 13)
                else ""
            )
            return (
                base
                + "\nDetected torch/torchvision binary mismatch.\n"
                f"Current Python version: {py_version}\n"
                + py_version_note
                + "Repair in current interpreter:\n"
                f"  {pip_cmd} uninstall -y torchvision torch torchaudio\n"
                f"  {pip_cmd} cache purge\n"
                f"  {pip_cmd} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
                f"  {pip_cmd} install --force-reinstall clip-anytorch --no-deps\n"
                "Then verify:\n"
                f"  {python_bin} -c \"import torch, torchvision, clip; print(torch.__version__, torchvision.__version__)\"\n"
            )

        if missing_name == "torch" or "No module named 'torch'" in exc_text:
            return (
                base
                + "\nTorch is missing.\n"
                "CPU example:\n"
                f"  {pip_cmd} install torch --index-url https://download.pytorch.org/whl/cpu\n"
                f"  {pip_cmd} install clip-anytorch --no-deps\n"
            )

        if missing_name == "open_clip" or "No module named 'open_clip'" in exc_text:
            return (
                base
                + "\nopen-clip-torch is missing.\n"
                "Run:\n"
                f"  {pip_cmd} install open-clip-torch\n"
            )

        if missing_name == "clip" or "No module named 'clip'" in exc_text:
            return (
                base
                + "\nFallback clip-anytorch backend is missing.\n"
                "Preferred install:\n"
                f"  {pip_cmd} install open-clip-torch\n"
            )

        if missing_name in {"tqdm", "ftfy", "regex"}:
            return (
                base
                + f"\nDetected missing CLIP helper dependency: {missing_name}\n"
                "Run:\n"
                f"  {pip_cmd} install ftfy regex tqdm\n"
                "Then verify:\n"
                f"  {python_bin} -c \"import ftfy, regex, tqdm, clip; print('ok')\"\n"
            )

        if stage == "model_load":
            return (
                base
                + "\nThe CLIP model failed during load.\n"
                + (
                    f"Requested device: {device}\n"
                    if device
                    else ""
                )
                + "If using GPU, ensure CUDA-compatible torch build.\n"
                + "If unsure, retry on CPU.\n"
                + "Example:\n"
                + f"  {python_bin} -m media_sorter.cli <source> <output> --device cpu\n"
            )

        return (
            base
            + "\nUnclassified setup issue.\n"
            + "Recommended reset in this interpreter:\n"
            + f"  {pip_cmd} uninstall -y clip-anytorch torchvision torch torchaudio setuptools\n"
            + f"  {pip_cmd} install \"setuptools<81\"\n"
            + f"  {pip_cmd} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
            + f"  {pip_cmd} install clip-anytorch --no-deps\n"
            + "Then retry."
        )

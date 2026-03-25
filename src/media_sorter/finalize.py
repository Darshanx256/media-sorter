from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image

from .config import SorterConfig
from .core import MediaClassifier


@dataclass(slots=True)
class FinalizeArtifacts:
    bundle_dir: Path
    model_path: Path
    quantized_model_path: Path | None
    config_path: Path
    embeddings_path: Path
    runner_path: Path
    requirements_path: Path


@dataclass(slots=True)
class BundleValidationResult:
    ok: bool
    checked_files: list[str]
    runner_checked: bool
    errors: list[str]


class BundleFinalizer:
    def __init__(self, config: SorterConfig) -> None:
        self.config = config

    def finalize(
        self,
        bundle_dir: Path | str,
        *,
        quantize: bool = True,
        include_runner: bool = True,
    ) -> FinalizeArtifacts:
        self._ensure_export_deps()

        bundle_path = Path(bundle_dir)
        bundle_path.mkdir(parents=True, exist_ok=True)

        export_config = SorterConfig(
            source_dir=self.config.source_dir,
            output_dir=self.config.output_dir,
            model_name=self.config.model_name,
            device="cpu",
            limit=self.config.limit,
            image_extensions=self.config.image_extensions,
            video_extensions=self.config.video_extensions,
            level_prompts=dict(self.config.level_prompts),
            subject_prompts=dict(self.config.subject_prompts),
            count_labels=tuple(self.config.count_labels),
            ignored_label=self.config.ignored_label,
            pet_label=self.config.pet_label,
            enable_pet_sorting=self.config.enable_pet_sorting,
            min_category_confidence=self.config.min_category_confidence,
            copy_mode=self.config.copy_mode,
            dry_run=self.config.dry_run,
            write_manifest=self.config.write_manifest,
            manifest_path=self.config.manifest_path,
            manifest_format=self.config.manifest_format,
            enable_index=self.config.enable_index,
            index_db_path=self.config.index_db_path,
            index_mode=self.config.index_mode,
            index_prune_missing=self.config.index_prune_missing,
            enable_face_sorting=self.config.enable_face_sorting,
            face_mode=self.config.face_mode,
            face_label=self.config.face_label,
            face_tags_dir=self.config.face_tags_dir,
            face_similarity_threshold=self.config.face_similarity_threshold,
            max_video_frames=self.config.max_video_frames,
            video_sampling_mode=self.config.video_sampling_mode,
            video_seconds_per_sample=self.config.video_seconds_per_sample,
            video_frame_skip=self.config.video_frame_skip,
            min_solo_frame_ratio=self.config.min_solo_frame_ratio,
            min_person_confidence=self.config.min_person_confidence,
            min_solo_confidence=self.config.min_solo_confidence,
            min_solo_margin=self.config.min_solo_margin,
        )
        classifier = MediaClassifier(export_config)

        preprocess_config = self._extract_preprocess_config(classifier.preprocess)
        embeddings = self._build_text_embeddings(classifier)

        model_path = bundle_path / "image_encoder.onnx"
        self._export_image_encoder(classifier, model_path, preprocess_config)

        quantized_model_path: Path | None = None
        if quantize:
            quantized_model_path = bundle_path / "image_encoder.int8.onnx"
            self._quantize_model(model_path, quantized_model_path)

        embeddings_path = bundle_path / "text_embeddings.npz"
        np.savez(
            embeddings_path,
            subject=embeddings["subject"],
            count=embeddings["count"],
            category=embeddings["category"],
        )

        config_path = bundle_path / "config.json"
        config_payload = self._build_bundle_config(
            preprocess_config=preprocess_config,
            quantized_model_path=quantized_model_path,
        )
        config_path.write_text(json.dumps(config_payload, ensure_ascii=True, indent=2), encoding="utf-8")

        requirements_path = bundle_path / "requirements.txt"
        requirements_path.write_text(
            "numpy>=1.25\nPillow>=10.0\nonnxruntime>=1.18\n",
            encoding="utf-8",
        )

        runner_path = bundle_path / "run_media_sort.py"
        if include_runner:
            runner_path.write_text(self._runner_script(), encoding="utf-8")

        artifacts = FinalizeArtifacts(
            bundle_dir=bundle_path,
            model_path=model_path,
            quantized_model_path=quantized_model_path,
            config_path=config_path,
            embeddings_path=embeddings_path,
            runner_path=runner_path,
            requirements_path=requirements_path,
        )
        validation = self.validate_bundle(artifacts, check_runner=include_runner)
        if not validation.ok:
            joined = "\n".join(validation.errors)
            raise RuntimeError(f"Finalize bundle validation failed:\n{joined}")
        return artifacts

    def validate_bundle(
        self,
        artifacts: FinalizeArtifacts,
        *,
        check_runner: bool = True,
    ) -> BundleValidationResult:
        errors: list[str] = []
        checked_files = [
            artifacts.model_path.name,
            artifacts.config_path.name,
            artifacts.embeddings_path.name,
            artifacts.requirements_path.name,
        ]
        if artifacts.quantized_model_path is not None:
            checked_files.append(artifacts.quantized_model_path.name)
        if artifacts.runner_path.exists():
            checked_files.append(artifacts.runner_path.name)

        for file_name in checked_files:
            if not (artifacts.bundle_dir / file_name).exists():
                errors.append(f"Missing bundle artifact: {file_name}")

        config_payload: dict[str, Any] = {}
        if artifacts.config_path.exists():
            try:
                config_payload = json.loads(artifacts.config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"Could not parse config.json: {type(exc).__name__}: {exc}")
        else:
            errors.append("Missing bundle artifact: config.json")

        runtime = config_payload.get("runtime", {})
        primary_model = runtime.get("primary_model")
        fallback_model = runtime.get("fallback_model")
        embeddings_file = runtime.get("embeddings_file")

        if not primary_model:
            errors.append("Primary model path missing from config.json")
        elif not (artifacts.bundle_dir / str(primary_model)).exists():
            # The runner falls back to fallback_model automatically, so only
            # raise an error when the fallback is also absent.
            if not fallback_model or not (artifacts.bundle_dir / str(fallback_model)).exists():
                errors.append(
                    "Neither primary nor fallback model exists in bundle "
                    f"(primary={primary_model}, fallback={fallback_model})"
                )
        if not fallback_model or not (artifacts.bundle_dir / str(fallback_model)).exists():
            errors.append("Fallback model path in config.json does not exist")
        if not embeddings_file or not (artifacts.bundle_dir / str(embeddings_file)).exists():
            errors.append("Embeddings path in config.json does not exist")

        bundle_version = config_payload.get("bundle_version")
        if not isinstance(bundle_version, int) or bundle_version < 1:
            errors.append("config.json must contain a valid integer bundle_version")

        labels = config_payload.get("labels", {})
        prompts = config_payload.get("prompts", {})
        features = config_payload.get("features", {})
        thresholds = config_payload.get("thresholds", {})

        expected_label_groups = ("subject", "count", "category")
        for group_name in expected_label_groups:
            label_values = labels.get(group_name)
            prompt_values = prompts.get(group_name)
            if not isinstance(label_values, list) or not label_values:
                errors.append(f"labels.{group_name} must be a non-empty list")
                continue
            if group_name == "count":
                if not isinstance(prompt_values, list) or len(prompt_values) != len(label_values):
                    errors.append("prompts.count must match labels.count length")
            else:
                if not isinstance(prompt_values, dict) or len(prompt_values) != len(label_values):
                    errors.append(f"prompts.{group_name} must match labels.{group_name} length")

        for key in (
            "min_person_confidence",
            "min_solo_confidence",
            "min_solo_margin",
            "min_solo_frame_ratio",
            "min_category_confidence",
        ):
            value = thresholds.get(key)
            if not isinstance(value, (int, float)):
                errors.append(f"thresholds.{key} must be numeric")

        if features.get("image_analysis") is not True:
            errors.append("features.image_analysis must be true")
        if features.get("video_analysis") not in {False, 0}:
            errors.append("features.video_analysis must be false for the current finalized runtime")
        if features.get("face_sorting") not in {False, 0}:
            errors.append("features.face_sorting must be false for the current finalized runtime")

        try:
            embeddings = np.load(artifacts.embeddings_path)
            for group_name in expected_label_groups:
                if group_name not in embeddings:
                    errors.append(f"text_embeddings.npz missing '{group_name}' array")
                    continue
                expected_rows = len(labels.get(group_name, []))
                actual_rows = int(embeddings[group_name].shape[0]) if embeddings[group_name].ndim >= 1 else 0
                if actual_rows != expected_rows:
                    errors.append(
                        f"text_embeddings.npz '{group_name}' rows ({actual_rows}) do not match labels.{group_name} ({expected_rows})"
                    )
        except Exception as exc:
            errors.append(f"Could not validate text_embeddings.npz: {type(exc).__name__}: {exc}")

        runner_checked = False
        if check_runner and artifacts.runner_path.exists():
            runner_checked = True
            errors.extend(self._validate_runner_execution(artifacts.runner_path))

        return BundleValidationResult(
            ok=not errors,
            checked_files=checked_files,
            runner_checked=runner_checked,
            errors=errors,
        )

    def _validate_runner_execution(self, runner_path: Path) -> list[str]:
        errors: list[str] = []
        with tempfile.TemporaryDirectory(prefix="media_sorter_runner_check_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            image_path = tmp_dir / "smoke.png"
            Image.new("RGB", (224, 224), color=(120, 180, 200)).save(image_path)
            completed = subprocess.run(
                [sys.executable, str(runner_path), str(image_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                stderr = completed.stderr.strip() or completed.stdout.strip()
                errors.append(f"Generated runner failed smoke test: {stderr or 'unknown error'}")
                return errors

            stdout = completed.stdout.strip().splitlines()
            if not stdout:
                errors.append("Generated runner produced no output during smoke test")
                return errors

            try:
                payload = json.loads(stdout[-1])
            except json.JSONDecodeError as exc:
                errors.append(f"Generated runner output is not valid JSON: {exc}")
                return errors

            for key in ("source", "subject", "subject_scores", "category", "category_scores"):
                if key not in payload:
                    errors.append(f"Generated runner output is missing '{key}'")
        return errors

    def _ensure_export_deps(self) -> None:
        missing: list[str] = []
        for module_name in ("torch", "onnx", "onnxruntime"):
            try:
                __import__(module_name)
            except Exception:
                missing.append(module_name)

        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                "Finalize dependencies are not installed.\n"
                f"Missing modules: {joined}\n"
                "Install the ML backend first, then add finalize extras.\n"
                "Suggested commands:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "  pip install open-clip-torch\n"
                "  pip install -e .[finalize]\n"
            )

    def _build_text_embeddings(self, classifier: MediaClassifier) -> dict[str, np.ndarray]:
        torch = classifier._torch
        with torch.no_grad():
            subject = classifier.model.encode_text(classifier._subject_tokens)
            count = classifier.model.encode_text(classifier._count_tokens)
            category = classifier.model.encode_text(classifier._category_tokens)

            subject = self._normalize_features(subject)
            count = self._normalize_features(count)
            category = self._normalize_features(category)

        return {
            "subject": subject,
            "count": count,
            "category": category,
        }

    @staticmethod
    def _normalize_features(tensor) -> np.ndarray:
        normalized = tensor / tensor.norm(dim=-1, keepdim=True)
        return normalized.cpu().numpy().astype("float32")

    def _extract_preprocess_config(self, preprocess) -> dict[str, Any]:
        resize_size: int | None = None
        crop_size: int | None = None
        mean: list[float] | None = None
        std: list[float] | None = None

        transforms = getattr(preprocess, "transforms", [])
        for transform in transforms:
            name = transform.__class__.__name__.lower()
            if "resize" in name:
                resize_size = self._coerce_size(getattr(transform, "size", None))
            elif "centercrop" in name:
                crop_size = self._coerce_size(getattr(transform, "size", None))
            elif "normalize" in name:
                mean = [float(x) for x in getattr(transform, "mean", (0.48145466, 0.4578275, 0.40821073))]
                std = [float(x) for x in getattr(transform, "std", (0.26862954, 0.26130258, 0.27577711))]

        crop_size = crop_size or resize_size or 224
        resize_size = resize_size or crop_size
        mean = mean or [0.48145466, 0.4578275, 0.40821073]
        std = std or [0.26862954, 0.26130258, 0.27577711]

        return {
            "resize_size": resize_size,
            "crop_size": crop_size,
            "mean": mean,
            "std": std,
        }

    @staticmethod
    def _coerce_size(value) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, (tuple, list)) and value:
            return int(value[0])
        return None

    def _export_image_encoder(
        self,
        classifier: MediaClassifier,
        model_path: Path,
        preprocess_config: dict[str, Any],
    ) -> None:
        import torch

        crop_size = int(preprocess_config["crop_size"])

        class ImageEncoderWrapper(torch.nn.Module):
            def __init__(self, model) -> None:
                super().__init__()
                self.model = model

            def forward(self, image):
                features = self.model.encode_image(image)
                return features / features.norm(dim=-1, keepdim=True)

        wrapper = ImageEncoderWrapper(classifier.model).cpu().eval()
        dummy = torch.randn(1, 3, crop_size, crop_size, dtype=torch.float32)
        torch.onnx.export(
            wrapper,
            dummy,
            str(model_path),
            input_names=["image"],
            output_names=["image_features"],
            opset_version=18,
            external_data=False,
        )

    @staticmethod
    def _quantize_model(model_path: Path, quantized_model_path: Path) -> None:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            str(model_path),
            str(quantized_model_path),
            weight_type=QuantType.QInt8,
        )

    def _build_bundle_config(
        self,
        *,
        preprocess_config: dict[str, Any],
        quantized_model_path: Path | None,
    ) -> dict[str, Any]:
        return {
            "bundle_version": 1,
            "runtime": {
                "primary_model": quantized_model_path.name if quantized_model_path is not None else "image_encoder.onnx",
                "fallback_model": "image_encoder.onnx",
                "embeddings_file": "text_embeddings.npz",
            },
            "preprocess": preprocess_config,
            "labels": {
                "subject": list(self.config.subject_prompts.keys()),
                "count": list(self.config.count_labels),
                "category": list(self.config.level_prompts.keys()),
            },
            "prompts": {
                "subject": dict(self.config.subject_prompts),
                "count": list(self.config.count_labels),
                "category": dict(self.config.level_prompts),
            },
            "thresholds": {
                "min_person_confidence": self.config.min_person_confidence,
                "min_solo_confidence": self.config.min_solo_confidence,
                "min_solo_margin": self.config.min_solo_margin,
                "min_solo_frame_ratio": self.config.min_solo_frame_ratio,
                "min_category_confidence": self.config.min_category_confidence,
            },
            "routing": {
                "copy_mode": self.config.copy_mode,
                "ignored_label": self.config.ignored_label,
                "pet_label": self.config.pet_label,
                "enable_pet_sorting": self.config.enable_pet_sorting,
            },
            "features": {
                "image_analysis": True,
                "video_analysis": False,
                "face_sorting": False,
                "runtime_prompt_editing": False,
            },
        }

    @staticmethod
    def _runner_script() -> str:
        return dedent(
            """\
            #!/usr/bin/env python3
            from __future__ import annotations

            import json
            from pathlib import Path
            import sys

            import numpy as np
            import onnxruntime as ort
            from PIL import Image


            def load_bundle(bundle_dir: Path):
                config = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
                embeddings = np.load(bundle_dir / config["runtime"]["embeddings_file"])
                model_name = config["runtime"]["primary_model"]
                model_path = bundle_dir / model_name
                if not model_path.exists():
                    model_path = bundle_dir / config["runtime"]["fallback_model"]
                session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                return config, embeddings, session


            def preprocess_image(image_path: Path, preprocess_config: dict[str, object]) -> np.ndarray:
                resize_size = int(preprocess_config["resize_size"])
                crop_size = int(preprocess_config["crop_size"])
                mean = np.array(preprocess_config["mean"], dtype=np.float32)
                std = np.array(preprocess_config["std"], dtype=np.float32)

                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    width, height = image.size
                    scale = resize_size / min(width, height)
                    resized = image.resize(
                        (int(round(width * scale)), int(round(height * scale))),
                        Image.Resampling.BICUBIC,
                    )
                    left = max(0, (resized.width - crop_size) // 2)
                    top = max(0, (resized.height - crop_size) // 2)
                    image = resized.crop((left, top, left + crop_size, top + crop_size))

                arr = np.asarray(image, dtype=np.float32) / 255.0
                arr = (arr - mean) / std
                arr = np.transpose(arr, (2, 0, 1))
                arr = np.expand_dims(arr, axis=0).astype(np.float32)
                return arr


            def softmax(values: np.ndarray) -> np.ndarray:
                shifted = values - np.max(values)
                exps = np.exp(shifted)
                return exps / np.sum(exps)


            def scores_from_embeddings(image_features: np.ndarray, text_embeddings: np.ndarray, labels: list[str]) -> dict[str, float]:
                logits = 100.0 * image_features @ text_embeddings.T
                probs = softmax(logits[0])
                return {label: float(probs[idx]) for idx, label in enumerate(labels)}


            def analyze_image(image_path: Path, config: dict[str, object], embeddings, session) -> dict[str, object]:
                image = preprocess_image(image_path, config["preprocess"])
                input_name = session.get_inputs()[0].name
                image_features = session.run(None, {input_name: image})[0]

                subject_labels = list(config["labels"]["subject"])
                count_labels = list(config["labels"]["count"])
                category_labels = list(config["labels"]["category"])
                solo_count_label = count_labels[0]
                no_person_count_label = count_labels[1]
                multi_person_count_label = count_labels[2]

                subject_scores = scores_from_embeddings(image_features, embeddings["subject"], subject_labels)
                count_scores = scores_from_embeddings(image_features, embeddings["count"], count_labels)
                category_scores = scores_from_embeddings(image_features, embeddings["category"], category_labels)

                subject = max(subject_scores, key=subject_scores.get)
                solo_label = max(count_scores, key=count_scores.get)
                category = max(category_scores, key=category_scores.get)

                thresholds = config["thresholds"]
                person_score = subject_scores.get("person", 0.0)
                solo_score = count_scores.get(solo_count_label, 0.0)
                non_solo = max(
                    count_scores.get(no_person_count_label, 0.0),
                    count_scores.get(multi_person_count_label, 0.0),
                )
                is_solo_person = (
                    subject == "person"
                    and person_score >= float(thresholds["min_person_confidence"])
                    and solo_score >= float(thresholds["min_solo_confidence"])
                    and (solo_score - non_solo) >= float(thresholds["min_solo_margin"])
                )

                return {
                    "source": str(image_path),
                    "subject": subject,
                    "subject_confidence": subject_scores[subject],
                    "subject_scores": subject_scores,
                    "is_solo_person": is_solo_person,
                    "solo_label": solo_label,
                    "solo_confidence": solo_score,
                    "count_scores": count_scores,
                    "category": category,
                    "category_confidence": category_scores[category],
                    "category_scores": category_scores,
                }


            def main(argv: list[str]) -> int:
                if len(argv) < 2:
                    print("Usage: run_media_sort.py <image-path> [more-images...]", file=sys.stderr)
                    return 2

                bundle_dir = Path(__file__).resolve().parent
                config, embeddings, session = load_bundle(bundle_dir)
                for raw_path in argv[1:]:
                    result = analyze_image(Path(raw_path), config, embeddings, session)
                    print(json.dumps(result, ensure_ascii=True))
                return 0


            if __name__ == "__main__":
                raise SystemExit(main(sys.argv))
            """
        )

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import ExifTags, Image

from .config import SorterConfig
from .core import MediaClassifier, Prediction
from .index import IndexStore, record_from_row


@dataclass(slots=True)
class AnalysisStats:
    counts: dict[str, int] = field(default_factory=dict)
    total_seen: int = 0
    image_files: int = 0
    video_files: int = 0
    skipped: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.counts)


SortStats = AnalysisStats


@dataclass(slots=True)
class MediaMetadata:
    file_name: str
    extension: str
    file_size: int
    mtime_ns: int
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    exif: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class MediaRecord:
    source: str
    media_type: str
    destination: str | None
    route_label: str | None
    metadata: MediaMetadata
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
    face_identity: str | None
    sampled_frames: list[int]
    status: str
    error: str | None = None


SortRecord = MediaRecord


class FaceSorter:
    def __init__(self, config: SorterConfig, classifier: MediaClassifier) -> None:
        self.config = config
        self.classifier = classifier
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Face sorting requires opencv-python-headless. Install with: pip install -e .[video]"
            ) from exc

        self._cv2 = cv2
        self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._cluster_centroids: dict[str, np.ndarray] = {}
        self._cluster_counts: dict[str, int] = {}
        self._next_cluster = 1
        self._tag_embeddings = self._load_tag_embeddings()

    def assign_for_image(self, file_path: Path) -> str | None:
        face = self._face_embedding_from_image(file_path)
        if face is None:
            return None

        if self.config.face_mode == "tagged":
            return self._assign_tag(face)
        return self._assign_cluster(face)

    def assign_for_video(self, file_path: Path, frame_indices: list[int]) -> str | None:
        face = self._face_embedding_from_video(file_path, frame_indices)
        if face is None:
            return None

        if self.config.face_mode == "tagged":
            return self._assign_tag(face)
        return self._assign_cluster(face)

    def _load_tag_embeddings(self) -> dict[str, np.ndarray]:
        if self.config.face_mode != "tagged":
            return {}
        if self.config.face_tags_dir is None or not self.config.face_tags_dir.exists():
            return {}

        tags: dict[str, np.ndarray] = {}
        for tag_dir in sorted(self.config.face_tags_dir.iterdir()):
            if not tag_dir.is_dir():
                continue
            embeddings: list[np.ndarray] = []
            for image_path in sorted(tag_dir.iterdir()):
                if image_path.suffix.lower() not in self.config.image_extensions:
                    continue
                emb = self._face_embedding_from_image(image_path)
                if emb is not None:
                    embeddings.append(emb)
            if embeddings:
                arr = np.stack(embeddings, axis=0)
                mean = arr.mean(axis=0)
                tags[tag_dir.name] = mean / (np.linalg.norm(mean) + 1e-12)

        return tags

    def _assign_tag(self, embedding: np.ndarray) -> str:
        if not self._tag_embeddings:
            return "unknown"

        best_tag = "unknown"
        best_score = -1.0
        for tag, ref in self._tag_embeddings.items():
            score = float(np.dot(embedding, ref))
            if score > best_score:
                best_tag = tag
                best_score = score

        if best_score < self.config.face_similarity_threshold:
            return "unknown"
        return best_tag

    def _assign_cluster(self, embedding: np.ndarray) -> str:
        if not self._cluster_centroids:
            return self._new_cluster(embedding)

        best_label = ""
        best_score = -1.0
        for label, centroid in self._cluster_centroids.items():
            score = float(np.dot(embedding, centroid))
            if score > best_score:
                best_label = label
                best_score = score

        if best_score < self.config.face_similarity_threshold:
            return self._new_cluster(embedding)

        count = self._cluster_counts[best_label]
        centroid = self._cluster_centroids[best_label]
        updated = (centroid * count + embedding) / (count + 1)
        self._cluster_centroids[best_label] = updated / (np.linalg.norm(updated) + 1e-12)
        self._cluster_counts[best_label] = count + 1
        return best_label

    def _new_cluster(self, embedding: np.ndarray) -> str:
        label = f"face_{self._next_cluster:03d}"
        self._next_cluster += 1
        self._cluster_centroids[label] = embedding
        self._cluster_counts[label] = 1
        return label

    def _face_embedding_from_image(self, file_path: Path) -> np.ndarray | None:
        img = self._cv2.imread(str(file_path))
        if img is None:
            return None
        return self._face_embedding_from_bgr(img)

    def _face_embedding_from_video(self, file_path: Path, frame_indices: list[int]) -> np.ndarray | None:
        capture = self._cv2.VideoCapture(str(file_path))
        if not capture.isOpened():
            return None

        try:
            embeddings: list[np.ndarray] = []
            for idx in frame_indices:
                capture.set(self._cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = capture.read()
                if not ok:
                    continue
                emb = self._face_embedding_from_bgr(frame)
                if emb is not None:
                    embeddings.append(emb)
            if not embeddings:
                return None

            avg = np.mean(np.stack(embeddings, axis=0), axis=0)
            return avg / (np.linalg.norm(avg) + 1e-12)
        finally:
            capture.release()

    def _face_embedding_from_bgr(self, bgr_frame) -> np.ndarray | None:
        gray = self._cv2.cvtColor(bgr_frame, self._cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda it: int(it[2] * it[3]))
        crop = bgr_frame[y : y + h, x : x + w]
        if crop.size == 0:
            return None

        rgb = self._cv2.cvtColor(crop, self._cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        emb = self.classifier.embedding(pil_img)
        return emb / (np.linalg.norm(emb) + 1e-12)


class MediaAnalyzer:
    def __init__(self, config: SorterConfig, classifier: MediaClassifier | None = None) -> None:
        self.config = config
        self.classifier = classifier or MediaClassifier(config)
        self.face_sorter = FaceSorter(config, self.classifier) if config.enable_face_sorting else None
        self.records: list[MediaRecord] = []
        self.manifest_output_path: Path | None = None
        self.index_db_path: Path | None = None
        self.index_updated_count: int = 0
        self.index_skipped_count: int = 0
        self.index_pruned_count: int | None = None
        self.stats = AnalysisStats(counts={"ok": 0, "error": 0, "skipped": 0})

    def run(self) -> AnalysisStats:
        seen_sources: set[str] = set()
        index_store: IndexStore | None = None

        if self.config.enable_index:
            self.index_db_path = self.config.index_db_path or (self._artifact_root() / "media_index.sqlite3")
            index_store = IndexStore(self.index_db_path, mode=self.config.index_mode)

        try:
            for file_path in self._iter_input_files():
                if self.config.limit is not None and self.stats.total_seen >= self.config.limit:
                    break

                seen_sources.add(str(file_path))

                if index_store is not None:
                    should_process, row = index_store.should_process(file_path)
                    if not should_process and row is not None:
                        cached = MediaRecord(**record_from_row(row, status="skipped"))
                        self.records.append(cached)
                        self.stats.skipped += 1
                        self.stats.counts["skipped"] = self.stats.counts.get("skipped", 0) + 1
                        self.index_skipped_count += 1
                        self.stats.total_seen += 1
                        continue

                record = self.analyze_file(file_path)
                self.records.append(record)
                self.stats.counts[record.status] = self.stats.counts.get(record.status, 0) + 1
                if record.status == "error":
                    self.stats.errors += 1

                if index_store is not None:
                    index_store.upsert_record(record, file_size=record.metadata.file_size, mtime_ns=record.metadata.mtime_ns)
                    self.index_updated_count += 1

                self.stats.total_seen += 1
        finally:
            if index_store is not None:
                if self.config.index_prune_missing:
                    self.index_pruned_count = index_store.prune_missing(seen_sources)
                index_store.commit()
                index_store.close()

        if self.config.write_manifest:
            self.manifest_output_path = self._write_manifest()

        return self.stats

    def analyze_file(self, file_path: Path) -> MediaRecord:
        file_size, mtime_ns = self._stat_file(file_path)
        media_type = self._media_type(file_path)
        metadata = self._collect_metadata(file_path, media_type, file_size, mtime_ns)

        try:
            prediction, sampled_frames = self._predict(file_path)
            face_identity = self._detect_face_identity(file_path, sampled_frames, prediction.subject)
            return MediaRecord(
                source=str(file_path),
                media_type=media_type,
                destination=None,
                route_label=None,
                metadata=metadata,
                subject=prediction.subject,
                subject_confidence=prediction.subject_confidence,
                subject_scores=prediction.subject_scores,
                is_solo_person=prediction.is_solo_person,
                solo_label=prediction.solo_label,
                solo_confidence=prediction.solo_confidence,
                count_scores=prediction.count_scores,
                category=prediction.category,
                category_confidence=prediction.category_confidence,
                category_scores=prediction.category_scores,
                face_identity=face_identity,
                sampled_frames=sampled_frames,
                status="ok",
            )
        except Exception as exc:
            return MediaRecord(
                source=str(file_path),
                media_type=media_type,
                destination=None,
                route_label=None,
                metadata=metadata,
                subject="unknown",
                subject_confidence=0.0,
                subject_scores={},
                is_solo_person=False,
                solo_label="unknown",
                solo_confidence=0.0,
                count_scores={},
                category=None,
                category_confidence=0.0,
                category_scores={},
                face_identity=None,
                sampled_frames=[],
                status="error",
                error=str(exc),
            )

    def _detect_face_identity(self, file_path: Path, sampled_frames: list[int], subject: str) -> str | None:
        if self.face_sorter is None or subject != "person":
            return None

        if file_path.suffix.lower() in self.config.image_extensions:
            return self.face_sorter.assign_for_image(file_path)
        return self.face_sorter.assign_for_video(file_path, sampled_frames)

    def _media_type(self, file_path: Path) -> str:
        if file_path.suffix.lower() in self.config.image_extensions:
            return "image"
        return "video"

    def _artifact_root(self) -> Path:
        return self.config.output_dir or (self.config.source_dir / ".media_sorter")

    def _stat_file(self, file_path: Path) -> tuple[int, int]:
        try:
            stat = file_path.stat()
            return int(stat.st_size), int(stat.st_mtime_ns)
        except OSError:
            return 0, 0

    def _collect_metadata(
        self, file_path: Path, media_type: str, file_size: int, mtime_ns: int
    ) -> MediaMetadata:
        if media_type == "image":
            return self._collect_image_metadata(file_path, file_size, mtime_ns)
        return self._collect_video_metadata(file_path, file_size, mtime_ns)

    def _collect_image_metadata(self, file_path: Path, file_size: int, mtime_ns: int) -> MediaMetadata:
        width = None
        height = None
        exif: dict[str, str] = {}
        try:
            with Image.open(file_path) as image:
                width, height = image.size
                raw_exif = image.getexif()
                if raw_exif:
                    for tag_id, value in raw_exif.items():
                        tag_name = str(ExifTags.TAGS.get(tag_id, tag_id))
                        exif[tag_name] = self._stringify_metadata_value(value)
        except Exception:
            pass

        return MediaMetadata(
            file_name=file_path.name,
            extension=file_path.suffix.lower(),
            file_size=file_size,
            mtime_ns=mtime_ns,
            width=width,
            height=height,
            exif=exif,
        )

    def _collect_video_metadata(self, file_path: Path, file_size: int, mtime_ns: int) -> MediaMetadata:
        width = None
        height = None
        frame_count = None
        fps = None
        duration_seconds = None

        try:
            import cv2

            capture = cv2.VideoCapture(str(file_path))
            if capture.isOpened():
                try:
                    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
                    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
                    fps_value = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                    fps = fps_value if fps_value > 0 else None
                    if frame_count is not None and fps is not None and fps > 0:
                        duration_seconds = float(frame_count / fps)
                finally:
                    capture.release()
        except Exception:
            pass

        return MediaMetadata(
            file_name=file_path.name,
            extension=file_path.suffix.lower(),
            file_size=file_size,
            mtime_ns=mtime_ns,
            width=width,
            height=height,
            frame_count=frame_count,
            fps=fps,
            duration_seconds=duration_seconds,
            exif={},
        )

    @staticmethod
    def _stringify_metadata_value(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _iter_input_files(self):
        for file_path in sorted(self.config.source_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in self.config.all_extensions:
                yield file_path

    def _predict(self, file_path: Path) -> tuple[Prediction, list[int]]:
        ext = file_path.suffix.lower()
        if ext in self.config.image_extensions:
            self.stats.image_files += 1
            return self.classifier.predict_image(file_path), []

        self.stats.video_files += 1
        return self._predict_video(file_path)

    def _predict_video(self, file_path: Path) -> tuple[Prediction, list[int]]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Video support requires opencv-python-headless. Install with: pip install -e .[video]"
            ) from exc

        capture = cv2.VideoCapture(str(file_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {file_path}")

        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

            if self.config.video_sampling_mode == "second" and fps > 0:
                step = max(1, int(round(fps * self.config.video_seconds_per_sample)))
            else:
                step = self.config.video_frame_skip

            if frame_count > 0:
                sample_idxs = list(range(0, frame_count, step))
            else:
                sample_idxs = list(range(0, self.config.video_frame_skip * 100, step))

            if self.config.max_video_frames is not None:
                sample_idxs = sample_idxs[: self.config.max_video_frames]

            predictions: list[Prediction] = []
            valid_idxs: list[int] = []
            for idx in sample_idxs:
                capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = capture.read()
                if not ok:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                predictions.append(self.classifier.predict_pil(pil_image))
                valid_idxs.append(idx)

            if not predictions:
                raise RuntimeError("No decodable frames found")

            return self._aggregate_video_predictions(file_path, predictions), valid_idxs
        finally:
            capture.release()

    def _aggregate_video_predictions(self, file_path: Path, predictions: list[Prediction]) -> Prediction:
        total = len(predictions)
        subject_scores = self._mean_scores([pred.subject_scores for pred in predictions])
        count_scores = self._mean_scores([pred.count_scores for pred in predictions])
        category_scores = self._mean_scores([pred.category_scores for pred in predictions])

        subject = max(subject_scores, key=subject_scores.get)
        subject_confidence = float(subject_scores[subject])

        solo_preds = [pred for pred in predictions if pred.is_solo_person]
        solo_ratio = len(solo_preds) / total
        is_solo = subject == "person" and bool(solo_preds) and solo_ratio >= self.config.min_solo_frame_ratio

        solo_label = max(count_scores, key=count_scores.get) if count_scores else self.config.count_labels[1]
        category = max(category_scores, key=category_scores.get) if category_scores else None
        category_confidence = float(category_scores[category]) if category is not None else 0.0

        return Prediction(
            file_path=file_path,
            subject=subject,
            subject_confidence=subject_confidence,
            subject_scores=subject_scores,
            is_solo_person=is_solo,
            solo_label=solo_label,
            solo_confidence=solo_ratio,
            count_scores=count_scores,
            category=category,
            category_confidence=category_confidence,
            category_scores=category_scores,
        )

    @staticmethod
    def _mean_scores(score_maps: list[dict[str, float]]) -> dict[str, float]:
        aggregate: dict[str, list[float]] = {}
        for score_map in score_maps:
            for label, score in score_map.items():
                aggregate.setdefault(label, []).append(float(score))
        return {label: float(np.mean(scores)) for label, scores in aggregate.items()}

    def _write_manifest(self) -> Path:
        default_name = "manifest.json" if self.config.manifest_format == "json" else "manifest.jsonl"
        manifest_path = self.config.manifest_path or (self._artifact_root() / default_name)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.manifest_format == "json":
            data = [asdict(record) for record in self.records]
            manifest_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
            return manifest_path

        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")
        return manifest_path


class MediaSorter:
    def __init__(self, config: SorterConfig, classifier: MediaClassifier | None = None) -> None:
        self.config = config
        self.analyzer = MediaAnalyzer(config, classifier=classifier)
        self.records: list[MediaRecord] = []
        self.manifest_output_path: Path | None = None
        self.index_db_path: Path | None = None
        self.index_updated_count: int = 0
        self.index_skipped_count: int = 0
        self.index_pruned_count: int | None = None
        self.stats = AnalysisStats(
            counts={label: 0 for label in self.config.level_prompts}
            | {self.config.ignored_label: 0, self.config.pet_label: 0}
        )

    def run(self) -> AnalysisStats:
        self._ensure_output_dirs()
        seen_sources: set[str] = set()
        index_store: IndexStore | None = None

        if self.config.enable_index:
            artifact_root = self.config.output_dir or (self.config.source_dir / ".media_sorter")
            self.index_db_path = self.config.index_db_path or (artifact_root / "media_index.sqlite3")
            index_store = IndexStore(self.index_db_path, mode=self.config.index_mode)

        try:
            for file_path in self.analyzer._iter_input_files():
                if self.config.limit is not None and self.stats.total_seen >= self.config.limit:
                    break

                seen_sources.add(str(file_path))

                if index_store is not None:
                    should_process, row = index_store.should_process(file_path)
                    if not should_process and row is not None:
                        cached = MediaRecord(**record_from_row(row, status="skipped"))
                        self.records.append(cached)
                        self.stats.skipped += 1
                        self.index_skipped_count += 1
                        self.stats.total_seen += 1
                        continue

                record = self.analyzer.analyze_file(file_path)
                self.stats.image_files = self.analyzer.stats.image_files
                self.stats.video_files = self.analyzer.stats.video_files

                if record.status == "error":
                    route_label = self.config.ignored_label
                    destination = self._destination_for_label(route_label)
                    self.stats.errors += 1
                else:
                    route_label, destination = self._decide_destination(record)

                record.route_label = route_label
                record.destination = str(destination) if destination is not None else None
                self._store(file_path, destination)
                self.stats.counts[route_label] = self.stats.counts.get(route_label, 0) + 1
                self.records.append(record)

                if index_store is not None:
                    index_store.upsert_record(record, file_size=record.metadata.file_size, mtime_ns=record.metadata.mtime_ns)
                    self.index_updated_count += 1

                self.stats.total_seen += 1
        finally:
            if index_store is not None:
                if self.config.index_prune_missing:
                    self.index_pruned_count = index_store.prune_missing(seen_sources)
                index_store.commit()
                index_store.close()

        if self.config.write_manifest:
            self.manifest_output_path = self._write_manifest()

        return self.stats

    def _destination_for_label(self, label: str) -> Path | None:
        if self.config.output_dir is None:
            return None
        return self.config.output_dir / label

    def _ensure_output_dirs(self) -> None:
        if self.config.output_dir is None or self.config.dry_run or self.config.copy_mode == "none":
            return

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        for level in self.config.level_prompts:
            (self.config.output_dir / level).mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / self.config.ignored_label).mkdir(parents=True, exist_ok=True)
        if self.config.enable_pet_sorting:
            (self.config.output_dir / self.config.pet_label).mkdir(parents=True, exist_ok=True)
        if self.config.enable_face_sorting:
            (self.config.output_dir / self.config.face_label).mkdir(parents=True, exist_ok=True)

    def _decide_destination(self, record: MediaRecord) -> tuple[str, Path | None]:
        if record.subject == "pet" and self.config.enable_pet_sorting:
            return self.config.pet_label, self._destination_for_label(self.config.pet_label)

        if not record.is_solo_person:
            return self.config.ignored_label, self._destination_for_label(self.config.ignored_label)

        if record.category is None:
            return self.config.ignored_label, self._destination_for_label(self.config.ignored_label)

        if record.category_confidence < self.config.min_category_confidence:
            return self.config.ignored_label, self._destination_for_label(self.config.ignored_label)

        if self.config.enable_face_sorting and record.face_identity:
            if self.config.output_dir is None:
                return f"{self.config.face_label}/{record.face_identity}", None
            return (
                f"{self.config.face_label}/{record.face_identity}",
                self.config.output_dir / self.config.face_label / record.face_identity,
            )

        return record.category, self._destination_for_label(record.category)

    def _store(self, source_file: Path, destination_dir: Path | None) -> None:
        if destination_dir is None or self.config.dry_run or self.config.copy_mode == "none":
            return

        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_file = destination_dir / source_file.name

        if self.config.copy_mode == "copy":
            shutil.copy2(source_file, destination_file)
        else:
            shutil.move(str(source_file), str(destination_file))

    def _write_manifest(self) -> Path:
        default_name = "manifest.json" if self.config.manifest_format == "json" else "manifest.jsonl"
        artifact_root = self.config.output_dir or (self.config.source_dir / ".media_sorter")
        manifest_path = self.config.manifest_path or (artifact_root / default_name)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.manifest_format == "json":
            data = [asdict(record) for record in self.records]
            manifest_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
            return manifest_path

        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")
        return manifest_path

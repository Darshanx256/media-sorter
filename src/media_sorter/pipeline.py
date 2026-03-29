from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, TypeVar

import numpy as np
from PIL import ExifTags, Image

T = TypeVar("T")


async def _run_in_executor(
    executor: ThreadPoolExecutor,
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))


logger = logging.getLogger(__name__)


def _create_io_executor(prefix: str, max_workers: int) -> ThreadPoolExecutor | None:
    executor: ThreadPoolExecutor | None = None
    try:
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=prefix)
        future = executor.submit(lambda: None)
        future.result(timeout=0.5)
        return executor
    except (Exception, TimeoutError) as exc:
        if executor is not None:
            executor.shutdown(wait=False)
        logger.debug("I/O executor %s unavailable, falling back to synchronous operations: %s", prefix, exc)
        return None


_IO_WAIT_TIMEOUT = 5.0

_FILENAME_DATE_REGEX = re.compile(r"(?<!\d)(\d{4})[-_]?(\d{2})[-_]?(\d{2})(?!\d)")

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
    duplicates: int = 0

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
    capture_date: str | None = None
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
        self.stats = AnalysisStats(counts={"ok": 0, "error": 0, "skipped": 0, "duplicate": 0})
        self._dedup_hash_index: dict[str, str] = {}
        self._embedding_index: list[tuple[np.ndarray, str]] = []
        self._record_registry: dict[str, MediaRecord] = {}
        max_workers = max(2, (os.cpu_count() or 1))
        self._io_executor = _create_io_executor("media-sorter-analyzer-io", max_workers)

    def run(self) -> AnalysisStats:
        return asyncio.run(self.run_async())

    async def _run_io(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        executor = self._io_executor
        if executor is None:
            return func(*args, **kwargs)
        func_name = getattr(func, "__name__", repr(func))
        try:
            return await asyncio.wait_for(
                _run_in_executor(executor, func, *args, **kwargs),
                timeout=_IO_WAIT_TIMEOUT,
            )
        except asyncio.TimeoutError as exc:
            logger.debug("Threaded I/O (%s) timed out; retrying synchronously (%s)", func_name, exc)
            self._shutdown_io_executor()
            return func(*args, **kwargs)

    def _shutdown_io_executor(self) -> None:
        if self._io_executor is not None:
            self._io_executor.shutdown(wait=False)
            self._io_executor = None

    async def run_async(
        self,
        progress_callback: Callable[[Path, AnalysisStats], None] | None = None,
        *,
        file_paths: list[Path] | None = None,
    ) -> AnalysisStats:
        seen_sources: set[str] = set()
        index_store: IndexStore | None = None

        try:
            if self.config.enable_index:
                self.index_db_path = self.config.index_db_path or (self._artifact_root() / "media_index.sqlite3")
                index_store = IndexStore(self.index_db_path, mode=self.config.index_mode)

            job_files = file_paths if file_paths is not None else await self._collect_input_files_async()
            try:
                for file_path in job_files:
                    if self.config.limit is not None and self.stats.total_seen >= self.config.limit:
                        break

                    seen_sources.add(str(file_path))

                    if index_store is not None:
                        should_process, row = index_store.should_process(file_path)
                        if not should_process and row is not None:
                            cached = MediaRecord(**record_from_row(row, status="skipped"))
                            self.records.append(cached)
                            self._record_registry[cached.source] = cached
                            self.stats.skipped += 1
                            self.stats.counts["skipped"] = self.stats.counts.get("skipped", 0) + 1
                            self.index_skipped_count += 1
                            self.stats.total_seen += 1
                            if progress_callback:
                                progress_callback(file_path, self.stats)
                            continue

                    record = await self.analyze_file_async(file_path)
                    self.records.append(record)
                    self.stats.counts[record.status] = self.stats.counts.get(record.status, 0) + 1
                    if record.status == "error":
                        self.stats.errors += 1
                    elif record.status == "duplicate":
                        self.stats.duplicates += 1

                    if index_store is not None:
                        index_store.upsert_record(record, file_size=record.metadata.file_size, mtime_ns=record.metadata.mtime_ns)
                        self.index_updated_count += 1

                    self.stats.total_seen += 1
                    if progress_callback:
                        progress_callback(file_path, self.stats)
            finally:
                if index_store is not None:
                    if self.config.index_prune_missing:
                        self.index_pruned_count = index_store.prune_missing(seen_sources)
                    index_store.commit()
                    index_store.close()

            if self.config.write_manifest:
                self.manifest_output_path = await self._write_manifest_async()

            return self.stats
        finally:
            self._shutdown_io_executor()

    def analyze_file(self, file_path: Path) -> MediaRecord:
        return asyncio.run(self.analyze_file_async(file_path))

    async def analyze_file_async(self, file_path: Path) -> MediaRecord:
        file_size, mtime_ns = await self._run_io(self._stat_file, file_path)
        media_type = self._media_type(file_path)
        file_hash: str | None = None

        if self.config.enable_deduplication:
            file_hash = await self._hash_file_async(file_path)
            existing_path = self._dedup_hash_index.get(file_hash)
            if existing_path:
                original_record = self._record_registry.get(existing_path)
                metadata = self._duplicate_metadata(file_path, file_size, mtime_ns, original_record)
                return self._build_duplicate_record(
                    file_path,
                    media_type,
                    metadata,
                    existing_path,
                    original_record,
                    "SHA256 hash match",
                )

        metadata = await self._run_io(
            self._collect_metadata, file_path, media_type, file_size, mtime_ns
        )
        try:
            prediction, sampled_frames = self._predict(file_path)
            embedding = prediction.embedding
            near_duplicate_path: str | None = None
            if self.config.enable_deduplication and embedding is not None:
                near_duplicate_path = self._find_near_duplicate(embedding)

            if near_duplicate_path:
                original_record = self._record_registry.get(near_duplicate_path)
                metadata = self._duplicate_metadata(file_path, file_size, mtime_ns, original_record)
                return self._build_duplicate_record(
                    file_path,
                    media_type,
                    metadata,
                    near_duplicate_path,
                    original_record,
                    "CLIP similarity",
                )

            face_identity = self._detect_face_identity(file_path, sampled_frames, prediction.subject)
            record = MediaRecord(
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
            self._register_unique_entry(str(file_path), file_hash, embedding, record)
            return record
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

    def _scan_input_files(self) -> list[Path]:
        if not self.config.source_dir.exists():
            return []
        return [
            file_path
            for file_path in sorted(self.config.source_dir.rglob("*"))
            if file_path.is_file() and file_path.suffix.lower() in self.config.all_extensions
        ]

    async def _collect_input_files_async(self) -> list[Path]:
        return await self._run_io(self._scan_input_files)

    async def _hash_file_async(self, file_path: Path) -> str:
        return await self._run_io(self._compute_sha256, file_path)

    def _compute_sha256(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _duplicate_metadata(
        self,
        file_path: Path,
        file_size: int,
        mtime_ns: int,
        original_record: MediaRecord | None,
    ) -> MediaMetadata:
        base_meta = original_record.metadata if original_record else None
        if base_meta:
            capture_date = base_meta.capture_date or self._resolve_capture_date(file_path, base_meta.exif, mtime_ns)
            return MediaMetadata(
                file_name=file_path.name,
                extension=file_path.suffix.lower(),
                file_size=file_size,
                mtime_ns=mtime_ns,
                width=base_meta.width,
                height=base_meta.height,
                frame_count=base_meta.frame_count,
                fps=base_meta.fps,
                duration_seconds=base_meta.duration_seconds,
                capture_date=capture_date,
                exif=dict(base_meta.exif),
            )

        capture_date = self._resolve_capture_date(file_path, {}, mtime_ns)
        return MediaMetadata(
            file_name=file_path.name,
            extension=file_path.suffix.lower(),
            file_size=file_size,
            mtime_ns=mtime_ns,
            capture_date=capture_date,
            exif={},
        )

    def _build_duplicate_record(
        self,
        file_path: Path,
        media_type: str,
        metadata: MediaMetadata,
        original_path: str,
        original_record: MediaRecord | None,
        reason: str,
    ) -> MediaRecord:
        if original_record:
            subject = original_record.subject
            subject_confidence = original_record.subject_confidence
            subject_scores = dict(original_record.subject_scores)
            is_solo_person = original_record.is_solo_person
            solo_label = original_record.solo_label
            solo_confidence = original_record.solo_confidence
            count_scores = dict(original_record.count_scores)
            category = original_record.category
            category_confidence = original_record.category_confidence
            category_scores = dict(original_record.category_scores)
            face_identity = original_record.face_identity
            sampled_frames = list(original_record.sampled_frames)
            original_source = original_record.source
        else:
            subject = "unknown"
            subject_confidence = 0.0
            subject_scores = {}
            is_solo_person = False
            solo_label = "unknown"
            solo_confidence = 0.0
            count_scores = {}
            category = None
            category_confidence = 0.0
            category_scores = {}
            face_identity = None
            sampled_frames = []
            original_source = original_path

        return MediaRecord(
            source=str(file_path),
            media_type=media_type,
            destination=None,
            route_label=None,
            metadata=metadata,
            subject=subject,
            subject_confidence=subject_confidence,
            subject_scores=subject_scores,
            is_solo_person=is_solo_person,
            solo_label=solo_label,
            solo_confidence=solo_confidence,
            count_scores=count_scores,
            category=category,
            category_confidence=category_confidence,
            category_scores=category_scores,
            face_identity=face_identity,
            sampled_frames=sampled_frames,
            status="duplicate",
            error=f"duplicate ({reason}) of {original_source}",
        )

    def _register_unique_entry(
        self,
        source_path: str,
        sha_hash: str | None,
        embedding: np.ndarray | None,
        record: MediaRecord,
    ) -> None:
        if sha_hash:
            self._dedup_hash_index[sha_hash] = source_path
        if embedding is not None:
            self._embedding_index.append((embedding.copy(), source_path))
        self._record_registry[source_path] = record

    def _find_near_duplicate(self, embedding: np.ndarray) -> str | None:
        threshold = self.config.dedup_similarity_threshold
        for candidate, path in self._embedding_index:
            score = float(np.dot(embedding, candidate))
            if score >= threshold:
                return path
        return None

    def _resolve_capture_date(self, file_path: Path, exif: dict[str, str], mtime_ns: int) -> str:
        for key in ("DateTimeOriginal", "DateTime", "CreateDate"):
            parsed = self._parse_exif_datetime(exif.get(key))
            if parsed:
                return parsed.isoformat()

        filename_date = self._extract_date_from_filename(file_path.name)
        if filename_date:
            return filename_date.isoformat()

        return datetime.fromtimestamp(mtime_ns / 1_000_000_000, tz=timezone.utc).isoformat()

    def _parse_exif_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        raw = str(value).split(".", 1)[0].strip()
        if not raw:
            return None
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _extract_date_from_filename(self, file_name: str) -> datetime | None:
        match = _FILENAME_DATE_REGEX.search(file_name)
        if not match:
            return None
        try:
            year, month, day = match.groups()
            return datetime(int(year), int(month), int(day), tzinfo=timezone.utc)
        except ValueError:
            return None
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

        capture_date = self._resolve_capture_date(file_path, exif, mtime_ns)
        return MediaMetadata(
            file_name=file_path.name,
            extension=file_path.suffix.lower(),
            file_size=file_size,
            mtime_ns=mtime_ns,
            width=width,
            height=height,
            capture_date=capture_date,
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

        capture_date = self._resolve_capture_date(file_path, {}, mtime_ns)
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
            capture_date=capture_date,
            exif={},
        )

    @staticmethod
    def _stringify_metadata_value(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _iter_input_files(self):
        yield from self._scan_input_files()

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

        # Use the averaged CLIP softmax probability for the solo class so that
        # solo_confidence carries the same unit as the image path (probability,
        # not frame ratio). solo_ratio is implicitly captured via is_solo_person.
        solo_confidence = float(count_scores.get(self.config.count_labels[0], 0.0)) if count_scores else 0.0

        embeddings = [pred.embedding for pred in predictions if pred.embedding is not None]
        aggregated_embedding: np.ndarray | None = None
        if embeddings:
            stacked = np.stack(embeddings, axis=0)
            mean_embedding = np.mean(stacked, axis=0)
            aggregated_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-12)

        return Prediction(
            file_path=file_path,
            subject=subject,
            subject_confidence=subject_confidence,
            subject_scores=subject_scores,
            is_solo_person=is_solo,
            solo_label=solo_label,
            solo_confidence=solo_confidence,
            count_scores=count_scores,
            category=category,
            category_confidence=category_confidence,
            category_scores=category_scores,
            embedding=aggregated_embedding,
        )

    @staticmethod
    def _mean_scores(score_maps: list[dict[str, float]]) -> dict[str, float]:
        aggregate: dict[str, list[float]] = {}
        for score_map in score_maps:
            for label, score in score_map.items():
                aggregate.setdefault(label, []).append(float(score))
        return {label: float(np.mean(scores)) for label, scores in aggregate.items()}

    def _write_manifest(self) -> Path:
        return _write_records_to_manifest(
            self.records, self.config, artifact_root=self._artifact_root()
        )

    async def _write_manifest_async(self) -> Path:
        return await self._run_io(self._write_manifest)


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
        base_counts = {label: 0 for label in self.config.level_prompts}
        base_counts.update(
            {
                self.config.ignored_label: 0,
                self.config.pet_label: 0,
                "duplicate": 0,
                "skipped": 0,
            }
        )
        self.stats = AnalysisStats(counts=base_counts)
        self.planned_moves: list[tuple[Path, Path, str]] = []
        sorter_workers = max(1, (os.cpu_count() or 1) // 2)
        self._io_executor = _create_io_executor("media-sorter-sorter-io", sorter_workers)

    def run(self) -> AnalysisStats:
        return asyncio.run(self.run_async())

    async def _run_io(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        executor = self._io_executor
        if executor is None:
            return func(*args, **kwargs)
        func_name = getattr(func, "__name__", repr(func))
        try:
            return await asyncio.wait_for(
                _run_in_executor(executor, func, *args, **kwargs),
                timeout=_IO_WAIT_TIMEOUT,
            )
        except asyncio.TimeoutError as exc:
            logger.debug("Threaded I/O (%s) timed out; retrying synchronously (%s)", func_name, exc)
            self._shutdown_io_executor()
            return func(*args, **kwargs)

    def _shutdown_io_executor(self) -> None:
        if self._io_executor is not None:
            self._io_executor.shutdown(wait=False)
            self._io_executor = None

    async def run_async(
        self,
        progress_callback: Callable[[Path, AnalysisStats], None] | None = None,
        *,
        file_paths: list[Path] | None = None,
    ) -> AnalysisStats:
        try:
            await self._ensure_output_dirs_async()
            seen_sources: set[str] = set()
            index_store: IndexStore | None = None

            if self.config.enable_index:
                artifact_root = self.config.output_dir or (self.config.source_dir / ".media_sorter")
                self.index_db_path = self.config.index_db_path or (artifact_root / "media_index.sqlite3")
                index_store = IndexStore(self.index_db_path, mode=self.config.index_mode)

            job_files = file_paths if file_paths is not None else await self.analyzer._collect_input_files_async()
            try:
                for file_path in job_files:
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
                            if progress_callback:
                                progress_callback(file_path, self.stats)
                            continue

                    record = await self.analyzer.analyze_file_async(file_path)
                    self.stats.image_files = self.analyzer.stats.image_files
                    self.stats.video_files = self.analyzer.stats.video_files

                    if record.status == "error":
                        route_label = self.config.ignored_label
                        destination = self._destination_for_label(route_label)
                        self.stats.errors += 1
                        self.stats.counts[route_label] = self.stats.counts.get(route_label, 0) + 1
                    elif record.status == "duplicate":
                        route_label = self.config.ignored_label
                        destination = None
                        self.stats.counts["duplicate"] = self.stats.counts.get("duplicate", 0) + 1
                    else:
                        route_label, destination = self._decide_destination(record)
                        if route_label:
                            self.stats.counts[route_label] = self.stats.counts.get(route_label, 0) + 1

                    record.route_label = route_label
                    record.destination = str(destination) if destination is not None else None

                    if destination is not None and self.config.copy_mode != "none":
                        destination_file = destination / file_path.name
                        if self.config.dry_run:
                            self.planned_moves.append((file_path, destination_file, self.config.copy_mode))
                        else:
                            await self._store_async(file_path, destination)

                    self.records.append(record)

                    if index_store is not None:
                        index_store.upsert_record(record, file_size=record.metadata.file_size, mtime_ns=record.metadata.mtime_ns)
                        self.index_updated_count += 1

                    self.stats.total_seen += 1
                    if progress_callback:
                        progress_callback(file_path, self.stats)
            finally:
                if index_store is not None:
                    if self.config.index_prune_missing:
                        self.index_pruned_count = index_store.prune_missing(seen_sources)
                    index_store.commit()
                    index_store.close()

            if self.config.write_manifest:
                self.manifest_output_path = await self._write_manifest_async()
            self.stats.duplicates = self.analyzer.stats.duplicates
            return self.stats
        finally:
            self._shutdown_io_executor()

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

    async def _ensure_output_dirs_async(self) -> None:
        await self._run_io(self._ensure_output_dirs)

    def _decide_destination(self, record: MediaRecord) -> tuple[str, Path | None]:
        if record.subject == "pet" and self.config.enable_pet_sorting:
            return self.config.pet_label, self._destination_for_label(self.config.pet_label)

        if record.category is None:
            return self.config.ignored_label, self._destination_for_label(self.config.ignored_label)

        if record.category_confidence < self.config.min_category_confidence:
            return self.config.ignored_label, self._destination_for_label(self.config.ignored_label)

        if self.config.enable_face_sorting and record.is_solo_person and record.face_identity:
            if self.config.output_dir is None:
                return f"{self.config.face_label}/{record.face_identity}", None
            return (
                f"{self.config.face_label}/{record.face_identity}",
                self.config.output_dir / self.config.face_label / record.face_identity,
            )

        return record.category, self._destination_for_label(record.category)

    def _store(self, source_file: Path, destination_dir: Path | None) -> None:
        if destination_dir is None:
            return

        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_file = destination_dir / source_file.name

        if self.config.copy_mode == "copy":
            shutil.copy2(source_file, destination_file)
        else:
            shutil.move(str(source_file), str(destination_file))

    async def _store_async(self, source_file: Path, destination_dir: Path | None) -> None:
        await self._run_io(self._store, source_file, destination_dir)

    def _write_manifest(self) -> Path:
        artifact_root = self.config.output_dir or (self.config.source_dir / ".media_sorter")
        return _write_records_to_manifest(
            self.records, self.config, artifact_root=artifact_root
        )

    async def _write_manifest_async(self) -> Path:
        return await self._run_io(self._write_manifest)


def _write_records_to_manifest(
    records: list[MediaRecord],
    config: SorterConfig,
    artifact_root: Path,
) -> Path:
    """Write *records* to a manifest file and return the path written.

    Shared by both :class:`MediaAnalyzer` and :class:`MediaSorter` so that
    manifest serialisation logic lives in exactly one place.
    """
    default_name = "manifest.json" if config.manifest_format == "json" else "manifest.jsonl"
    manifest_path = config.manifest_path or (artifact_root / default_name)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if config.manifest_format == "json":
        data = [asdict(record) for record in records]
        manifest_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
        return manifest_path

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")
    return manifest_path

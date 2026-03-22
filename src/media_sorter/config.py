from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_LEVEL_PROMPTS: dict[str, str] = {
    "portrait": "a portrait photo of one person as the main subject",
    "group": "a photo of multiple people together",
    "pet": "a clear photo of a pet animal such as a dog or cat",
    "food": "a photo of food, a meal, or a drink",
    "travel": "a travel photo of a landmark, destination, or scenic place",
    "outdoor": "an outdoor photo in nature, a park, beach, or street scene",
    "document": "a photo or scan of a document, receipt, note, or paper",
    "screenshot": "a screenshot of a phone, app, website, or computer screen",
    "product": "a product photo of an object for catalog or listing use",
    "art": "artwork, illustration, drawing, poster, or graphic design image",
    "meme": "a meme, joke image, or internet reaction image",
    "other": "a miscellaneous photo that does not fit the other categories well",
}

DEFAULT_COUNT_LABELS: tuple[str, str, str] = (
    "a photo of exactly one person",
    "a photo of no people",
    "a photo of two or more people",
)

DEFAULT_SUBJECT_PROMPTS: dict[str, str] = {
    "person": "a photo of a person",
    "pet": "a photo of a pet animal like a dog or cat",
    "other": "a photo of an object, scene, or something without people or pets",
}

DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
)

DEFAULT_VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
)


@dataclass(slots=True)
class SorterConfig:
    source_dir: Path
    output_dir: Path | None = None
    model_name: str = "ViT-B/32"
    device: str = "cpu"
    limit: int | None = None

    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS
    video_extensions: tuple[str, ...] = DEFAULT_VIDEO_EXTENSIONS

    level_prompts: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LEVEL_PROMPTS))
    subject_prompts: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SUBJECT_PROMPTS))
    count_labels: tuple[str, str, str] = DEFAULT_COUNT_LABELS
    ignored_label: str = "ignored"
    pet_label: str = "pets"
    enable_pet_sorting: bool = True

    min_category_confidence: float = 0.0
    copy_mode: str = "none"  # copy | move | none
    dry_run: bool = False
    write_manifest: bool = False
    manifest_path: Path | None = None
    manifest_format: str = "jsonl"  # jsonl | json
    enable_index: bool = False
    index_db_path: Path | None = None
    index_mode: str = "full"  # full | update
    index_prune_missing: bool = False

    enable_face_sorting: bool = False
    face_mode: str = "unnamed"  # unnamed | tagged
    face_label: str = "faces"
    face_tags_dir: Path | None = None
    face_similarity_threshold: float = 0.82

    max_video_frames: int | None = None
    video_sampling_mode: str = "second"  # second | skip
    video_seconds_per_sample: float = 1.0
    video_frame_skip: int = 10
    min_solo_frame_ratio: float = 0.5
    min_person_confidence: float = 0.35
    min_solo_confidence: float = 0.45
    min_solo_margin: float = 0.05

    def __post_init__(self) -> None:
        self.source_dir = Path(self.source_dir)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)

        if self.copy_mode not in {"copy", "move", "none"}:
            raise ValueError("copy_mode must be 'copy', 'move' or 'none'")

        if self.copy_mode in {"copy", "move"} and self.output_dir is None:
            raise ValueError("output_dir is required when copy_mode is 'copy' or 'move'")

        if len(self.count_labels) != 3:
            raise ValueError("count_labels must have exactly 3 prompts")

        if not self.level_prompts:
            raise ValueError("level_prompts cannot be empty")

        if not self.subject_prompts:
            raise ValueError("subject_prompts cannot be empty")

        if not (0.0 <= self.min_category_confidence <= 1.0):
            raise ValueError("min_category_confidence must be between 0.0 and 1.0")

        if self.face_mode not in {"unnamed", "tagged"}:
            raise ValueError("face_mode must be 'unnamed' or 'tagged'")

        if not (0.0 <= self.face_similarity_threshold <= 1.0):
            raise ValueError("face_similarity_threshold must be between 0.0 and 1.0")

        if self.face_tags_dir is not None:
            self.face_tags_dir = Path(self.face_tags_dir)

        if self.manifest_path is not None:
            self.manifest_path = Path(self.manifest_path)

        if self.manifest_format not in {"jsonl", "json"}:
            raise ValueError("manifest_format must be 'jsonl' or 'json'")

        if self.index_db_path is not None:
            self.index_db_path = Path(self.index_db_path)

        if self.index_mode not in {"full", "update"}:
            raise ValueError("index_mode must be 'full' or 'update'")

        if self.max_video_frames is not None and self.max_video_frames < 1:
            raise ValueError("max_video_frames must be >= 1 when set")

        if self.video_sampling_mode not in {"second", "skip"}:
            raise ValueError("video_sampling_mode must be 'second' or 'skip'")

        if self.video_seconds_per_sample <= 0:
            raise ValueError("video_seconds_per_sample must be > 0")

        if self.video_frame_skip < 1:
            raise ValueError("video_frame_skip must be >= 1")

        if not (0.0 <= self.min_solo_frame_ratio <= 1.0):
            raise ValueError("min_solo_frame_ratio must be between 0.0 and 1.0")

        if not (0.0 <= self.min_person_confidence <= 1.0):
            raise ValueError("min_person_confidence must be between 0.0 and 1.0")

        if not (0.0 <= self.min_solo_confidence <= 1.0):
            raise ValueError("min_solo_confidence must be between 0.0 and 1.0")

        if not (0.0 <= self.min_solo_margin <= 1.0):
            raise ValueError("min_solo_margin must be between 0.0 and 1.0")

    @property
    def all_extensions(self) -> tuple[str, ...]:
        return self.image_extensions + self.video_extensions

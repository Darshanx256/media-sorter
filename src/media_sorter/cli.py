import argparse
import json
from pathlib import Path
import sys

from .config import DEFAULT_LEVEL_PROMPTS, DEFAULT_SUBJECT_PROMPTS, SorterConfig
from .finalize import BundleFinalizer
from .pipeline import MediaAnalyzer, MediaSorter


def _parse_prompts(raw: str | None) -> dict[str, str]:
    if not raw:
        return dict(DEFAULT_LEVEL_PROMPTS)

    prompts = json.loads(raw)
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError("--prompts must be a non-empty JSON object")

    return {str(k): str(v) for k, v in prompts.items()}


def _parse_subject_prompts(raw: str | None) -> dict[str, str]:
    if not raw:
        return dict(DEFAULT_SUBJECT_PROMPTS)

    prompts = json.loads(raw)
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError("--subject-prompts must be a non-empty JSON object")

    return {str(k): str(v) for k, v in prompts.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="media-sorter",
        description="Analyze media into structured records by default, with optional sorting workflows on top.",
    )
    parser.add_argument("source", help="Source folder containing media")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Optional destination folder for sorted media or analysis artifacts",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--model", default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of files")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum category confidence before the bundled sorter routes media into a category bucket",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "move", "none"],
        default="none",
        help="How to apply results: none (analysis only), copy, or move",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run without writing files")
    parser.add_argument(
        "--prompts",
        default=None,
        help='JSON dict for category prompts, e.g. {"portrait":"a studio portrait photo"}',
    )
    parser.add_argument(
        "--subject-prompts",
        default=None,
        help='JSON dict for subject detection, e.g. {"person":"a person","pet":"a pet","other":"other"}',
    )
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=None,
        help="Optional hard cap on sampled frames per video",
    )
    parser.add_argument(
        "--video-sampling-mode",
        choices=["second", "skip"],
        default="second",
        help="Video frame sampling strategy: time-based or fixed frame skip",
    )
    parser.add_argument(
        "--video-seconds-per-sample",
        type=float,
        default=1.0,
        help="In 'second' mode, sample one frame every N seconds",
    )
    parser.add_argument(
        "--video-frame-skip",
        type=int,
        default=10,
        help="In 'skip' mode, sample every Nth frame",
    )
    parser.add_argument(
        "--min-solo-frame-ratio",
        type=float,
        default=0.5,
        help="Minimum ratio of solo-person frames required when using the default people-oriented video filter",
    )
    parser.add_argument(
        "--min-person-confidence",
        type=float,
        default=0.35,
        help="Minimum subject confidence required for person class before solo routing",
    )
    parser.add_argument(
        "--min-solo-confidence",
        type=float,
        default=0.45,
        help="Minimum count-model confidence for exactly-one-person class",
    )
    parser.add_argument(
        "--min-solo-margin",
        type=float,
        default=0.05,
        help="Minimum margin between solo confidence and best non-solo confidence",
    )
    parser.add_argument(
        "--no-pet-sorting",
        action="store_true",
        help="Disable pet bucket and route pet media through normal flow",
    )
    parser.add_argument(
        "--face-sorting",
        action="store_true",
        help="Enable face-based sorting for accepted person media in the bundled sorter",
    )
    parser.add_argument(
        "--face-mode",
        choices=["unnamed", "tagged"],
        default="unnamed",
        help="Unnamed clustering or tagged identity matching",
    )
    parser.add_argument(
        "--face-tags-dir",
        default=None,
        help="Directory with reference faces for tagged mode: <dir>/<tag_name>/*.jpg",
    )
    parser.add_argument(
        "--face-similarity-threshold",
        type=float,
        default=0.82,
        help="Cosine similarity threshold for face matching/clustering",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write machine-readable output records for app indexing",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Manifest output path. Defaults under output directory",
    )
    parser.add_argument(
        "--manifest-format",
        choices=["jsonl", "json"],
        default="jsonl",
        help="Manifest format for integration pipelines",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Enable persistent SQLite media index for app backends",
    )
    parser.add_argument(
        "--index-db-path",
        default=None,
        help="Path to SQLite index database. Defaults under output directory",
    )
    parser.add_argument(
        "--index-mode",
        choices=["full", "update"],
        default="full",
        help="Indexing mode: full rebuild pass or update unchanged-aware pass",
    )
    parser.add_argument(
        "--index-prune-missing",
        action="store_true",
        help="When indexing, remove database entries no longer present in source",
    )
    return parser


def build_finalize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="media-sorter finalize",
        description="Export a compact app-ready bundle with a frozen image encoder, prompt embeddings, and a tiny runtime script.",
    )
    parser.add_argument("bundle_dir", help="Directory to write the finalized deployment bundle into")
    parser.add_argument("--device", default="cpu", help="Torch device to initialize before export; cpu is recommended")
    parser.add_argument("--model", default="ViT-B/32", help="CLIP model name")
    parser.add_argument(
        "--prompts",
        default=None,
        help='JSON dict for category prompts, e.g. {"portrait":"a studio portrait photo"}',
    )
    parser.add_argument(
        "--subject-prompts",
        default=None,
        help='JSON dict for subject detection, e.g. {"person":"a person","pet":"a pet","other":"other"}',
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum category confidence saved into the bundle metadata",
    )
    parser.add_argument(
        "--min-person-confidence",
        type=float,
        default=0.35,
        help="Minimum subject confidence for person class in the generated runner",
    )
    parser.add_argument(
        "--min-solo-confidence",
        type=float,
        default=0.45,
        help="Minimum solo confidence in the generated runner",
    )
    parser.add_argument(
        "--min-solo-margin",
        type=float,
        default=0.05,
        help="Minimum margin between solo and non-solo confidence in the generated runner",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT8 quantization and keep only the base ONNX model",
    )
    parser.add_argument(
        "--no-runner",
        action="store_true",
        help="Skip generating the standalone Python runner script",
    )
    return parser


def _run_finalize(argv: list[str]) -> int:
    parser = build_finalize_parser()
    try:
        args = parser.parse_args(argv)
        config = SorterConfig(
            source_dir=Path("."),
            output_dir=None,
            device=args.device,
            model_name=args.model,
            level_prompts=_parse_prompts(args.prompts),
            subject_prompts=_parse_subject_prompts(args.subject_prompts),
            min_category_confidence=args.min_confidence,
            min_person_confidence=args.min_person_confidence,
            min_solo_confidence=args.min_solo_confidence,
            min_solo_margin=args.min_solo_margin,
        )
        finalizer = BundleFinalizer(config)
        artifacts = finalizer.finalize(
            args.bundle_dir,
            quantize=not args.no_quantize,
            include_runner=not args.no_runner,
        )
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Finalize error:\n{exc}", file=sys.stderr)
        return 2

    print("Finalize complete")
    print(f"Bundle: {artifacts.bundle_dir}")
    print(f"Model: {artifacts.model_path}")
    if artifacts.quantized_model_path is not None:
        print(f"Quantized model: {artifacts.quantized_model_path}")
    print(f"Config: {artifacts.config_path}")
    print(f"Embeddings: {artifacts.embeddings_path}")
    print(f"Requirements: {artifacts.requirements_path}")
    if artifacts.runner_path.exists():
        print(f"Runner: {artifacts.runner_path}")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "finalize":
        return _run_finalize(sys.argv[2:])

    parser = build_parser()
    try:
        args = parser.parse_args()

        config = SorterConfig(
            source_dir=args.source,
            output_dir=args.output,
            device=args.device,
            model_name=args.model,
            limit=args.limit,
            level_prompts=_parse_prompts(args.prompts),
            subject_prompts=_parse_subject_prompts(args.subject_prompts),
            min_category_confidence=args.min_confidence,
            copy_mode=args.mode,
            dry_run=args.dry_run,
            write_manifest=args.write_manifest,
            manifest_path=args.manifest_path,
            manifest_format=args.manifest_format,
            enable_index=args.index,
            index_db_path=args.index_db_path,
            index_mode=args.index_mode,
            index_prune_missing=args.index_prune_missing,
            max_video_frames=args.max_video_frames,
            video_sampling_mode=args.video_sampling_mode,
            video_seconds_per_sample=args.video_seconds_per_sample,
            video_frame_skip=args.video_frame_skip,
            min_solo_frame_ratio=args.min_solo_frame_ratio,
            min_person_confidence=args.min_person_confidence,
            min_solo_confidence=args.min_solo_confidence,
            min_solo_margin=args.min_solo_margin,
            enable_pet_sorting=not args.no_pet_sorting,
            enable_face_sorting=args.face_sorting,
            face_mode=args.face_mode,
            face_tags_dir=args.face_tags_dir,
            face_similarity_threshold=args.face_similarity_threshold,
        )

        runner = MediaAnalyzer(config) if args.mode == "none" else MediaSorter(config)
        stats = runner.run()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Runtime setup error:\n{exc}", file=sys.stderr)
        print(f"\nPython in use: {sys.executable}", file=sys.stderr)
        print(
            f"Use interpreter-specific pip: {sys.executable} -m pip ...",
            file=sys.stderr,
        )
        return 2

    print("Processing complete")
    print(f"Total seen: {stats.total_seen}")
    print(f"Images: {stats.image_files}")
    print(f"Videos: {stats.video_files}")
    print(f"Skipped (index): {stats.skipped}")
    print(f"Errors: {stats.errors}")
    print("Counts:")
    for label, count in stats.as_dict().items():
        print(f"  {label}: {count}")
    if runner.manifest_output_path is not None:
        print(f"Manifest: {runner.manifest_output_path}")
    if runner.index_db_path is not None:
        print(f"Index DB: {runner.index_db_path}")
        print(f"Indexed rows updated: {runner.index_updated_count}")
        print(f"Indexed rows skipped: {runner.index_skipped_count}")
        if runner.index_pruned_count is not None:
            print(f"Index rows pruned: {runner.index_pruned_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

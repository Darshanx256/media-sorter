from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from typer.core import TyperGroup

from .config import SorterConfig
from .doctor import render_doctor_report
from .finalize import BundleFinalizer
from .pipeline import AnalysisStats, MediaAnalyzer, MediaSorter
from .prompt_packs import resolve_level_prompts, resolve_subject_prompts
from .i18n import _

ANALYZE_COMMANDS = {"analyze", "finalize", "doctor"}


class MediaSorterGroup(TyperGroup):
    def parse_args(self, ctx: typer.Context, args: list[str]) -> list[str]:
        adjusted_args = list(args)
        should_inject = (
            adjusted_args
            and not adjusted_args[0].startswith("-")
            and adjusted_args[0] not in ANALYZE_COMMANDS
        )
        if should_inject:
            adjusted_args.insert(0, "analyze")
        return super().parse_args(ctx, adjusted_args)


app = typer.Typer(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    cls=MediaSorterGroup,
)
console = Console()
Runner = MediaAnalyzer | MediaSorter


def _format_path(value: Path | None) -> str:
    return str(value) if value is not None else "-"


async def _run_with_progress(runner: Runner, file_paths: list[Path]) -> AnalysisStats:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    description = _("Analyzing media")
    with progress:
        task_id = progress.add_task(description, total=len(file_paths))

        def progress_callback(current_path: Path, stats: AnalysisStats) -> None:
            progress.update(
                task_id,
                advance=1,
                description=f"{description} ({current_path.name})",
            )

        stats = await runner.run_async(progress_callback=progress_callback, file_paths=file_paths)
    return stats


async def _run_main_async(config: SorterConfig, mode: str) -> tuple[AnalysisStats, Runner, list[Path]]:
    if mode == "none":
        runner = MediaAnalyzer(config)
        file_paths = await runner._collect_input_files_async()
    else:
        sorter = MediaSorter(config)
        runner = sorter
        file_paths = await sorter.analyzer._collect_input_files_async()
    stats = await _run_with_progress(runner, file_paths)
    return stats, runner, file_paths


def _print_summary(runner: Runner, stats: AnalysisStats, file_paths: list[Path]) -> None:
    table = Table(title=_("Run summary"), box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Discovered files", str(len(file_paths)))
    table.add_row("Processed", str(stats.total_seen))
    table.add_row("Images", str(stats.image_files))
    table.add_row("Videos", str(stats.video_files))
    if stats.skipped:
        table.add_row("Skipped", str(stats.skipped))
    if stats.errors:
        table.add_row("Errors", str(stats.errors))
    if stats.duplicates:
        table.add_row("Duplicates", str(stats.duplicates))

    counts = stats.as_dict()
    for label in sorted(counts):
        if label in {"ok", "error", "skipped", "duplicate"}:
            continue
        count = counts[label]
        if count:
            table.add_row(label, str(count))

    if runner.manifest_output_path is not None:
        table.add_row("Manifest", _format_path(runner.manifest_output_path))

    if runner.index_db_path is not None:
        table.add_row("Index DB", _format_path(runner.index_db_path))
        table.add_row("Index updates", str(runner.index_updated_count))
        table.add_row("Index skips", str(runner.index_skipped_count))
        if runner.index_pruned_count is not None:
            table.add_row("Index pruned", str(runner.index_pruned_count))

    console.print(table)


def _print_dry_run_table(sorter: MediaSorter) -> None:
    if not sorter.planned_moves:
        console.print(_("Dry run detected no file operations."))
        return

    console.rule("[yellow]Dry-run preview")
    table = Table(show_lines=True)
    table.add_column("Action", style="bold")
    table.add_column("Source", style="dim")
    table.add_column("Destination", style="green")

    for source, destination, action in sorter.planned_moves:
        table.add_row(action.capitalize(), str(source), str(destination))

    console.print(table)


def _run_analyze(
    source: Path,
    output: Path | None,
    device: str,
    model: str,
    limit: int | None,
    min_confidence: float,
    mode: str,
    dry_run: bool,
    prompts: str | None,
    prompts_file: Path | None,
    subject_prompts: str | None,
    subject_prompts_file: Path | None,
    max_video_frames: int | None,
    video_sampling_mode: str,
    video_seconds_per_sample: float,
    video_frame_skip: int,
    min_solo_frame_ratio: float,
    min_person_confidence: float,
    min_solo_confidence: float,
    min_solo_margin: float,
    no_pet_sorting: bool,
    face_sorting: bool,
    face_mode: str,
    face_tags_dir: Path | None,
    face_similarity_threshold: float,
    write_manifest: bool,
    manifest_path: Path | None,
    manifest_format: str,
    index: bool,
    index_db_path: Path | None,
    index_mode: str,
    index_prune_missing: bool,
    dedup_similarity: float,
    disable_deduplication: bool,
) -> None:
    try:
        level_prompts = resolve_level_prompts(inline_json=prompts, prompts_path=prompts_file)
        subject_data = resolve_subject_prompts(inline_json=subject_prompts, prompts_path=subject_prompts_file)

        config = SorterConfig(
            source_dir=source,
            output_dir=output,
            device=device,
            model_name=model,
            limit=limit,
            level_prompts=level_prompts,
            subject_prompts=subject_data,
            min_category_confidence=min_confidence,
            copy_mode=mode,
            dry_run=dry_run,
            write_manifest=write_manifest,
            manifest_path=manifest_path,
            manifest_format=manifest_format,
            enable_index=index,
            index_db_path=index_db_path,
            index_mode=index_mode,
            index_prune_missing=index_prune_missing,
            max_video_frames=max_video_frames,
            video_sampling_mode=video_sampling_mode,
            video_seconds_per_sample=video_seconds_per_sample,
            video_frame_skip=video_frame_skip,
            min_solo_frame_ratio=min_solo_frame_ratio,
            min_person_confidence=min_person_confidence,
            min_solo_confidence=min_solo_confidence,
            min_solo_margin=min_solo_margin,
            enable_pet_sorting=not no_pet_sorting,
            enable_face_sorting=face_sorting,
            face_mode=face_mode,
            face_tags_dir=face_tags_dir,
            face_similarity_threshold=face_similarity_threshold,
            enable_deduplication=not disable_deduplication,
            dedup_similarity_threshold=dedup_similarity,
        )
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(code=2)

    try:
        stats, runner, file_paths = asyncio.run(_run_main_async(config, mode))
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(code=2)
    except RuntimeError as exc:
        console.print(f"[red]Runtime error:[/] {exc}")
        raise typer.Exit(code=2)

    console.rule(_("Processing complete"))
    _print_summary(runner, stats, file_paths)
    if isinstance(runner, MediaSorter) and config.copy_mode != "none" and config.dry_run:
        _print_dry_run_table(runner)


@app.command(
    "analyze",
    help=_(
        "Alias for the default analyze workflow (use when the source path matches a subcommand name)."
    ),
)
def analyze(
    source: Path = typer.Argument(..., help=_("Source folder containing media")),
    output: Path | None = typer.Argument(None, help=_("Optional destination folder for sorted media or analysis artifacts")),
    device: str = typer.Option("cpu", help=_("Torch device, e.g. cpu or cuda")),
    model: str = typer.Option("ViT-B/32", help=_("CLIP model name")),
    limit: int | None = typer.Option(None, help=_("Optional max number of files")),
    min_confidence: float = typer.Option(0.0, help=_("Minimum category confidence before sorting")),
    mode: str = typer.Option("none", case_sensitive=False, help=_("How to apply results: none (analysis only), copy, or move")),
    dry_run: bool = typer.Option(False, help=_("Run without writing files and show a preview of actions")),
    prompts: str | None = typer.Option(None, help=_('JSON dict for category prompts, e.g. {"portrait":"a studio portrait photo"}')),
    prompts_file: Path | None = typer.Option(None, help=_("Path to a JSON or YAML category prompt pack")),
    subject_prompts: str | None = typer.Option(None, help=_('JSON dict for subject detection')),
    subject_prompts_file: Path | None = typer.Option(None, help=_("Path to a subject prompt pack")),
    max_video_frames: int | None = typer.Option(None, help=_("Optional hard cap on sampled frames per video")),
    video_sampling_mode: str = typer.Option("second", case_sensitive=False, help=_("Video frame sampling strategy")),
    video_seconds_per_sample: float = typer.Option(1.0, help=_("Seconds between samples when using second mode")),
    video_frame_skip: int = typer.Option(10, help=_("Frame skip when using skip mode")),
    min_solo_frame_ratio: float = typer.Option(0.5, help=_("Minimum ratio of solo-person frames")),
    min_person_confidence: float = typer.Option(0.35, help=_("Minimum person confidence before solo routing")),
    min_solo_confidence: float = typer.Option(0.45, help=_("Minimum solo confidence for one person")),
    min_solo_margin: float = typer.Option(0.05, help=_("Minimum margin between solo and non-solo")),
    no_pet_sorting: bool = typer.Option(False, help=_("Disable the pet bucket")),
    face_sorting: bool = typer.Option(False, help=_("Enable face-based sorting")),
    face_mode: str = typer.Option("unnamed", case_sensitive=False, help=_("Face sorting mode: unnamed or tagged")),
    face_tags_dir: Path | None = typer.Option(None, help=_("Directory with reference faces for tagged mode")),
    face_similarity_threshold: float = typer.Option(0.82, help=_("Cosine similarity threshold for face matching")),
    write_manifest: bool = typer.Option(False, help=_("Write machine-readable manifest records")),
    manifest_path: Path | None = typer.Option(None, help=_("Manifest output path")),
    manifest_format: str = typer.Option("jsonl", case_sensitive=False, help=_("Manifest format: jsonl or json")),
    index: bool = typer.Option(False, help=_("Enable persistent SQLite media index")),
    index_db_path: Path | None = typer.Option(None, help=_("Path to SQLite index database")),
    index_mode: str = typer.Option("full", case_sensitive=False, help=_("Index mode: full or update")),
    index_prune_missing: bool = typer.Option(False, help=_("Remove missing entries when indexing")),
    dedup_similarity: float = typer.Option(0.98, help=_("CLIP similarity threshold for near-duplicate detection")),
    disable_deduplication: bool = typer.Option(False, help=_("Turn off multi-tier deduplication")),
) -> None:
    _run_analyze(
        source,
        output,
        device,
        model,
        limit,
        min_confidence,
        mode,
        dry_run,
        prompts,
        prompts_file,
        subject_prompts,
        subject_prompts_file,
        max_video_frames,
        video_sampling_mode,
        video_seconds_per_sample,
        video_frame_skip,
        min_solo_frame_ratio,
        min_person_confidence,
        min_solo_confidence,
        min_solo_margin,
        no_pet_sorting,
        face_sorting,
        face_mode,
        face_tags_dir,
        face_similarity_threshold,
        write_manifest,
        manifest_path,
        manifest_format,
        index,
        index_db_path,
        index_mode,
        index_prune_missing,
        dedup_similarity,
        disable_deduplication,
    )


@app.command("finalize", help=_("Export an image-first bundle with frozen prompts and ONNX models."))
def finalize(
    bundle_dir: Path = typer.Argument(..., help=_("Directory to write the finalized bundle into")),
    device: str = typer.Option("cpu", help=_("Torch device to initialize before export")),
    model: str = typer.Option("ViT-B/32", help=_("CLIP model name")),
    prompts: str | None = typer.Option(None, help=_('JSON dict for category prompts in the bundle')),
    prompts_file: Path | None = typer.Option(None, help=_("Path to a JSON or YAML category prompt pack")),
    subject_prompts: str | None = typer.Option(None, help=_('JSON dict for subject detection prompts')),
    subject_prompts_file: Path | None = typer.Option(None, help=_("Path to a subject prompt pack")),
    min_confidence: float = typer.Option(0.0, help=_("Minimum category confidence saved to the bundle")),
    min_person_confidence: float = typer.Option(0.35, help=_("Minimum person confidence recorded in the runner")),
    min_solo_confidence: float = typer.Option(0.45, help=_("Minimum solo confidence recorded in the runner")),
    min_solo_margin: float = typer.Option(0.05, help=_("Minimum solo margin recorded in the runner")),
    no_quantize: bool = typer.Option(False, help=_("Skip exporting the INT8 quantized model")),
    no_runner: bool = typer.Option(False, help=_("Skip generating the standalone Python runner script")),
) -> None:
    try:
        level_prompts = resolve_level_prompts(inline_json=prompts, prompts_path=prompts_file)
        subject_data = resolve_subject_prompts(inline_json=subject_prompts, prompts_path=subject_prompts_file)

        config = SorterConfig(
            source_dir=Path("."),
            output_dir=None,
            device=device,
            model_name=model,
            level_prompts=level_prompts,
            subject_prompts=subject_data,
            min_category_confidence=min_confidence,
            min_person_confidence=min_person_confidence,
            min_solo_confidence=min_solo_confidence,
            min_solo_margin=min_solo_margin,
        )
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(code=2)

    try:
        finalizer = BundleFinalizer(config)
        artifacts = finalizer.finalize(bundle_dir, quantize=not no_quantize, include_runner=not no_runner)
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(code=2)
    except RuntimeError as exc:
        console.print(f"[red]Finalize error:[/]\n{exc}")
        raise typer.Exit(code=2)

    console.print("[bold green]Finalize complete[/]")
    console.print(_("Bundle:"), artifacts.bundle_dir)
    console.print(_("Model:"), artifacts.model_path)
    if artifacts.quantized_model_path is not None:
        console.print(_("Quantized model:"), artifacts.quantized_model_path)
    console.print(_("Config:"), artifacts.config_path)
    console.print(_("Embeddings:"), artifacts.embeddings_path)
    console.print(_("Requirements:"), artifacts.requirements_path)
    if artifacts.runner_path.exists():
        console.print(_("Runner:"), artifacts.runner_path)


@app.command("doctor", help=_("Inspect the Python environment for ML, finalize, and optional video readiness."))
def doctor(
    expect_video: bool = typer.Option(False, help=_("Treat video runtime support as expected")),
    expect_finalize: bool = typer.Option(False, help=_("Treat finalize/export support as expected")),
) -> None:
    console.print(
        render_doctor_report(expect_video=expect_video, expect_finalize=expect_finalize),
        end="",
    )

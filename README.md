# Media Sorter

`media-sorter` is a Python package and CLI for analyzing images and videos into structured records that applications can use for tagging, indexing, moderation, search, routing, and optional folder-based sorting.

License: MIT. See [LICENSE](LICENSE).

## At A Glance

- default behavior: analyze media and return structured records
- optional built-in workflow: sort files into folders
- optional deployment path: export a compact finalize bundle for app use
- supports images and videos in the authoring/runtime library
- finalized bundle currently targets image inference first

## Overview

`media-sorter` is designed as a middle layer in a media pipeline.

By default, it does not assume your application wants files copied into category folders. Instead, it favors an analysis-first workflow:

1. scan media files
2. extract file and media metadata
3. score media against configurable prompt sets
4. return structured records
5. let your application decide what to do next

That makes it useful for many app shapes:

- media ingestion backends
- internal review tools
- search and retrieval systems
- moderation pipelines
- curation workflows
- desktop automation
- apps that want classification metadata but not file movement

The package still includes a bundled `MediaSorter` for users who do want folder routing, but sorting is now an optional layer built on top of analysis instead of the default identity of the project.

## Choose The Right API

Use `MediaAnalyzer` if:

- your app wants metadata, scores, and structured records
- your app has its own business logic
- you do not want file movement by default

Use `MediaSorter` if:

- you want the built-in routing policy
- you want optional copy or move behavior
- category folders are part of your intended workflow

Use `MediaClassifier` if:

- you only want prompt scoring
- you want direct access to embeddings
- you are building custom logic below the analyzer layer

Use `BundleFinalizer` if:

- prompts and thresholds are already settled
- another app or team needs a compact deployment artifact
- you want to avoid shipping the full authoring stack into production

## Why This Design

Many media libraries are either:

- too low-level, forcing every app to rebuild the same metadata and scoring plumbing
- too opinionated, assuming a fixed folder-routing or end-user workflow

`media-sorter` aims for the middle:

- higher-level than raw model calls
- lower-level than a hardcoded app workflow
- configurable through prompts and thresholds
- usable from both Python code and the CLI

## Core Concepts

### `MediaAnalyzer`

`MediaAnalyzer` is the primary library surface.

It recursively scans supported media files and produces structured `MediaRecord` objects containing:

- source path
- media type
- file metadata
- image or video metadata
- EXIF data for images when available
- prompt score maps
- selected subject and category
- optional solo-person signal
- optional face identity signal
- sampled frame indexes for videos
- status and error information

This is the main API to use when your application wants to own the business logic.

### `MediaSorter`

`MediaSorter` is the bundled opinionated workflow built on top of the analyzer.

It reuses the same analysis data, then applies built-in routing rules such as:

- route pet media to `pets/`
- route non-accepted media to `ignored/`
- route accepted category matches to category folders
- optionally route person media to `faces/<identity>/`

Use it when you want a ready-made sorting pipeline, not when you want to build your own policy.

### `MediaClassifier`

`MediaClassifier` is the lower-level scoring layer.

Use it when you want:

- direct image prediction
- prompt scoring for your own prompt sets
- CLIP-style image embeddings

### `BundleFinalizer`

`BundleFinalizer` turns a configured development setup into a smaller deployment bundle for applications.

Instead of shipping the whole authoring stack into production, the finalize flow exports:

- an ONNX image encoder
- an optional quantized ONNX image encoder
- frozen prompt embeddings
- deployment config
- a minimal runtime script

This is the recommended path when a team has already settled on prompts and thresholds and wants a more portable inference bundle.

## Installation

### Base Package

```bash
pip install -e .
```

The base install contains the Python package plus lightweight runtime dependencies such as `numpy` and `Pillow`.

### ML Backend

The package intentionally does not auto-install `torch` or `open-clip-torch`, because users may want different CPU or GPU builds.

CPU example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install open-clip-torch
```

GPU example:

```bash
# choose the correct command for your CUDA build from pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/<your-cuda-build>
pip install open-clip-torch
```

### Video And Face Support

```bash
pip install -e .[video]
```

The `video` extra installs `opencv-python-headless`.

### Finalize Support

```bash
pip install -e .[finalize]
```

The finalize flow also needs the ML backend installed at export time:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install open-clip-torch
```

The `finalize` extra installs the ONNX export/runtime stack used to generate deployment bundles.

### Guided Installer

```bash
./scripts/install.sh
```

## Quick Start

### Analyze Media Without Sorting

```python
from media_sorter import MediaAnalyzer, SorterConfig

config = SorterConfig(
    source_dir="./input_media",
    output_dir=None,
    device="cpu",
)

analyzer = MediaAnalyzer(config)
stats = analyzer.run()

print(stats.as_dict())

record = analyzer.records[0]
print(record.source)
print(record.media_type)
print(record.metadata.width, record.metadata.height)
print(record.subject, record.subject_confidence)
print(record.subject_scores)
print(record.category, record.category_confidence)
print(record.category_scores)
```

Typical record shape:

```python
{
    "source": "./input_media/example.jpg",
    "media_type": "image",
    "metadata": {
        "file_name": "example.jpg",
        "extension": ".jpg",
        "width": 1024,
        "height": 768,
    },
    "subject": "person",
    "subject_confidence": 0.91,
    "subject_scores": {"person": 0.91, "pet": 0.02, "other": 0.07},
    "category": "portrait",
    "category_confidence": 0.73,
    "category_scores": {"portrait": 0.73, "outdoor": 0.19, "product": 0.08},
    "status": "ok",
}
```

### Score Your Own Prompts Directly

```python
from pathlib import Path

from media_sorter import MediaClassifier, SorterConfig

config = SorterConfig(source_dir=".", output_dir=None, device="cpu")
classifier = MediaClassifier(config)

scores = classifier.score_image_prompts(
    Path("./example.jpg"),
    {
        "portrait": "a portrait photo",
        "outdoor": "a person outdoors",
        "product": "a product photo on a clean background",
    },
)

print(scores)
```

### Use The Bundled Sorter

```python
from media_sorter import MediaSorter, SorterConfig

config = SorterConfig(
    source_dir="./input_media",
    output_dir="./sorted_output",
    copy_mode="copy",
    device="cpu",
)

sorter = MediaSorter(config)
stats = sorter.run()
print(stats.as_dict())
```

## CLI Usage

### Analyze Only

Analysis-only is the default mode:

```bash
media-sorter ./input_media
```

That analyzes media and prints a summary without copying or moving files.

### Write A Manifest

```bash
media-sorter ./input_media \
  --write-manifest \
  --manifest-format jsonl
```

If no output directory is supplied, analysis artifacts are written under:

```text
./input_media/.media_sorter/
```

### Sort Into Folders

```bash
media-sorter ./input_media ./sorted_output --mode copy
```

Or move files instead:

```bash
media-sorter ./input_media ./sorted_output --mode move
```

### Index Results In SQLite

```bash
media-sorter ./input_media \
  --index \
  --index-mode update
```

To remove missing sources from the index:

```bash
media-sorter ./input_media \
  --index \
  --index-mode update \
  --index-prune-missing
```

### Finalize A Deployment Bundle

```bash
media-sorter finalize ./app_bundle
```

Useful options:

- `--no-quantize` keeps only the base ONNX model
- `--no-runner` skips generating the standalone runtime script
- `--prompts` freezes a custom category prompt set into the bundle
- `--prompts-file` loads category prompts from JSON or YAML
- `--subject-prompts` freezes custom subject prompts into the bundle
- `--subject-prompts-file` loads subject prompts from JSON or YAML

The finalize command is intended for the point where prompts and thresholds are already chosen and a developer wants a smaller, more portable output for application integration.

In practice, the quantized model is usually the deployment artifact you actually ship.

### Check The Environment

```bash
media-sorter doctor
media-sorter doctor --expect-finalize --expect-video
```

The doctor command reports:

- Python executable and version
- ML backend readiness
- finalize/export readiness
- optional video runtime readiness
- actionable install hints for missing pieces

### Load Prompt Packs From Files

Inline JSON flags still work, but file-based prompt packs are easier to maintain for real projects.

Category prompt pack example:

```yaml
level_prompts:
  portrait: a portrait photo of one person
  group: a photo of multiple people together
  food: a photo of food or a meal
```

Subject prompt pack example:

```yaml
subject_prompts:
  person: a photo of a person
  pet: a photo of a pet animal
  other: a photo of a scene or object without people or pets
```

Use them from the CLI:

```bash
media-sorter ./input_media \
  --prompts-file ./category_prompts.yaml \
  --subject-prompts-file ./subject_prompts.yaml
```

Prompt precedence is deterministic:

- inline JSON flags override prompt-pack files
- prompt-pack files override built-in defaults

## Public API

### `MediaAnalyzer`

Use `MediaAnalyzer` when your application wants structured output and will decide the downstream behavior itself.

Responsibilities:

- recursive source scanning
- image and video analysis
- metadata extraction
- prompt-based scoring
- manifest writing
- SQLite indexing

### `MediaSorter`

Use `MediaSorter` when you want built-in routing and optional file movement.

Responsibilities:

- consume analyzer output
- apply bundled routing rules
- copy or move accepted files
- emit manifests and index records for the routed workflow

### `MediaClassifier`

Useful methods:

- `predict_image(...)`
- `predict_pil(...)`
- `score_image_prompts(...)`
- `score_pil_prompts(...)`
- `embedding(...)`

### `BundleFinalizer`

Use `BundleFinalizer` when you want to turn a configured authoring setup into a compact deployment artifact for another application or team.

## Data Model

`MediaAnalyzer` and `MediaSorter` both produce `MediaRecord` objects.

Important fields:

- `source`
- `media_type`
- `destination`
- `route_label`
- `metadata`
- `subject`
- `subject_confidence`
- `subject_scores`
- `is_solo_person`
- `solo_label`
- `solo_confidence`
- `count_scores`
- `category`
- `category_confidence`
- `category_scores`
- `face_identity`
- `sampled_frames`
- `status`
- `error`

The nested `MediaMetadata` object contains:

- `file_name`
- `extension`
- `file_size`
- `mtime_ns`
- `width`
- `height`
- `frame_count`
- `fps`
- `duration_seconds`
- `exif`

## Prompt Model

The package currently organizes scoring around three prompt groups:

- subject prompts
- count prompts
- category prompts

By default:

- subject prompts identify broad subject type such as person, pet, or other
- count prompts estimate whether an image looks like no people, exactly one person, or multiple people
- category prompts represent your application-specific classes

Important note:

- the shipped default category prompts are now a general gallery-oriented starter pack:
  `portrait`, `group`, `pet`, `food`, `travel`, `outdoor`, `document`, `screenshot`, `product`, `art`, `meme`, `other`
- for real application use, you will still get better results if you provide category prompts tuned to your own domain
- the library becomes much more useful once the prompt set reflects your actual domain

The bundled sorter uses those groups to make routing decisions, but applications can ignore the built-in routing and use the raw score maps directly.

## Optional People-Oriented Policy

The library still includes people-oriented controls because they are useful for some workflows, but they are now optional policy knobs rather than the meaning of the whole package.

Relevant flags and config values include:

- `--min-person-confidence`
- `--min-solo-confidence`
- `--min-solo-margin`
- `--min-solo-frame-ratio`
- `--no-pet-sorting`
- `--face-sorting`
- `--face-mode`
- `--face-tags-dir`

If your application does not need these behaviors, you can still use the analyzer and raw scores without relying on the bundled sorter’s routing policy.

## Finalize Workflow

The finalize flow is designed for app deployment, not authoring.

It currently favors:

- image inference first
- frozen prompts at export time
- precomputed text embeddings
- ONNX Runtime for the generated runner
- a compact runtime handoff for other apps

The generated bundle currently contains:

- `image_encoder.onnx`
- `image_encoder.int8.onnx` when quantization is enabled
- `text_embeddings.npz`
- `config.json`
- `requirements.txt`
- `run_media_sort.py`

Typical bundle layout:

```text
app_bundle/
  config.json
  image_encoder.onnx
  image_encoder.int8.onnx
  requirements.txt
  run_media_sort.py
  text_embeddings.npz
```

### What Gets Frozen Into The Bundle

The bundle stores:

- preprocess settings
- prompt labels
- prompt text
- thresholds
- routing-related labels
- feature flags for the exported runtime

That means deployment consumers do not need to rebuild the prompt set or the text encoder during normal runtime use.

### Finalize Validation

Finalize now validates the bundle before returning success:

- required files must exist
- `config.json` must reference valid bundle files
- embedding matrix sizes must match configured labels
- the generated runner is smoke-tested against a tiny image when the runner is included

That keeps finalize closer to a release workflow instead of a best-effort export only.

### Current Bundle Scope

The finalized runtime is currently best thought of as an image-first deployment path.

Current strengths:

- easy app integration
- small standalone runner script
- quantized model option
- frozen prompt embeddings
- JSON output from the generated runtime

Current limitations:

- prompts are frozen at export time
- video analysis is not implemented in the generated standalone runtime
- face sorting is not implemented in the generated standalone runtime
- the unquantized base ONNX model can still be large, so the quantized model is usually the practical runtime artifact

## CLI Reference

### Analysis Command

Positional arguments:

- `source`: source directory to scan recursively
- `output`: optional output directory for sorted results or analysis artifacts

Important flags:

- `--mode`: `none`, `copy`, or `move`
- `--device`: torch device such as `cpu` or `cuda`
- `--model`: CLIP model name
- `--limit`: maximum number of files
- `--dry-run`: compute results without writing files
- `--write-manifest`: write JSON or JSONL records
- `--index`: enable SQLite indexing

Prompt flags:

- `--prompts`
- `--prompts-file`
- `--subject-prompts`
- `--subject-prompts-file`

Video flags:

- `--video-sampling-mode`
- `--video-seconds-per-sample`
- `--video-frame-skip`
- `--max-video-frames`

### Finalize Command

```bash
media-sorter finalize ./bundle_dir
```

Important flags:

- `--device`
- `--model`
- `--prompts`
- `--prompts-file`
- `--subject-prompts`
- `--subject-prompts-file`
- `--min-confidence`
- `--min-person-confidence`
- `--min-solo-confidence`
- `--min-solo-margin`
- `--no-quantize`
- `--no-runner`

### Doctor Command

```bash
media-sorter doctor --expect-finalize --expect-video
```

Important flags:

- `--expect-finalize`
- `--expect-video`

## Artifact Placement

When analysis artifacts are requested without an explicit output directory, the library writes them to:

```text
<source_dir>/.media_sorter/
```

This keeps manifests and index files available without forcing a visible sorting workflow into the source tree.

## Project Layout

```text
src/media_sorter/
  __init__.py
  __main__.py
  cli.py
  config.py
  core.py
  finalize.py
  index.py
  pipeline.py
```

## Development Notes

- Python requirement: `>=3.10`
- package layout: `src/`
- build configuration: `pyproject.toml`
- current deployment export path: ONNX + ONNX Runtime
- automated test suite now covers config validation, prompt packs, CLI surfaces, analyzer/sorter smoke, finalize validation, and optional secondary-path smoke tests

Because the project uses prompt-driven zero-shot classification, you should validate prompts and thresholds against representative media from your own domain before relying on a particular routing or moderation policy.

## Current State Of The Project

Today the package supports three practical usage modes:

- analysis-first middleware for applications
- optional bundled folder sorting
- app-friendly finalize bundles for image inference deployment

That makes the project much more flexible than a plain folder sorter while still keeping a ready-made sorting workflow available for users who want it.

## License

This repository and its packaged distributions are released under the MIT License. See [LICENSE](LICENSE) for the full text.

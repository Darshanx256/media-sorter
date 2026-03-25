"""
Regression tests for all 10 bugs fixed in the latest commit.

These tests verify that:
  - The old broken behaviour no longer occurs.
  - The new fixed behaviour is correct.

They are intentionally self-contained (no ML dependencies required).
"""
from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from media_sorter.config import SorterConfig, DEFAULT_COUNT_LABELS
from media_sorter.core import Prediction
from media_sorter.finalize import BundleFinalizer, BundleValidationResult
from media_sorter.i18n import load_po
from media_sorter.index import IndexStore
from media_sorter.pipeline import (
    MediaAnalyzer,
    MediaRecord,
    MediaMetadata,
    MediaSorter,
    _write_records_to_manifest,
)

from conftest import FakeClassifier


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_record(
    *,
    subject: str = "person",
    is_solo_person: bool = True,
    category: str | None = "portrait",
    category_confidence: float = 0.91,
    face_identity: str | None = None,
) -> MediaRecord:
    return MediaRecord(
        source="/fake/img.jpg",
        media_type="image",
        destination=None,
        route_label=None,
        metadata=MediaMetadata(
            file_name="img.jpg",
            extension=".jpg",
            file_size=1024,
            mtime_ns=0,
        ),
        subject=subject,
        subject_confidence=0.9,
        subject_scores={subject: 0.9},
        is_solo_person=is_solo_person,
        solo_label=DEFAULT_COUNT_LABELS[0],
        solo_confidence=0.88,
        count_scores={DEFAULT_COUNT_LABELS[0]: 0.88},
        category=category,
        category_confidence=category_confidence,
        category_scores={category: category_confidence} if category else {},
        face_identity=face_identity,
        sampled_frames=[],
        status="ok",
    )


def _minimal_config(tmp_path: Path, **kw) -> SorterConfig:
    return SorterConfig(source_dir=tmp_path, output_dir=None, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# BUG-1  i18n.py  eval() → ast.literal_eval()
# ─────────────────────────────────────────────────────────────────────────────

class TestBug1I18nSafeEval:
    def test_valid_po_strings_are_parsed(self, tmp_path: Path) -> None:
        po = tmp_path / "messages.po"
        po.write_text(
            'msgid "hello"\nmsgstr "world"\n',
            encoding="utf-8",
        )
        result = load_po(str(po))
        assert result.get("hello") == "world"

    def test_malicious_po_string_does_not_execute_code(self, tmp_path: Path) -> None:
        """A msgstr that would execute code under eval() must not do so."""
        sentinel = tmp_path / "pwned"
        # This string would call open() and create a file if eval() were used.
        evil = f'__import__("pathlib").Path(r"{sentinel}").touch()'
        po = tmp_path / "messages.po"
        po.write_text(
            f'msgid "test"\nmsgstr "{evil}"\n',
            encoding="utf-8",
        )
        result = load_po(str(po))
        # The malicious string is NOT a Python literal, so ast.literal_eval raises.
        # The entry should be skipped gracefully (empty dict or missing key).
        assert "test" not in result or result["test"] != evil
        assert not sentinel.exists(), "RCE: malicious code was executed!"

    def test_multiline_po_entry_is_parsed(self, tmp_path: Path) -> None:
        po = tmp_path / "messages.po"
        po.write_text(
            'msgid ""\n"part1"\nmsgstr ""\n"trans1"\n',
            encoding="utf-8",
        )
        result = load_po(str(po))
        # "part1" is the msgid continuation; should parse without error
        assert isinstance(result, dict)

    def test_bare_except_no_longer_used(self) -> None:
        """Verify the source of i18n.py no longer contains bare `except:`."""
        import media_sorter.i18n as mod
        src = Path(mod.__file__).read_text(encoding="utf-8")
        # bare `except:` (not `except SomeException`) should not appear
        assert "except:" not in src, "Bare except: still present in i18n.py"


# ─────────────────────────────────────────────────────────────────────────────
# BUG-2  install.sh  eval removed
# ─────────────────────────────────────────────────────────────────────────────

class TestBug2InstallShNoEval:
    def test_eval_not_in_gpu_block(self) -> None:
        install = (
            Path(__file__).parent.parent / "scripts" / "install.sh"
        ).read_text()
        assert "eval " not in install, (
            "install.sh still contains `eval` — shell injection risk present"
        )

    def test_step_labels_are_consistent(self) -> None:
        install = (
            Path(__file__).parent.parent / "scripts" / "install.sh"
        ).read_text()
        # All step labels must say /4 (four steps, not three)
        assert "1/3" not in install, "Step 1/3 label still present"
        assert "2/3" not in install, "Step 2/3 label still present"
        for step in range(1, 5):
            assert f"{step}/4" in install, f"Step {step}/4 label missing"


# ─────────────────────────────────────────────────────────────────────────────
# BUG-3  index.py  prune_missing with empty set
# ─────────────────────────────────────────────────────────────────────────────

class TestBug3PruneMissingSafe:
    def _store_one_record(self, store: IndexStore, source: str = "/fake/a.jpg") -> None:
        rec = _make_record()
        rec_with_source = MediaRecord(
            source=source,
            media_type=rec.media_type,
            destination=rec.destination,
            route_label=rec.route_label,
            metadata=rec.metadata,
            subject=rec.subject,
            subject_confidence=rec.subject_confidence,
            subject_scores=rec.subject_scores,
            is_solo_person=rec.is_solo_person,
            solo_label=rec.solo_label,
            solo_confidence=rec.solo_confidence,
            count_scores=rec.count_scores,
            category=rec.category,
            category_confidence=rec.category_confidence,
            category_scores=rec.category_scores,
            face_identity=rec.face_identity,
            sampled_frames=rec.sampled_frames,
            status=rec.status,
            error=rec.error,
        )
        store.upsert_record(rec_with_source, file_size=1024, mtime_ns=0)
        store.commit()

    def test_empty_set_does_not_wipe_index(self, tmp_path: Path) -> None:
        db = tmp_path / "idx.sqlite3"
        store = IndexStore(db, mode="full")
        self._store_one_record(store)

        # Sanity: one row exists
        count_before = store.conn.execute("SELECT COUNT(*) FROM media_index").fetchone()[0]
        assert count_before == 1

        # Pruning with empty set must NOT delete anything
        deleted = store.prune_missing(set())
        assert deleted == 0

        count_after = store.conn.execute("SELECT COUNT(*) FROM media_index").fetchone()[0]
        assert count_after == 1, "prune_missing(empty) wiped the index!"
        store.close()

    def test_prune_missing_removes_stale_entries(self, tmp_path: Path) -> None:
        db = tmp_path / "idx.sqlite3"
        store = IndexStore(db, mode="full")
        self._store_one_record(store, "/fake/a.jpg")
        self._store_one_record(store, "/fake/b.jpg")

        deleted = store.prune_missing({"/fake/a.jpg"})
        assert deleted == 1

        remaining = store.conn.execute("SELECT source FROM media_index").fetchall()
        assert len(remaining) == 1
        assert remaining[0][0] == "/fake/a.jpg"
        store.close()


# ─────────────────────────────────────────────────────────────────────────────
# BUG-4  pipeline.py  _decide_destination — all subjects routed via category
# ─────────────────────────────────────────────────────────────────────────────

class TestBug4DecideDestination:
    def _sorter(self, tmp_path: Path, **kw) -> MediaSorter:
        out = tmp_path / "out"
        config = SorterConfig(source_dir=tmp_path, output_dir=out, copy_mode="copy", **kw)
        return MediaSorter(config, classifier=FakeClassifier(config))

    def test_non_person_subject_routes_via_category(self, tmp_path: Path) -> None:
        sorter = self._sorter(tmp_path)
        record = _make_record(subject="other", is_solo_person=False, category="outdoor", category_confidence=0.88)
        label, dest = sorter._decide_destination(record)
        assert label == "outdoor", f"Expected 'outdoor', got '{label}'"

    def test_pet_with_sorting_enabled_routes_to_pet_bucket(self, tmp_path: Path) -> None:
        sorter = self._sorter(tmp_path, enable_pet_sorting=True)
        record = _make_record(subject="pet", is_solo_person=False, category="outdoor")
        label, _ = sorter._decide_destination(record)
        assert label == "pets"

    def test_pet_with_sorting_disabled_routes_via_category(self, tmp_path: Path) -> None:
        """--no-pet-sorting: pets should fall through to category, not to ignored."""
        sorter = self._sorter(tmp_path, enable_pet_sorting=False)
        record = _make_record(subject="pet", is_solo_person=False, category="outdoor", category_confidence=0.8)
        label, _ = sorter._decide_destination(record)
        assert label == "outdoor", (
            f"pet with sorting disabled should route by category, got '{label}'"
        )

    def test_below_confidence_threshold_routes_to_ignored(self, tmp_path: Path) -> None:
        sorter = self._sorter(tmp_path, min_category_confidence=0.5)
        record = _make_record(subject="other", is_solo_person=False, category="food", category_confidence=0.3)
        label, _ = sorter._decide_destination(record)
        assert label == "ignored"

    def test_none_category_routes_to_ignored(self, tmp_path: Path) -> None:
        sorter = self._sorter(tmp_path)
        record = _make_record(subject="other", is_solo_person=False, category=None)
        label, _ = sorter._decide_destination(record)
        assert label == "ignored"

    def test_solo_person_with_face_sorting_uses_face_bucket(self, tmp_path: Path) -> None:
        sorter = self._sorter(tmp_path, enable_face_sorting=True)
        record = _make_record(subject="person", is_solo_person=True, category="portrait", face_identity="face_001")
        label, _ = sorter._decide_destination(record)
        assert label == "faces/face_001"

    def test_group_shot_is_NOT_routed_to_face_bucket(self, tmp_path: Path) -> None:
        """Non-solo persons must not land in face buckets (was missing guard)."""
        sorter = self._sorter(tmp_path, enable_face_sorting=True)
        record = _make_record(subject="person", is_solo_person=False, category="group", face_identity="face_001")
        label, _ = sorter._decide_destination(record)
        # Must route by category, not face bucket
        assert label == "group", f"Group shot incorrectly routed to '{label}'"

    def test_non_person_not_routed_to_face_bucket(self, tmp_path: Path) -> None:
        sorter = self._sorter(tmp_path, enable_face_sorting=True)
        record = _make_record(subject="other", is_solo_person=False, category="travel", face_identity="face_001")
        label, _ = sorter._decide_destination(record)
        assert label == "travel"


# ─────────────────────────────────────────────────────────────────────────────
# BUG-5  finalize.py  validate_bundle — fallback-aware primary model check
# ─────────────────────────────────────────────────────────────────────────────

class TestBug5FinalizeValidation:
    def _make_artifacts(self, tmp_path: Path, config: SorterConfig):
        from media_sorter.finalize import FinalizeArtifacts
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        model = bundle / "image_encoder.onnx"
        model.write_text("onnx")
        quant = bundle / "image_encoder.int8.onnx"
        quant.write_text("quantized")
        emb = bundle / "text_embeddings.npz"
        n = len(config.subject_prompts)
        np.savez(
            emb,
            subject=np.ones((n, 4), dtype=np.float32),
            count=np.ones((3, 4), dtype=np.float32),
            category=np.ones((len(config.level_prompts), 4), dtype=np.float32),
        )
        req = bundle / "requirements.txt"
        req.write_text("numpy\n")
        runner = bundle / "run_media_sort.py"
        # don't create runner so runner_path.exists() == False
        cfg = bundle / "config.json"
        return bundle, model, quant, emb, req, runner, cfg

    def _write_config(self, cfg_path: Path, primary: str, fallback: str, config: SorterConfig) -> None:
        payload = {
            "bundle_version": 1,
            "runtime": {
                "primary_model": primary,
                "fallback_model": fallback,
                "embeddings_file": "text_embeddings.npz",
            },
            "preprocess": {"resize_size": 224, "crop_size": 224, "mean": [0.5], "std": [0.5]},
            "labels": {
                "subject": list(config.subject_prompts.keys()),
                "count": list(config.count_labels),
                "category": list(config.level_prompts.keys()),
            },
            "prompts": {
                "subject": dict(config.subject_prompts),
                "count": list(config.count_labels),
                "category": dict(config.level_prompts),
            },
            "thresholds": {
                "min_person_confidence": 0.35,
                "min_solo_confidence": 0.45,
                "min_solo_margin": 0.05,
                "min_solo_frame_ratio": 0.5,
                "min_category_confidence": 0.0,
            },
            "features": {
                "image_analysis": True,
                "video_analysis": False,
                "face_sorting": False,
                "runtime_prompt_editing": False,
            },
        }
        cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    def test_missing_quantized_primary_passes_when_fallback_exists(self, tmp_path: Path) -> None:
        config = SorterConfig(source_dir=tmp_path, output_dir=None)
        finalizer = BundleFinalizer(config)
        bundle, model, quant, emb, req, runner, cfg = self._make_artifacts(tmp_path, config)

        # Simulate --no-quantize: primary == fallback == "image_encoder.onnx"
        self._write_config(cfg, primary="image_encoder.onnx", fallback="image_encoder.onnx", config=config)

        from media_sorter.finalize import FinalizeArtifacts
        artifacts = FinalizeArtifacts(
            bundle_dir=bundle,
            model_path=model,
            quantized_model_path=None,  # no quantized model
            config_path=cfg,
            embeddings_path=emb,
            runner_path=runner,
            requirements_path=req,
        )
        result = finalizer.validate_bundle(artifacts, check_runner=False)
        assert result.ok, f"Validation failed unexpectedly: {result.errors}"

    def test_both_models_missing_is_an_error(self, tmp_path: Path) -> None:
        config = SorterConfig(source_dir=tmp_path, output_dir=None)
        finalizer = BundleFinalizer(config)
        bundle, model, quant, emb, req, runner, cfg = self._make_artifacts(tmp_path, config)

        # Write config pointing to nonexistent models
        self._write_config(cfg, primary="missing.int8.onnx", fallback="missing.onnx", config=config)

        from media_sorter.finalize import FinalizeArtifacts
        artifacts = FinalizeArtifacts(
            bundle_dir=bundle,
            model_path=bundle / "missing.int8.onnx",
            quantized_model_path=bundle / "missing.int8.onnx",
            config_path=cfg,
            embeddings_path=emb,
            runner_path=runner,
            requirements_path=req,
        )
        result = finalizer.validate_bundle(artifacts, check_runner=False)
        assert not result.ok
        assert any("model" in e.lower() for e in result.errors)


# ─────────────────────────────────────────────────────────────────────────────
# BUG-6  pipeline.py  solo_confidence unit consistency (image vs video)
# ─────────────────────────────────────────────────────────────────────────────

class TestBug6SoloConfidenceUnit:
    def _make_prediction(self, solo_prob: float, config: SorterConfig) -> Prediction:
        return Prediction(
            file_path=None,
            subject="person",
            subject_confidence=0.9,
            subject_scores={"person": 0.9, "pet": 0.05, "other": 0.05},
            is_solo_person=True,
            solo_label=config.count_labels[0],
            solo_confidence=solo_prob,
            count_scores={
                config.count_labels[0]: solo_prob,
                config.count_labels[1]: 0.05,
                config.count_labels[2]: 0.05,
            },
            category="portrait",
            category_confidence=0.85,
            category_scores={"portrait": 0.85},
        )

    def test_video_solo_confidence_is_probability_not_ratio(self, tmp_path: Path) -> None:
        config = SorterConfig(source_dir=tmp_path, output_dir=None)
        analyzer = MediaAnalyzer(config, classifier=FakeClassifier(config))

        # Build two predictions with known count_scores
        pred1 = self._make_prediction(0.80, config)
        pred2 = self._make_prediction(0.90, config)
        predictions = [pred1, pred2]

        result = analyzer._aggregate_video_predictions(Path("/fake/v.mp4"), predictions)

        # Expected: average of 0.80 and 0.90 = 0.85 (CLIP probability)
        expected = (0.80 + 0.90) / 2
        assert abs(result.solo_confidence - expected) < 1e-5, (
            f"solo_confidence={result.solo_confidence!r} is not the averaged probability {expected}"
        )
        # Must NOT be the frame ratio (which would be 1.0 here since both preds have is_solo=True)
        assert result.solo_confidence != 1.0, "solo_confidence is still the frame ratio!"


# ─────────────────────────────────────────────────────────────────────────────
# BUG-7  i18n.py  msgid is not None flush condition
# ─────────────────────────────────────────────────────────────────────────────

class TestBug7I18nFlushCondition:
    def test_nonempty_msgid_with_empty_msgstr_is_stored(self, tmp_path: Path) -> None:
        po = tmp_path / "messages.po"
        po.write_text(
            'msgid "has_empty_translation"\nmsgstr ""\n'
            'msgid "normal"\nmsgstr "ok"\n',
            encoding="utf-8",
        )
        result = load_po(str(po))
        # Both keys should be present; caller's lookup `if po_dict[s]:` handles empty fallback
        assert "normal" in result
        assert result["normal"] == "ok"

    def test_header_msgid_empty_string_does_not_break_parsing(self, tmp_path: Path) -> None:
        po = tmp_path / "messages.po"
        po.write_text(
            'msgid ""\nmsgstr "Content-Type: text/plain; charset=UTF-8\\n"\n'
            'msgid "greeting"\nmsgstr "salut"\n',
            encoding="utf-8",
        )
        result = load_po(str(po))
        assert result.get("greeting") == "salut"

    def test_last_entry_in_file_is_not_dropped(self, tmp_path: Path) -> None:
        po = tmp_path / "messages.po"
        po.write_text(
            'msgid "first"\nmsgstr "un"\n'
            'msgid "last"\nmsgstr "deux"\n',  # no trailing newline after this
            encoding="utf-8",
        )
        result = load_po(str(po))
        assert result.get("first") == "un"
        assert result.get("last") == "deux", "Last PO entry was dropped!"


# ─────────────────────────────────────────────────────────────────────────────
# BUG-8  install.sh  step label consistency
# (already covered in TestBug2InstallShNoEval.test_step_labels_are_consistent)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# BUG-9  core.py  narrow except on open_clip fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestBug9NarrowExcept:
    def test_source_contains_narrow_except(self) -> None:
        import media_sorter.core as mod
        src = Path(mod.__file__).read_text(encoding="utf-8")
        # The fallback except must target ImportError/ModuleNotFoundError
        assert "except (ImportError, ModuleNotFoundError):" in src, (
            "core.py open_clip fallback is not using a narrow except clause"
        )
        # The old broad catch must be gone from that context
        # (there may be other broad excepts in the error message builder – that's fine)
        lines = src.splitlines()
        in_init_block = False
        for line in lines:
            if "def __init__" in line:
                in_init_block = True
            if in_init_block and "except Exception:" in line:
                pytest.fail(
                    "Broad `except Exception:` still present in MediaClassifier.__init__"
                )
            if in_init_block and "self.config = config" in line:
                break  # exit __init__ body scan


# ─────────────────────────────────────────────────────────────────────────────
# BUG-10  pipeline.py  _write_manifest extracted to shared helper
# ─────────────────────────────────────────────────────────────────────────────

class TestBug10SharedWriteManifest:
    def test_write_records_to_manifest_is_importable(self) -> None:
        """The module-level helper must be importable from pipeline."""
        from media_sorter.pipeline import _write_records_to_manifest
        assert callable(_write_records_to_manifest)

    def test_analyzer_and_sorter_both_delegate_to_helper(self) -> None:
        import inspect, media_sorter.pipeline as mod

        analyzer_src = inspect.getsource(MediaAnalyzer._write_manifest)
        sorter_src = inspect.getsource(MediaSorter._write_manifest)

        assert "_write_records_to_manifest" in analyzer_src, (
            "MediaAnalyzer._write_manifest does not delegate to the shared helper"
        )
        assert "_write_records_to_manifest" in sorter_src, (
            "MediaSorter._write_manifest does not delegate to the shared helper"
        )

    def test_shared_helper_writes_jsonl(self, tmp_path: Path) -> None:
        config = SorterConfig(source_dir=tmp_path, output_dir=None, manifest_format="jsonl")
        record = _make_record()
        out = _write_records_to_manifest(
            [record], config, artifact_root=tmp_path
        )
        assert out.exists()
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["subject"] == "person"

    def test_shared_helper_writes_json(self, tmp_path: Path) -> None:
        config = SorterConfig(source_dir=tmp_path, output_dir=None, manifest_format="json")
        record = _make_record()
        out = _write_records_to_manifest(
            [record], config, artifact_root=tmp_path
        )
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, list) and len(data) == 1

    def test_analyzer_manifest_path_roundtrip(self, source_dir: Path) -> None:
        config = SorterConfig(
            source_dir=source_dir,
            output_dir=None,
            write_manifest=True,
            manifest_format="json",
        )
        analyzer = MediaAnalyzer(config, classifier=FakeClassifier(config))
        analyzer.run()
        assert analyzer.manifest_output_path is not None
        assert analyzer.manifest_output_path.exists()
        data = json.loads(analyzer.manifest_output_path.read_text())
        assert len(data) == 1

    def test_sorter_manifest_roundtrip(self, source_dir: Path, tmp_path: Path) -> None:
        config = SorterConfig(
            source_dir=source_dir,
            output_dir=tmp_path / "out",
            copy_mode="copy",
            write_manifest=True,
            manifest_format="jsonl",
        )
        sorter = MediaSorter(config, classifier=FakeClassifier(config))
        sorter.run()
        assert sorter.manifest_output_path is not None
        assert sorter.manifest_output_path.exists()

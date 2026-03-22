from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline import MediaRecord


class IndexStore:
    def __init__(self, db_path: Path, mode: str = "full") -> None:
        self.db_path = Path(db_path)
        self.mode = mode
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        self.conn.close()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS media_index (
                source TEXT PRIMARY KEY,
                media_type TEXT NOT NULL,
                destination TEXT NOT NULL DEFAULT '',
                route_label TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                subject TEXT NOT NULL,
                subject_confidence REAL NOT NULL,
                subject_scores_json TEXT NOT NULL DEFAULT '{}',
                is_solo_person INTEGER NOT NULL,
                solo_label TEXT NOT NULL DEFAULT '',
                solo_confidence REAL NOT NULL,
                count_scores_json TEXT NOT NULL DEFAULT '{}',
                category TEXT,
                category_confidence REAL NOT NULL,
                category_scores_json TEXT NOT NULL DEFAULT '{}',
                face_identity TEXT,
                sampled_frames TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                file_size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_media_index_status ON media_index(status)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_media_index_route_label ON media_index(route_label)")
        self._ensure_missing_columns(
            {
                "route_label": "TEXT",
                "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
                "subject_scores_json": "TEXT NOT NULL DEFAULT '{}'",
                "solo_label": "TEXT NOT NULL DEFAULT ''",
                "count_scores_json": "TEXT NOT NULL DEFAULT '{}'",
                "category_scores_json": "TEXT NOT NULL DEFAULT '{}'",
            }
        )
        self.conn.commit()

    def _ensure_missing_columns(self, column_defs: dict[str, str]) -> None:
        existing = {
            str(row["name"])
            for row in self.conn.execute("PRAGMA table_info(media_index)").fetchall()
        }
        for column_name, ddl in column_defs.items():
            if column_name in existing:
                continue
            self.conn.execute(f"ALTER TABLE media_index ADD COLUMN {column_name} {ddl}")

    def should_process(self, path: Path) -> tuple[bool, sqlite3.Row | None]:
        if self.mode != "update":
            return True, None

        row = self.conn.execute(
            "SELECT * FROM media_index WHERE source = ?",
            (str(path),),
        ).fetchone()
        if row is None:
            return True, None

        stat = path.stat()
        if int(row["file_size"]) != int(stat.st_size) or int(row["mtime_ns"]) != int(stat.st_mtime_ns):
            return True, row

        return False, row

    def upsert_record(self, record: "MediaRecord", file_size: int, mtime_ns: int) -> None:
        payload = asdict(record)
        self.conn.execute(
            """
            INSERT INTO media_index (
                source, media_type, destination, route_label, metadata_json, subject, subject_confidence,
                subject_scores_json, is_solo_person, solo_label, solo_confidence, count_scores_json,
                category, category_confidence, category_scores_json, face_identity, sampled_frames,
                status, error, file_size, mtime_ns, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
                media_type = excluded.media_type,
                destination = excluded.destination,
                route_label = excluded.route_label,
                metadata_json = excluded.metadata_json,
                subject = excluded.subject,
                subject_confidence = excluded.subject_confidence,
                subject_scores_json = excluded.subject_scores_json,
                is_solo_person = excluded.is_solo_person,
                solo_label = excluded.solo_label,
                solo_confidence = excluded.solo_confidence,
                count_scores_json = excluded.count_scores_json,
                category = excluded.category,
                category_confidence = excluded.category_confidence,
                category_scores_json = excluded.category_scores_json,
                face_identity = excluded.face_identity,
                sampled_frames = excluded.sampled_frames,
                status = excluded.status,
                error = excluded.error,
                file_size = excluded.file_size,
                mtime_ns = excluded.mtime_ns,
                updated_at = excluded.updated_at
            """,
            (
                payload["source"],
                payload["media_type"],
                payload["destination"] or "",
                payload["route_label"],
                json.dumps(payload["metadata"], ensure_ascii=True),
                payload["subject"],
                float(payload["subject_confidence"]),
                json.dumps(payload["subject_scores"], ensure_ascii=True),
                1 if payload["is_solo_person"] else 0,
                payload["solo_label"],
                float(payload["solo_confidence"]),
                json.dumps(payload["count_scores"], ensure_ascii=True),
                payload["category"],
                float(payload["category_confidence"]),
                json.dumps(payload["category_scores"], ensure_ascii=True),
                payload["face_identity"],
                json.dumps(payload["sampled_frames"], ensure_ascii=True),
                payload["status"],
                payload["error"],
                int(file_size),
                int(mtime_ns),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def commit(self) -> None:
        self.conn.commit()

    def prune_missing(self, source_paths: set[str]) -> int:
        if not source_paths:
            deleted = self.conn.execute("DELETE FROM media_index").rowcount
            self.conn.commit()
            return int(deleted)

        placeholders = ",".join(["?"] * len(source_paths))
        deleted = self.conn.execute(
            f"DELETE FROM media_index WHERE source NOT IN ({placeholders})",
            tuple(sorted(source_paths)),
        ).rowcount
        self.conn.commit()
        return int(deleted)


def record_from_row(row: sqlite3.Row, status: str = "skipped") -> dict[str, Any]:
    from .pipeline import MediaMetadata

    metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    subject_scores = json.loads(row["subject_scores_json"]) if row["subject_scores_json"] else {}
    count_scores = json.loads(row["count_scores_json"]) if row["count_scores_json"] else {}
    category_scores = json.loads(row["category_scores_json"]) if row["category_scores_json"] else {}
    sampled = json.loads(row["sampled_frames"]) if row["sampled_frames"] else []

    return {
        "source": row["source"],
        "media_type": row["media_type"],
        "destination": row["destination"] or None,
        "route_label": row["route_label"],
        "metadata": MediaMetadata(**metadata),
        "subject": row["subject"],
        "subject_confidence": float(row["subject_confidence"]),
        "subject_scores": {str(k): float(v) for k, v in subject_scores.items()},
        "is_solo_person": bool(row["is_solo_person"]),
        "solo_label": row["solo_label"],
        "solo_confidence": float(row["solo_confidence"]),
        "count_scores": {str(k): float(v) for k, v in count_scores.items()},
        "category": row["category"],
        "category_confidence": float(row["category_confidence"]),
        "category_scores": {str(k): float(v) for k, v in category_scores.items()},
        "face_identity": row["face_identity"],
        "sampled_frames": [int(x) for x in sampled],
        "status": status,
        "error": row["error"],
    }

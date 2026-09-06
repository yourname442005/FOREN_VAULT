import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "forenvault.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _get_existing_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    cursor = connection.execute(
        f"PRAGMA table_info({table_name})"
    )
    return {row[1] for row in cursor.fetchall()}


def initialize_database():
    connection = get_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evidence_id TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            file_type TEXT NOT NULL DEFAULT 'unknown',
            vendor TEXT NOT NULL DEFAULT 'unknown',
            vendor_confidence REAL NOT NULL DEFAULT 0.0,
            detection_method TEXT NOT NULL DEFAULT 'none'
        )
        """
    )

    existing = _get_existing_columns(
        connection, "evidence"
    )

    alter_columns = [
        (
            "file_type",
            "TEXT NOT NULL DEFAULT 'unknown'",
        ),
        (
            "vendor",
            "TEXT NOT NULL DEFAULT 'unknown'",
        ),
        (
            "vendor_confidence",
            "REAL NOT NULL DEFAULT 0.0",
        ),
        (
            "detection_method",
            "TEXT NOT NULL DEFAULT 'none'",
        ),
    ]

    for column_name, column_def in alter_columns:
        if column_name not in existing:
            connection.execute(
                f"ALTER TABLE evidence "
                f"ADD COLUMN {column_name} {column_def}"
            )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chain_of_custody (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE NOT NULL,
            evidence_id TEXT NOT NULL,
            action TEXT NOT NULL,
            description TEXT NOT NULL,
            actor TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            details TEXT,
            FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id)
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chain_of_custody_evidence
        ON chain_of_custody(evidence_id)
        """
    )

    connection.commit()
    connection.close()
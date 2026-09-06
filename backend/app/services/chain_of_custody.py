import json
from datetime import datetime, timezone
from uuid import uuid4

from app.core.database import get_connection


def record_custody_event(
    evidence_id: str,
    action: str,
    description: str,
    actor: str = "FORENVAULT",
    details: dict | None = None,
) -> dict:
    event_id = f"COC-{uuid4().hex[:12].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()

    details_json = json.dumps(details) if details is not None else None

    connection = get_connection()

    connection.execute(
        """
        INSERT INTO chain_of_custody (
            event_id,
            evidence_id,
            action,
            description,
            actor,
            timestamp,
            details
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            evidence_id,
            action,
            description,
            actor,
            timestamp,
            details_json,
        ),
    )

    connection.commit()
    connection.close()

    return {
        "event_id": event_id,
        "evidence_id": evidence_id,
        "action": action,
        "description": description,
        "actor": actor,
        "timestamp": timestamp,
        "details": details,
    }


def get_custody_history(evidence_id: str) -> list[dict]:
    connection = get_connection()

    rows = connection.execute(
        """
        SELECT
            event_id,
            evidence_id,
            action,
            description,
            actor,
            timestamp,
            details
        FROM chain_of_custody
        WHERE evidence_id = ?
        ORDER BY timestamp ASC
        """,
        (evidence_id,),
    ).fetchall()

    connection.close()

    history = []

    for row in rows:
        details = None

        if row["details"]:
            try:
                details = json.loads(row["details"])
            except json.JSONDecodeError:
                details = row["details"]

        history.append(
            {
                "event_id": row["event_id"],
                "evidence_id": row["evidence_id"],
                "action": row["action"],
                "description": row["description"],
                "actor": row["actor"],
                "timestamp": row["timestamp"],
                "details": details,
            }
        )

    return history
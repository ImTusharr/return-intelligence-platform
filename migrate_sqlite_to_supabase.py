
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(override=True)

DATABASE_URL = os.getenv("DATABASE_URL")
SQLITE_PATH = Path("instance/return_insight.db")

TABLES = [
    "users",
    "return_requests",
    "return_images",
    "return_analyses",
]

# Timestamp columns that need conversion from SQLite strings.
DATETIME_COLUMNS = {
    "return_requests": ["created_at"],
    "return_analyses": ["reviewed_at", "created_at", "updated_at"],
}


def get_source_counts(connection):
    counts = {}

    for table in TABLES:
        counts[table] = connection.execute(
            f'SELECT COUNT(*) FROM "{table}"'
        ).fetchone()[0]

    return counts


def convert_datetime_values(table, row):
    values = dict(row)

    for column in DATETIME_COLUMNS.get(table, []):
        value = values.get(column)

        if value is None or isinstance(value, datetime):
            continue

        parsed = datetime.fromisoformat(value)

        # Preserve UTC meaning for timezone-aware database columns.
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        values[column] = parsed

    return values


def check_source_schema(connection):
    existing_tables = {
        row[0]
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
    }

    missing = set(TABLES) - existing_tables

    if missing:
        raise RuntimeError(
            f"Missing SQLite tables: {', '.join(sorted(missing))}"
        )


def main():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is missing from .env")

    if not DATABASE_URL.startswith(("postgresql://", "postgresql+")):
        raise RuntimeError(
            "DATABASE_URL must point to PostgreSQL."
        )

    if not SQLITE_PATH.is_file():
        raise FileNotFoundError(
            f"SQLite database not found: {SQLITE_PATH.resolve()}"
        )

    sqlite_conn = sqlite3.connect(str(SQLITE_PATH))
    sqlite_conn.row_factory = sqlite3.Row
    engine = None

    try:
        integrity = sqlite_conn.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        if integrity != "ok":
            raise RuntimeError(
                f"SQLite integrity check failed: {integrity}"
            )

        check_source_schema(sqlite_conn)
        source_counts = get_source_counts(sqlite_conn)

        print("SQLite integrity:", integrity)
        print("Source counts:", source_counts)

        # Import the application's actual SQLAlchemy models.
        from app import app
        from extensions import db

        engine = create_engine(
            DATABASE_URL,
            connect_args={"connect_timeout": 10},
        )

        # Create only missing tables; never drop existing tables.
        with app.app_context():
            db.create_all()

            for table in TABLES:
                if table not in db.metadata.tables:
                    raise RuntimeError(
                        f"SQLAlchemy model table not found: {table}"
                    )

        # The records are copied in one PostgreSQL transaction.
        with engine.begin() as target:
            destination_counts = {}

            for table in TABLES:
                destination_counts[table] = target.execute(
                    text(f'SELECT COUNT(*) FROM "{table}"')
                ).scalar_one()

            print("Existing Supabase counts:", destination_counts)

            if any(destination_counts.values()):
                raise RuntimeError(
                    "Migration stopped because destination tables "
                    "already contain records. No rows were copied. "
                    "Inspect the destination before trying again."
                )

            # Check source columns against the application's models.
            with app.app_context():
                for table in TABLES:
                    source_columns = {
                        row["name"]
                        for row in sqlite_conn.execute(
                            f'PRAGMA table_info("{table}")'
                        ).fetchall()
                    }

                    model_columns = set(
                        db.metadata.tables[table].columns.keys()
                    )

                    if source_columns != model_columns:
                        raise RuntimeError(
                            f"Schema mismatch for {table}. "
                            f"SQLite columns: {sorted(source_columns)}; "
                            f"model columns: {sorted(model_columns)}. "
                            "Review the schema before migrating."
                        )

                    # SQLAlchemy table objects retain column types and
                    # correctly bind Python datetime values.
                    model_table = db.metadata.tables[table]

                    rows = sqlite_conn.execute(
                        f'SELECT * FROM "{table}" ORDER BY id'
                    ).fetchall()

                    if not rows:
                        print(f"{table}: 0 records to copy")
                        continue

                    converted_rows = [
                        convert_datetime_values(table, row)
                        for row in rows
                    ]

                    target.execute(
                        model_table.insert(),
                        converted_rows,
                    )

                    print(
                        f"{table}: copied {len(converted_rows)} records"
                    )

            # Reset serial/identity sequences to the largest imported ID.
            for table in TABLES:
                max_id = target.execute(
                    text(f'SELECT MAX(id) FROM "{table}"')
                ).scalar_one()

                sequence_name = target.execute(
                    text(
                        """
                        SELECT pg_get_serial_sequence(:table_name, 'id')
                        """
                    ),
                    {"table_name": f"public.{table}"},
                ).scalar_one()

                if sequence_name and max_id is not None:
                    target.execute(
                        text(
                            """
                            SELECT setval(
                                CAST(:sequence_name AS regclass),
                                :max_id,
                                true
                            )
                            """
                        ),
                        {
                            "sequence_name": sequence_name,
                            "max_id": max_id,
                        },
                    )

            # Verify final row counts before committing.
            final_counts = {}

            for table in TABLES:
                final_counts[table] = target.execute(
                    text(f'SELECT COUNT(*) FROM "{table}"')
                ).scalar_one()

            print("Final Supabase counts:", final_counts)

            if source_counts != final_counts:
                raise RuntimeError(
                    "Source and destination counts differ. "
                    "The transaction will be rolled back."
                )

            # Check important foreign-key relationships.
            invalid_user_refs = target.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM return_requests r
                    LEFT JOIN users u ON u.id = r.user_id
                    WHERE u.id IS NULL
                    """
                )
            ).scalar_one()

            invalid_image_refs = target.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM return_images i
                    LEFT JOIN return_requests r
                      ON r.id = i.return_request_id
                    WHERE r.id IS NULL
                    """
                )
            ).scalar_one()

            invalid_analysis_refs = target.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM return_analyses a
                    LEFT JOIN return_requests r
                      ON r.id = a.return_request_id
                    WHERE r.id IS NULL
                    """
                )
            ).scalar_one()

            invalid_reviewer_refs = target.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM return_analyses a
                    LEFT JOIN users u ON u.id = a.reviewer_id
                    WHERE a.reviewer_id IS NOT NULL
                      AND u.id IS NULL
                    """
                )
            ).scalar_one()

            invalid_refs = {
                "return_requests.user_id": invalid_user_refs,
                "return_images.return_request_id": invalid_image_refs,
                "return_analyses.return_request_id": invalid_analysis_refs,
                "return_analyses.reviewer_id": invalid_reviewer_refs,
            }

            print("Foreign-key validation:", invalid_refs)

            if any(invalid_refs.values()):
                raise RuntimeError(
                    "Foreign-key validation failed. "
                    "The transaction will be rolled back."
                )

        print("\nMigration completed successfully!")
        print("All four tables match the SQLite source.")
        print("Foreign-key validation passed.")
        print("Original SQLite database remains unchanged.")

    finally:
        if engine is not None:
            engine.dispose()

        sqlite_conn.close()


if __name__ == "__main__":
    main()

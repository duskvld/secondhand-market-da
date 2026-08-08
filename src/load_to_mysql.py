"""검증을 통과한 7개 CSV를 기존 MySQL 테이블이 모두 비어 있을 때만 적재한다."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


LOAD_ORDER = [
    "users",
    "listings",
    "product_views",
    "favorites",
    "chatrooms",
    "transactions",
    "price_history",
]

REQUIRED_ENVIRONMENT = [
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "MYSQL_DATABASE",
]


def _connection_settings(project_root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    load_dotenv(project_root / ".env", override=False)
    missing = [name for name in REQUIRED_ENVIRONMENT if os.getenv(name) is None]
    if missing:
        return None, missing
    try:
        port = int(os.environ["MYSQL_PORT"])
    except ValueError as exc:
        raise ValueError("MYSQL_PORT는 정수여야 합니다.") from exc
    return {
        "host": os.environ["MYSQL_HOST"],
        "port": port,
        "user": os.environ["MYSQL_USER"],
        "password": os.environ["MYSQL_PASSWORD"],
        "database": os.environ["MYSQL_DATABASE"],
        "charset": "utf8mb4",
    }, []


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def load_to_mysql(
    project_root: Path,
    load_enabled: bool,
    passed_integrity: bool,
) -> dict[str, Any]:
    """안전 조건을 확인한 뒤 한 트랜잭션으로 적재하고 상태를 반환한다."""

    if not load_enabled:
        return {"success": False, "status": "skipped_disabled", "message": "config에서 MySQL 적재가 비활성화됨"}
    if not passed_integrity:
        return {"success": False, "status": "skipped_integrity_errors", "message": "치명적 무결성 오류로 적재 중단"}
    settings, missing = _connection_settings(project_root)
    if missing:
        return {
            "success": False,
            "status": "skipped_missing_environment",
            "message": "MySQL 환경변수가 없어 CSV 생성과 검증까지만 완료",
            "missing_environment_variables": missing,
        }
    assert settings is not None

    try:
        import mysql.connector
    except ImportError as exc:
        return {
            "success": False,
            "status": "skipped_missing_package",
            "message": f"mysql-connector-python을 불러오지 못함: {exc}",
        }

    expected_database = settings["database"]
    connection = None
    try:
        connection = mysql.connector.connect(**settings)
        cursor = connection.cursor()
        cursor.execute("SELECT DATABASE()")
        selected_database = cursor.fetchone()[0]
        if selected_database != expected_database:
            return {
                "success": False,
                "status": "aborted_database_mismatch",
                "message": "연결된 데이터베이스가 MYSQL_DATABASE와 다름",
            }

        placeholders = ", ".join(["%s"] * len(LOAD_ORDER))
        cursor.execute(
            f"SELECT table_name FROM information_schema.tables "
            f"WHERE table_schema = %s AND table_name IN ({placeholders})",
            [expected_database, *LOAD_ORDER],
        )
        existing_tables = {row[0] for row in cursor.fetchall()}
        missing_tables = sorted(set(LOAD_ORDER) - existing_tables)
        if missing_tables:
            return {
                "success": False,
                "status": "aborted_missing_tables",
                "message": "carrot_chat에 필요한 7개 테이블이 모두 존재하지 않음",
                "missing_tables": missing_tables,
            }

        existing_counts: dict[str, int] = {}
        for table in LOAD_ORDER:
            cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
            existing_counts[table] = int(cursor.fetchone()[0])
        nonempty = {table: count for table, count in existing_counts.items() if count > 0}
        if nonempty:
            return {
                "success": False,
                "status": "aborted_nonempty_tables",
                "message": "기존 데이터를 보호하기 위해 적재하지 않음",
                "existing_row_counts": existing_counts,
            }

        inserted_counts: dict[str, int] = {}
        for table in LOAD_ORDER:
            frame = pd.read_csv(project_root / "generated/csv" / f"{table}.csv")
            columns = list(frame.columns)
            column_sql = ", ".join(f"`{column}`" for column in columns)
            value_sql = ", ".join(["%s"] * len(columns))
            statement = f"INSERT INTO `{table}` ({column_sql}) VALUES ({value_sql})"
            records = [tuple(_clean_value(value) for value in row) for row in frame.itertuples(index=False, name=None)]
            batch_size = 1000
            for offset in range(0, len(records), batch_size):
                cursor.executemany(statement, records[offset : offset + batch_size])
            inserted_counts[table] = len(records)
        connection.commit()
        return {
            "success": True,
            "status": "loaded",
            "message": "7개 테이블 적재 완료",
            "inserted_row_counts": inserted_counts,
        }
    except Exception as exc:
        if connection is not None:
            connection.rollback()
        return {
            "success": False,
            "status": "failed",
            "message": f"MySQL 적재 실패: {type(exc).__name__}: {exc}",
        }
    finally:
        if connection is not None and connection.is_connected():
            connection.close()

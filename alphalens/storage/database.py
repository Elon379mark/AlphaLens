from pathlib import Path
import logging
import warnings
from typing import Any, Dict, List, Optional
from database.session import DatabaseSessionManager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Deprecated: Legacy compatibility wrapper for DatabaseManager.
    Reroutes connection requests to the new SQLAlchemy-based DatabaseSessionManager.
    """
    def __init__(self, dsn: Optional[str] = None, sqlite_path: Optional[str] = None):
        warnings.warn(
            "alphalens.storage.database.DatabaseManager is deprecated. "
            "Please use the new database package imports from `database/` instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if sqlite_path is None:
            sqlite_path = str(Path(__file__).resolve().parents[2] / "alphalens_local.db")
        self.db_manager = DatabaseSessionManager(dsn=dsn, sqlite_path=sqlite_path)
        self.is_sqlite = self.db_manager.is_sqlite
        self._raw_conn = self.db_manager.sync_engine.raw_connection()
        # For SQLite, register the row factory on the underlying driver connection
        if self.is_sqlite:
            import sqlite3
            self._raw_conn.connection.row_factory = sqlite3.Row

    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        warnings.warn(
            "DatabaseManager.execute_query is deprecated. Please use the new database package/SQLAlchemy.",
            DeprecationWarning,
            stacklevel=2
        )
        cursor = self._raw_conn.cursor()
        try:
            cursor.execute(query, params or ())
            self._raw_conn.commit()
            if query.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                if self.is_sqlite:
                    return [dict(row) for row in rows]
                else:
                    # Convert psycopg2 or DB-API rows to list of dicts
                    desc = cursor.description
                    if desc:
                        colnames = [col[0] for col in desc]
                        return [dict(zip(colnames, row)) for row in rows]
                    return []
            return []
        except Exception as e:
            self._raw_conn.rollback()
            logger.error(f"Legacy Database query error: {e}\nQuery: {query}")
            raise
        finally:
            cursor.close()

    def setup_tables(self):
        """Initializes tables using the new declarative mapping."""
        self.db_manager.create_tables()

    def close(self):
        try:
            self._raw_conn.close()
        except Exception:
            pass


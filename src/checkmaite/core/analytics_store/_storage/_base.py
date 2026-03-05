from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import polars as pl

from checkmaite.core.analytics_store._schema import BaseRecord


class StorageBackend(Protocol):
    """Protocol for analytics store storage backends.

    Implement this protocol to add new storage backends (e.g., Delta Lake,
    Lance, SQL databases). The default implementation is ParquetBackend.

    Storage is organized by table name:
    - ``runs`` — automatically populated run metadata
    - ``dataeval_cleaning``, ``maite_evaluation``, etc. — capability data

    The primary query interface is SQL-based via query_sql(), which provides
    maximum flexibility for querying across tables.
    """

    def write(self, records: Sequence[BaseRecord]) -> None:
        """Write records to storage.

        Records are batched by table name for efficiency.

        Parameters
        ----------
        records
            List of records to write. Each record is a ``BaseRecord``
            subclass with ``table_name`` and ``model_dump()``.
        """
        ...

    def list_tables(self) -> list[str]:
        """List available tables in the store.

        Returns
        -------
        list[str]
            List of table names.
        """
        ...

    def describe_table(self, table_name: str) -> dict[str, str]:
        """Get schema information for a table.

        Parameters
        ----------
        table_name
            Name of the table to describe.

        Returns
        -------
        dict[str, str]
            Mapping of column names to their type strings.
        """
        ...

    def query_sql(self, sql: str) -> pl.DataFrame:
        """Execute a SQL query against the store.

        All record fields are scalar (no nested types), so standard SQL
        filtering, aggregation and JOINs work directly.

        Available tables are named after capability types (e.g., "dataeval_cleaning",
        "maite_evaluation"). Use these names in your SQL queries.

        Parameters
        ----------
        sql
            SQL query string.

        Returns
        -------
        pl.DataFrame
            Query results.
        """
        ...

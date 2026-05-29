from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import polars as pl

from checkmaite.core.analytics_store._schema import BaseRecord


@dataclass(frozen=True)
class StorageWriteReceipt:
    """Metadata emitted by a single storage ``write()`` call.

    Attributes
    ----------
    run_table_files
        Mapping of ``run_uid`` to payload-table file/object URIs written by the
        call. The auto-generated ``runs`` table is intentionally excluded.
    """

    run_table_files: dict[str, dict[str, str]] = field(default_factory=dict)

    def resolve_run_uri(self, run_uid: str) -> str | None:
        """Resolve the payload storage URI for ``run_uid`` from this receipt.

        Returns
        -------
        str | None
            The payload file/object URI written for this run when present in
            this receipt, else ``None``.
        """
        table_to_file = self.run_table_files.get(run_uid)
        if not table_to_file:
            return None

        if len(table_to_file) > 1:
            raise ValueError(
                f"Ambiguous run URI resolution for run_uid {run_uid!r}: "
                f"multiple payload tables present {sorted(table_to_file)!r}"
            )

        return next(iter(table_to_file.values()))


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

        Backend implementations are expected to preserve analytics-store
        idempotency semantics across repeated writes:
        - capability payload tables deduplicate by ``run_uid`` across writes,
        - the ``runs`` table deduplicates rows without ``run_event_id`` by
          mapping key ``(run_uid, capability_table, entity_type, entity_id)``,
          and rows with ``run_event_id`` by provenance-aware mapping key
          ``(run_uid, capability_table, entity_type, entity_id, run_event_id)``.

        Parameters
        ----------
        records
            List of records to write. Each record is a ``BaseRecord``
            subclass with ``table_name`` and ``model_dump()``.
        """
        ...

    def write_with_receipt(self, records: Sequence[BaseRecord]) -> StorageWriteReceipt:
        """Write records to storage and return concrete write metadata.

        This is the receipt-aware variant used by callers that need exact file
        or object URIs for the data written by this call.
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

    def get_run_uri(self, run_uid: str) -> str:
        """Return a concrete payload-data URI for ``run_uid``."""
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

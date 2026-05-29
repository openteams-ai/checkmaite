import time
import uuid
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import polars as pl
from upath import UPath

from checkmaite.core.analytics_store._schema import BaseRecord
from checkmaite.core.analytics_store._storage._base import StorageWriteReceipt

_RUNS_BASE_KEY = ["run_uid", "capability_table", "entity_type", "entity_id"]
_RUNS_EVENT_KEY = [*_RUNS_BASE_KEY, "run_event_id"]


class ParquetBackend:
    """Parquet-based storage backend for the analytics store.

    Uses Polars for reading, writing, and querying Parquet files.
    Data is written as plain Parquet files in a flat directory per table::

        {base_path}/
            dataeval_cleaning/
                {timestamp}.parquet
            maite_evaluation/
                {timestamp}.parquet
            runs/
                {timestamp}.parquet

    This layout is intentionally simple — plain Parquet in flat
    directories — so that the files are directly readable by any tool
    or future backend (Delta Lake, Iceberg, DuckDB, Snowflake, etc.)
    without requiring awareness of a custom directory convention.
    Date-based filtering uses the ``created_at`` column in SQL rather
    than physical partitioning.

    Parameters
    ----------
    uri
        Base directory or object-store prefix for storing parquet files.
    storage_options
        Optional Polars storage options passed through to parquet I/O.
    """

    def __init__(self, uri: str, storage_options: dict[str, Any] | None = None) -> None:
        self._path = UPath(uri)
        self._storage_options = dict(storage_options or {})

    @property
    def path(self) -> UPath:
        """The base directory or object-store prefix where parquet files are stored."""
        return self._path

    def _table_path(self, table_name: str) -> UPath:
        return self._path / table_name

    @staticmethod
    def _group_records(records: Sequence[BaseRecord]) -> dict[str, list[dict[str, Any]]]:
        batches: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            # mode="python" preserves native Python types (e.g. datetime) so that
            # Polars maps them to the correct dtypes (e.g. Datetime instead of Utf8).
            # mode="json" would serialize datetimes to ISO strings, losing type
            # fidelity in the Parquet file and breaking temporal SQL comparisons.
            batches[record.table_name].append(record.model_dump(mode="python"))
        return batches

    @staticmethod
    def _validate_schema_compatibility(
        table_name: str,
        batch: pl.DataFrame,
        existing_schema: dict[str, pl.DataType],
    ) -> None:
        for col_name, new_dtype in batch.schema.items():
            if (
                col_name in existing_schema
                and existing_schema[col_name] != new_dtype
                and existing_schema[col_name] != pl.Null
                and new_dtype != pl.Null
            ):
                raise TypeError(
                    f"Schema mismatch for table '{table_name}', column '{col_name}': "
                    f"existing type is {existing_schema[col_name]}, "
                    f"new type is {new_dtype}. "
                    f"Changing column types is not supported."
                )

    @staticmethod
    def _dedupe_runs_within_batch(batch: pl.DataFrame) -> pl.DataFrame:
        # If there is no run-event column, or it is all null, all rows use the base key.
        if "run_event_id" not in batch.columns or batch.get_column("run_event_id").null_count() == batch.height:
            # Keep one row per run/entity mapping, keeping the first one seen.
            return batch.unique(subset=_RUNS_BASE_KEY, maintain_order=True)

        # Add row numbers so we can restore the original order after splitting rows.
        indexed = batch.with_row_index("__row_nr")
        # Store the deduped row groups that we will join back together.
        parts: list[pl.DataFrame] = []

        # Rows without run_event_id use the base key.
        without_event = indexed.filter(pl.col("run_event_id").is_null()).unique(
            subset=_RUNS_BASE_KEY,
            maintain_order=True,
        )
        # Add rows that remain after base-key dedupe.
        if not without_event.is_empty():
            # Save them for the final output.
            parts.append(without_event)

        # Rows with run_event_id use the event key.
        with_event = indexed.filter(pl.col("run_event_id").is_not_null()).unique(
            subset=_RUNS_EVENT_KEY,
            maintain_order=True,
        )
        # Add rows that remain after event-key dedupe.
        if not with_event.is_empty():
            # Save them for the final output.
            parts.append(with_event)

        # If the input had no rows, there is nothing to join back together.
        if not parts:
            # Return the original empty batch.
            return batch

        # Join the groups, restore input order, and drop the temporary row number.
        return pl.concat(parts, how="diagonal_relaxed").sort("__row_nr").drop("__row_nr")

    @staticmethod
    def _dedupe_runs_batch(
        batch: pl.DataFrame,
        existing_lf: pl.LazyFrame,
        existing_schema: dict[str, pl.DataType],
    ) -> pl.DataFrame:
        # If existing data lacks the base key columns, we cannot compare rows.
        if not all(col in existing_schema for col in _RUNS_BASE_KEY):
            # Keep the whole batch.
            return batch

        # If there is no run-event column, or it is all null, all rows use the base key.
        if "run_event_id" not in batch.columns or batch.get_column("run_event_id").null_count() == batch.height:
            # Read the base keys that are already stored.
            existing_keys = existing_lf.select(_RUNS_BASE_KEY).unique().collect()
            # Keep only rows whose base key is not already stored.
            return batch.join(existing_keys, on=_RUNS_BASE_KEY, how="anti")

        # Add row numbers so we can restore the original order after splitting rows.
        indexed = batch.with_row_index("__row_nr")
        # Store the row groups that survive comparison with existing data.
        parts: list[pl.DataFrame] = []

        # Rows without run_event_id compare to existing rows by the base key.
        without_event = indexed.filter(pl.col("run_event_id").is_null())
        # Skip this work when there are no no-event rows.
        if not without_event.is_empty():
            # Read the base keys that are already stored.
            existing_base_keys = existing_lf.select(_RUNS_BASE_KEY).unique().collect()
            # Keep only no-event rows whose base key is not already stored.
            without_event = without_event.join(existing_base_keys, on=_RUNS_BASE_KEY, how="anti")
            # Add rows that remain after comparing with existing data.
            if not without_event.is_empty():
                # Save them for the final output.
                parts.append(without_event)

        # Rows with run_event_id compare to existing rows by the event key.
        with_event = indexed.filter(pl.col("run_event_id").is_not_null())
        # Skip this work when there are no event rows.
        if not with_event.is_empty():
            # Older stored data may not have run_event_id yet.
            if "run_event_id" in existing_schema:
                # Read the event keys that are already stored.
                existing_event_keys = existing_lf.select(_RUNS_EVENT_KEY).unique().collect()
                # Keep only event rows whose event key is not already stored.
                with_event = with_event.join(existing_event_keys, on=_RUNS_EVENT_KEY, how="anti")
            # Add rows that remain after comparing with existing data.
            if not with_event.is_empty():
                # Save them for the final output.
                parts.append(with_event)

        # If no rows survived, return an empty batch with the same columns.
        if not parts:
            # head(0) keeps the schema but removes all rows.
            return batch.head(0)

        # Join the groups, restore input order, and drop the temporary row number.
        return pl.concat(parts, how="diagonal_relaxed").sort("__row_nr").drop("__row_nr")

    @staticmethod
    def _dedupe_payload_batch(batch: pl.DataFrame, existing_lf: pl.LazyFrame) -> pl.DataFrame:
        existing_uids = existing_lf.select("run_uid").unique().collect().to_series().to_list()
        return batch.filter(~pl.col("run_uid").is_in(existing_uids))

    def _write_impl(self, records: Sequence[BaseRecord]) -> StorageWriteReceipt:
        if not records:
            return StorageWriteReceipt()

        run_table_files: dict[str, dict[str, str]] = defaultdict(dict)

        for table_name, rows in self._group_records(records).items():
            batch = pl.DataFrame(rows)
            if table_name == "runs":
                batch = self._dedupe_runs_within_batch(batch)

            output_dir = self._table_path(table_name)
            output_dir.mkdir(parents=True, exist_ok=True)

            parquet_glob = str(output_dir / "*.parquet")
            try:
                existing_lf = pl.scan_parquet(
                    parquet_glob,
                    missing_columns="insert",
                    storage_options=(self._storage_options or None),
                )
                existing_schema = existing_lf.collect_schema()
                self._validate_schema_compatibility(table_name, batch, existing_schema)

                if table_name == "runs":
                    batch = self._dedupe_runs_batch(batch, existing_lf, existing_schema)
                elif "run_uid" in existing_schema:
                    batch = self._dedupe_payload_batch(batch, existing_lf)

                if batch.is_empty():
                    continue
            except (pl.exceptions.ComputeError, pl.exceptions.SchemaError, FileNotFoundError, OSError):
                pass  # No existing files yet

            timestamp = int(time.time() * 1000)
            output_file = output_dir / f"{timestamp}_{uuid.uuid4().hex[:8]}.parquet"
            output_file_str = str(output_file)
            batch.write_parquet(output_file_str, storage_options=(self._storage_options or None))

            if table_name != "runs" and "run_uid" in batch.columns:
                run_uids = batch.get_column("run_uid").drop_nulls().unique().to_list()
                for run_uid in run_uids:
                    run_table_files[str(run_uid)][table_name] = output_file_str

        return StorageWriteReceipt(
            run_table_files={run_uid: dict(table_map) for run_uid, table_map in run_table_files.items()},
        )

    def write(self, records: Sequence[BaseRecord]) -> None:
        """Write records to storage, batched by table.

        Records are grouped by table name and written as one Parquet
        file per table per ``write()`` call.

        Dedupe semantics are table-specific:
        - capability payload tables are idempotent by ``run_uid`` across
          repeated write calls,
        - the ``runs`` table deduplicates rows without ``run_event_id`` by
          mapping key ``(run_uid, capability_table, entity_type, entity_id)``,
          and rows with ``run_event_id`` by provenance-aware mapping key
          ``(run_uid, capability_table, entity_type, entity_id, run_event_id)``.
        """
        _ = self._write_impl(records)

    def write_with_receipt(self, records: Sequence[BaseRecord]) -> StorageWriteReceipt:
        """Write records to storage and return concrete write metadata."""
        return self._write_impl(records)

    def list_tables(self) -> list[str]:
        """List available tables in the store."""
        try:
            if not self._path.exists():
                return []
            return [d.name for d in self._path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        except (FileNotFoundError, OSError):
            return []

    def get_run_uri(self, run_uid: str) -> str:
        """Return concrete parquet file path for payload data associated with ``run_uid``."""
        tables = [table_name for table_name in self.list_tables() if table_name != "runs"]
        if not tables:
            raise ValueError(f"Run UID {run_uid!r} not found: parquet store {str(self._path)!r} has no payload tables")

        for table_name in tables:
            parquet_glob = str(self._table_path(table_name) / "*.parquet")
            try:
                lf = pl.scan_parquet(
                    parquet_glob,
                    missing_columns="insert",
                    include_file_paths="__file_path",
                    storage_options=(self._storage_options or None),
                )
                schema = lf.collect_schema()
                if "run_uid" not in schema or "__file_path" not in schema:
                    continue

                match = lf.filter(pl.col("run_uid") == run_uid).select("__file_path").unique().collect()
            except (pl.exceptions.ComputeError, pl.exceptions.SchemaError, FileNotFoundError, OSError):
                continue

            if match.is_empty():
                continue

            return str(match["__file_path"][0])

        raise ValueError(f"Run UID {run_uid!r} not found in parquet store {str(self._path)!r}")

    def describe_table(self, table_name: str) -> dict[str, str]:
        """Get schema information for a table.

        Parameters
        ----------
        table_name
            Name of the table to describe.

        Returns
        -------
        dict[str, str]
            Mapping of column names to their Polars dtype strings.

        Raises
        ------
        ValueError
            If the table does not exist or has no data.
        """
        table_path = self._table_path(table_name)

        if not table_path.exists():
            raise ValueError(f"Table '{table_name}' does not exist")

        parquet_glob = str(table_path / "*.parquet")

        try:
            lf = pl.scan_parquet(
                parquet_glob,
                missing_columns="insert",
                storage_options=(self._storage_options or None),
            )
            schema = lf.collect_schema()
            return {name: str(dtype) for name, dtype in schema.items()}
        except Exception as e:
            raise ValueError(f"Could not read schema for table '{table_name}': {e}") from e

    def query_sql(self, sql: str) -> pl.DataFrame:
        """Execute a SQL query using Polars SQLContext.

        All available tables are registered with their table names.
        """
        tables = self.list_tables()
        if not tables:
            return pl.DataFrame()

        ctx = pl.SQLContext()

        for table_name in tables:
            table_path = self._table_path(table_name)
            files = sorted(table_path.glob("*.parquet"), key=str)
            if not files:
                continue

            try:
                frames = [
                    pl.scan_parquet(
                        str(f),
                        missing_columns="insert",
                        storage_options=(self._storage_options or None),
                    )
                    for f in files
                ]
                lf = pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]
                ctx.register(table_name, lf)
            except (pl.exceptions.ComputeError, pl.exceptions.SchemaError, FileNotFoundError, OSError):
                continue

        registered = ctx.tables()
        try:
            result = ctx.execute(sql).collect()
        except Exception as e:
            raise ValueError(
                f"SQL query failed.\n"
                f"  Registered tables: {registered}\n"
                f"  Error: {e}\n"
                f"  Note: Polars SQLContext supports a subset of SQL. "
                f"CTEs, some window functions, and certain aggregations may not be available."
            ) from e

        return result

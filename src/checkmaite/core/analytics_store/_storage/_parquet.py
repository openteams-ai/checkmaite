import time
import uuid
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from checkmaite.core.analytics_store._schema import BaseRecord


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
    path
        Base directory for storing parquet files.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path).expanduser().resolve()

    @property
    def path(self) -> Path:
        """The base directory where parquet files are stored."""
        return self._path

    def _table_path(self, table_name: str) -> Path:
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
    def _dedupe_runs_batch(
        batch: pl.DataFrame,
        existing_lf: pl.LazyFrame,
        existing_schema: dict[str, pl.DataType],
    ) -> pl.DataFrame:
        runs_key = ["run_uid", "capability_table", "entity_type", "entity_id"]
        if all(col in existing_schema for col in runs_key):
            existing_keys = existing_lf.select(runs_key).unique().collect()
            batch = batch.join(existing_keys, on=runs_key, how="anti")
        return batch

    @staticmethod
    def _dedupe_payload_batch(batch: pl.DataFrame, existing_lf: pl.LazyFrame) -> pl.DataFrame:
        existing_uids = existing_lf.select("run_uid").unique().collect().to_series().to_list()
        return batch.filter(~pl.col("run_uid").is_in(existing_uids))

    def write(self, records: Sequence[BaseRecord]) -> None:
        """Write records to storage, batched by table.

        Records are grouped by table name and written as one Parquet
        file per table per ``write()`` call.

        Dedupe semantics are table-specific:
        - capability payload tables are idempotent by ``run_uid`` across
          repeated write calls,
        - the ``runs`` table is deduplicated by mapping key
          ``(run_uid, capability_table, entity_type, entity_id)``.
        """
        if not records:
            return

        for table_name, rows in self._group_records(records).items():
            batch = pl.DataFrame(rows)
            if table_name == "runs":
                batch = batch.unique(
                    subset=["run_uid", "capability_table", "entity_type", "entity_id"],
                    maintain_order=True,
                )

            output_dir = self._table_path(table_name)
            output_dir.mkdir(parents=True, exist_ok=True)

            parquet_glob = str(output_dir / "*.parquet")
            try:
                existing_lf = pl.scan_parquet(parquet_glob, missing_columns="insert")
                existing_schema = existing_lf.collect_schema()
                self._validate_schema_compatibility(table_name, batch, existing_schema)

                if table_name == "runs":
                    batch = self._dedupe_runs_batch(batch, existing_lf, existing_schema)
                elif "run_uid" in existing_schema:
                    batch = self._dedupe_payload_batch(batch, existing_lf)

                if batch.is_empty():
                    continue
            except (pl.exceptions.ComputeError, pl.exceptions.SchemaError):
                pass  # No existing files yet

            timestamp = int(time.time() * 1000)
            output_file = output_dir / f"{timestamp}_{uuid.uuid4().hex[:8]}.parquet"
            batch.write_parquet(output_file)

    def list_tables(self) -> list[str]:
        """List available tables in the store."""
        if not self._path.exists():
            return []

        return [d.name for d in self._path.iterdir() if d.is_dir() and not d.name.startswith(".")]

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
            lf = pl.scan_parquet(parquet_glob, missing_columns="insert")
            schema = lf.collect_schema()
            return {name: str(dtype) for name, dtype in schema.items()}
        except Exception as e:
            raise ValueError(f"Could not read schema for table '{table_name}': {e}") from e

    def query_sql(self, sql: str) -> pl.DataFrame:
        """Execute a SQL query using Polars SQLContext.

        All available tables are registered with their table names.
        """
        if not self._path.exists():
            return pl.DataFrame()

        tables = self.list_tables()
        if not tables:
            return pl.DataFrame()

        ctx = pl.SQLContext()

        for table_name in tables:
            table_path = self._table_path(table_name)
            files = sorted(table_path.glob("*.parquet"))
            if not files:
                continue

            try:
                frames = [pl.scan_parquet(f) for f in files]
                lf = pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]
                ctx.register(table_name, lf)
            except (pl.exceptions.ComputeError, pl.exceptions.SchemaError, OSError):
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

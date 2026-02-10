from __future__ import annotations

import polars as pl

from jatic_ri.core.analytics_store._schema import BaseRecord, RunRecord
from jatic_ri.core.analytics_store._storage import StorageBackend
from jatic_ri.core.capability_core import CapabilityRunBase


class AnalyticsStore:
    """Main interface for the analytics store.

    Provides methods to write records from capability runs and query stored data.
    Data is organized into per-capability tables (e.g., "dataeval_cleaning",
    "maite_evaluation"), plus an automatic ``runs`` table that maps each
    ``run_uid`` to human-readable identifiers (dataset IDs, model IDs, etc.).

    The primary query interface is SQL-based via query_sql(), which provides
    maximum flexibility for filtering, joining, and aggregating data.

    Parameters
    ----------
    backend
        Storage backend to use (e.g., ParquetBackend).

    Example
    -------
    ```python
    from jatic_ri.core.analytics_store import AnalyticsStore, ParquetBackend

    # Create store
    store = AnalyticsStore(ParquetBackend("./my_analytics_store"))

    # Run capabilities and write to store
    store.write([run1, run2])

    # Query via SQL — use the 'runs' table to filter by dataset
    results = store.query_sql('''
        SELECT c.*
        FROM dataeval_cleaning c
        JOIN runs r ON c.run_uid = r.run_uid
        WHERE r.entity_type = 'dataset' AND r.entity_id = 'CIFAR-10'
    ''')
    ```
    """

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend

    @staticmethod
    def _build_run_records(run: CapabilityRunBase, capability_table: str) -> list[RunRecord]:
        """Build ``runs`` table rows for a single capability run."""
        run_uid = run.run_uid
        cap_id = run.capability_id
        return [
            *[
                RunRecord(
                    run_uid=run_uid,
                    capability_id=cap_id,
                    capability_table=capability_table,
                    entity_type="dataset",
                    entity_id=ds["id"],
                )
                for ds in run.dataset_metadata
            ],
            *[
                RunRecord(
                    run_uid=run_uid,
                    capability_id=cap_id,
                    capability_table=capability_table,
                    entity_type="model",
                    entity_id=m["id"],
                )
                for m in run.model_metadata
            ],
            *[
                RunRecord(
                    run_uid=run_uid,
                    capability_id=cap_id,
                    capability_table=capability_table,
                    entity_type="metric",
                    entity_id=mt["id"],
                )
                for mt in run.metric_metadata
            ],
        ]

    def write(self, runs: list[CapabilityRunBase]) -> list[BaseRecord]:
        """Write capability runs to storage.

        For each run, this method:
        1. Emits rows into the ``runs`` table (automatic — maps run_uid
           to dataset/model/metric IDs).
        2. Calls ``run.extract()`` for capability-specific records.

        Parameters
        ----------
        runs
            List of capability runs to write.

        Returns
        -------
        list[BaseRecord]
            The capability-specific records that were written.
        """
        if not runs:
            return []

        all_records: list[BaseRecord] = []
        capability_records: list[BaseRecord] = []

        for run in runs:
            # Capability-specific records (extracted first so we know the table name)
            extracted = list(run.extract())
            capability_records.extend(extracted)
            all_records.extend(extracted)

            # Auto-populate the runs table (only if extract produced records)
            if extracted:
                capability_table = extracted[0].table_name
                all_records.extend(self._build_run_records(run, capability_table))

        self._backend.write(all_records)

        return capability_records

    def list_tables(self) -> list[str]:
        """List available tables in the store.

        Returns
        -------
        list[str]
            List of table names (e.g., ["dataeval_cleaning", "maite_evaluation"]).
        """
        return self._backend.list_tables()

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

        Example
        -------
        ```python
        schema = store.describe_table("dataeval_cleaning")
        # {
        #     'run_uid': 'String',
        #     'dataset_id': 'String',
        #     'exact_duplicate_count': 'Int64',
        #     'exact_duplicate_ratio': 'Float64',
        #     ...
        # }
        ```
        """
        return self._backend.describe_table(table_name)

    # TODO: Add convenience helpers (e.g. query_by_entity) or pre-registered
    # SQL views that flatten the ``runs`` table JOIN for common queries like
    # filtering by dataset.  Currently users must write the full JOIN manually:
    #
    #     SELECT c.* FROM dataeval_cleaning c
    #     JOIN runs r ON c.run_uid = r.run_uid
    #     WHERE r.entity_type = 'dataset' AND r.entity_id = 'CIFAR-10'

    def query_sql(self, sql: str) -> pl.DataFrame:
        """Execute a SQL query against the store.

        Tables are named after capability types. Use these names in your SQL queries.

        Parameters
        ----------
        sql
            SQL query string.

        Returns
        -------
        pl.DataFrame
            Query results.

        Example
        -------
        ```python
        # Query dataeval_cleaning table
        df = store.query_sql("SELECT run_uid, exact_duplicate_count FROM dataeval_cleaning")

        # Filter by date
        df = store.query_sql('''
            SELECT * FROM maite_evaluation
            WHERE created_at >= '2024-01-18'
        ''')

        # Join across capability tables (single-dataset convention)
        df = store.query_sql('''
            SELECT d.dataset_id, d.image_count, m.metric_value
            FROM dataeval_cleaning d
            JOIN maite_evaluation m ON d.dataset_id = m.dataset_id
            WHERE m.metric_name = 'accuracy'
        ''')
        ```
        """
        return self._backend.query_sql(sql)

from __future__ import annotations

import polars as pl

from checkmaite.core.analytics_store._provenance import Provenance, ProvenanceLike
from checkmaite.core.analytics_store._schema import BaseRecord, RunRecord
from checkmaite.core.analytics_store._storage import StorageBackend, StorageWriteReceipt
from checkmaite.core.capability_core import CapabilityRunBase


class AnalyticsStore:
    """Main interface for the analytics store.

    Provides methods to write records from capability runs and query stored data.
    Data is organized into per-capability tables (e.g., "dataeval_cleaning",
    "maite_evaluation"), plus an automatic ``runs`` table that maps each
    ``run_uid`` to human-readable identifiers (dataset IDs, model IDs, etc.).

    Write semantics distinguish payload artifacts from run-event history:
    payload tables are deduplicated by ``run_uid`` across separate write calls,
    while the ``runs`` table is deduplicated by the mapping key
    ``(run_uid, capability_table, entity_type, entity_id)`` unless callers
    provide an explicit ``run_event_id``. Run event IDs let separate jobs or
    other explicit run events preserve distinct run-history rows even when they
    reference the same deduplicated payload artifacts.

    The primary query interface is SQL-based via query_sql(), which provides
    maximum flexibility for filtering, joining, and aggregating data.

    Parameters
    ----------
    backend
        Storage backend to use (e.g., ParquetBackend).

    Example
    -------
    ```python
    from checkmaite.core.analytics_store import AnalyticsStore, ParquetBackend

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
    def _build_run_records(
        run: CapabilityRunBase,
        capability_table: str,
        provenance: Provenance,
    ) -> list[RunRecord]:
        """Build ``runs`` table rows for a single capability run.

        One row is emitted per mapped entity (dataset, model, metric). Repeated
        writes of the same run remain idempotent unless callers provide
        distinct explicit ``run_event_id`` values.
        """
        run_uid = run.run_uid
        cap_id = run.capability_id
        provenance_kwargs = provenance.model_dump(mode="python", exclude_none=True)
        return [
            *[
                RunRecord(
                    run_uid=run_uid,
                    capability_id=cap_id,
                    capability_table=capability_table,
                    entity_type="dataset",
                    entity_id=ds["id"],
                    **provenance_kwargs,
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
                    **provenance_kwargs,
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
                    **provenance_kwargs,
                )
                for mt in run.metric_metadata
            ],
        ]

    def _collect_records(
        self,
        runs: list[CapabilityRunBase],
        *,
        provenance: ProvenanceLike | None = None,
    ) -> list[BaseRecord]:
        all_records: list[BaseRecord] = []
        base_provenance = Provenance.from_optional(provenance)

        for run in runs:
            # Capability-specific records (extracted first so we know the table name)
            extracted = list(run.extract())
            all_records.extend(extracted)

            # Auto-populate the runs table (only if extract produced records)
            if extracted:
                capability_tables = {record.table_name for record in extracted}
                if len(capability_tables) != 1:
                    raise ValueError(
                        "AnalyticsStore.write() expects extract() to emit records "
                        f"for exactly one payload table, got {sorted(capability_tables)!r}"
                    )

                capability_table = next(iter(capability_tables))
                all_records.extend(self._build_run_records(run, capability_table, base_provenance))

        return all_records

    def write(
        self,
        runs: list[CapabilityRunBase],
        *,
        provenance: ProvenanceLike | None = None,
    ) -> None:
        """Write capability runs to storage.

        For each run, this method:
        1. Emits rows into the ``runs`` table (automatic — maps run_uid
           to dataset/model/metric IDs).
        2. Calls ``run.extract()`` for capability-specific records.

        Repeated writes are expected to be idempotent for payload tables at the
        storage-backend layer: capability payload tables should not accumulate
        duplicate data for the same ``run_uid`` across separate writes. The
        ``runs`` table preserves old idempotent mapping semantics unless callers
        provide explicit ``run_event_id`` values, as job backends do.

        Parameters
        ----------
        runs
            List of capability runs to write.
        provenance
            Optional explicit provenance values applied to every generated
            ``runs`` row. Process defaults are not read automatically; pass
            ``get_provenance_defaults()`` if you want to use configured defaults.
        """
        _ = self.write_with_receipt(
            runs,
            provenance=provenance,
        )

    def write_with_receipt(
        self,
        runs: list[CapabilityRunBase],
        *,
        provenance: ProvenanceLike | None = None,
    ) -> StorageWriteReceipt:
        """Write capability runs to storage and return concrete write metadata.

        This is the receipt-aware variant used by job-submission paths that need
        to resolve exact storage locations for newly written runs.
        """
        if not runs:
            return StorageWriteReceipt()

        return self._backend.write_with_receipt(
            self._collect_records(
                runs,
                provenance=provenance,
            )
        )

    def list_tables(self) -> list[str]:
        """List available tables in the store.

        Returns
        -------
        list[str]
            List of table names (e.g., ["dataeval_cleaning", "maite_evaluation"]).
        """
        return self._backend.list_tables()

    def get_run_uri(self, run_uid: str) -> str:
        """Return a concrete URI for ``run_uid`` in this store."""
        return self._backend.get_run_uri(run_uid)

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

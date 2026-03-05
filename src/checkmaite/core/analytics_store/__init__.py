"""Analytics Store - structured storage for capability run data.

This module provides storage for capability run outputs in a queryable format,
with each capability type having its own table with typed fields.

Key Components
--------------
- BaseRecord: Base class for capability-specific records
- AnalyticsStore: Main API for writing runs and querying via SQL
- ParquetBackend: Default storage backend using Parquet files

Usage
-----
```python
from checkmaite.core.analytics_store import AnalyticsStore, ParquetBackend

# Create store
store = AnalyticsStore(ParquetBackend("./my_analytics_store"))

# Run capabilities
run1 = capability.run(datasets=[ds1])
run2 = capability.run(datasets=[ds2])

# Write runs to store
store.write([run1, run2])

# List available tables
tables = store.list_tables()  # ["runs", "dataeval_cleaning", "maite_evaluation"]

# Query capability data filtered by dataset (via the auto-populated 'runs' table)
results = store.query_sql('''
    SELECT c.*
    FROM dataeval_cleaning c
    JOIN runs r ON c.run_uid = r.run_uid
    WHERE r.entity_type = 'dataset' AND r.entity_id = 'CIFAR-10'
''')

# Join across capability tables (single-dataset capabilities use dataset_id)
results = store.query_sql('''
    SELECT d.image_count, m.metric_value
    FROM dataeval_cleaning d
    JOIN maite_evaluation m ON d.dataset_id = m.dataset_id
    WHERE m.metric_name = 'accuracy'
''')
```

Custom Records
--------------
To add extraction support to a new capability, create a record class
that extends BaseRecord (a Pydantic model) and override the ``extract()``
method on your CapabilityRunBase subclass.

**All record fields must be scalar types** (str, int, float, bool, bytes,
datetime, or Optional variants).  Nested types (list, dict, etc.) are
rejected at class-definition time.  If a capability produces
variable-length data, ``extract()`` should return multiple records.

**Convention:** Single-dataset capabilities should include a field
named ``dataset_id: str``.  This enables direct JOINs across
capability tables without routing through the ``runs`` table.
Multi-dataset capabilities do not currently have a convention for
cross-capability JOINs; use the ``runs`` table instead.

```python
from checkmaite.core.analytics_store import BaseRecord


class MyCapabilityRecord(BaseRecord, table_name="my_capability"):
    dataset_id: str  # Convention: enables cross-capability JOINs
    my_metric: float
    my_count: int


class MyCapabilityRun(CapabilityRunBase[MyConfig, MyOutputs]):
    def extract(self) -> list[MyCapabilityRecord]:
        run_uid = self.compute_uid(...)
        return [
            MyCapabilityRecord(
                run_uid=run_uid,
                dataset_id=d["id"],
                my_metric=self.outputs.some_value,
                my_count=self.outputs.count,
            )
            for d in self.dataset_metadata
        ]
```
"""

from checkmaite.core.analytics_store._schema import BaseRecord, RunRecord
from checkmaite.core.analytics_store._storage import ParquetBackend, StorageBackend
from checkmaite.core.analytics_store._store import AnalyticsStore

__all__ = [
    "BaseRecord",
    "AnalyticsStore",
    "ParquetBackend",
    "RunRecord",
    "StorageBackend",
]

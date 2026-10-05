# Save & query results

Persist capability runs to an analytics store so you can compare results across runs, datasets, and capabilities with SQL.

## 1. Create a store

```python
from checkmaite.core.analytics_store import AnalyticsStore, ParquetBackend

store = AnalyticsStore(ParquetBackend("./analytics_store"))
```

The Parquet backend writes plain Parquet files, one table per capability, under the given directory.

## 2. Write runs

Pass the runs returned by `capability.run(...)` to `store.write(...)`:

```python
store.write([bias_run, feasibility_run])
store.list_tables()
```

Writing the same run again is a no-op, so re-executing a notebook doesn't duplicate rows.

## 3. Query and join

`store.query_sql(...)` returns a Polars DataFrame:

```python
store.query_sql("SELECT dataset_id, ber_upper, ber_lower FROM dataeval_feasibility")
```

Join capability tables on shared columns such as `dataset_id`:

```python
store.query_sql(
    "SELECT b.dataset_id, b.balance_mean, f.ber_upper "
    "FROM dataeval_bias b "
    "JOIN dataeval_feasibility f USING (dataset_id)"
)
```

To filter by model, metric, or any other entity, join through the auto-populated `runs` table:

```python
store.query_sql(
    "SELECT e.* FROM maite_evaluation e "
    "JOIN runs r ON e.run_uid = r.run_uid "
    "WHERE r.entity_type = 'model' AND r.entity_id = 'resnet50'"
)
```

## Going further

- Table schemas, schema evolution, and adding store support to a capability: [Analytics store guide](../development/analytics_store_guide.ipynb).
- Writing from remote workers: [Analytics store in distributed execution](../development/job_submission/analytics_store.md) and [Job backend configuration](../development/job_submission/configure_job_backend.md).

## Related tutorials

- [Analytics Store](../tool-usage/analytics_store_tutorial.ipynb): writes five capabilities to one store and joins them.
- [Object Detection Workflow via API](checkmaite_api_od.ipynb) and [Image Classification Workflow via API](checkmaite_api_ic.ipynb): produce the runs you would save.
- DataEval [Bias](../tool-usage/dataeval_bias_tutorial.ipynb), [Feasibility](../tool-usage/dataeval_feasibility_tutorial.ipynb), [Linting](../tool-usage/dataeval_linting_tutorial.ipynb), and [Shift](../tool-usage/dataeval_shift_tutorial.ipynb): dataset runs with their own analytics tables.
- [NRTK](../tool-usage/nrtk_tutorial.ipynb) and [XAITK](../tool-usage/xaitk_tutorial.ipynb): model-evaluation runs with their own analytics tables.

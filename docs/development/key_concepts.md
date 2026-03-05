# Key Concepts: Capabilities, Runs, and Caching

This page documents the core abstractions used to define, execute, and cache evaluations: **Capabilities**, **Runs**, and the **caching layer**.

---

## Core Concepts

### Capability

A **Capability** represents a specific evaluation task — for example, running model inference and computing metrics on a dataset. It is the top-level abstraction that users interact with.

A Capability is responsible for:

- Defining the **configuration** accepted by the evaluation (via a `Config` object)
- Knowing how to **execute** an evaluation (`_run`)
- Knowing how to **check the cache** before executing (handled by `run`)



---

### Run

A **Run** is an object that stores everything associated with a *specific execution* of a Capability. This includes:

- The **configuration** for that execution (e.g., model, dataset, metric settings)
- The **outputs** produced (e.g., predictions, metric results)
- A method `collect_report_consumables()` for generating visualizations or reports from those outputs

Outputs are serialized using **Pydantic**, which handles conversion of Python objects (numpy arrays, pandas DataFrames, torch tensors, etc.) to bytes for storage in the cache. Custom serialization for additional types can be registered via `binary_deserializer.register(...)`.


---

### Implementing a New Capability

Each tool must implement:

1. **`Config`** — A Pydantic model declaring what configuration options the Capability accepts. Can be `pass` if no configuration is needed.
2. **`Outputs`** — A Pydantic model declaring what outputs will be stored and cached.
3. **`Run`** — Contains the `Config`, the `Outputs`, and the `collect_report_consumables()` method for visualization.
4. **Capability class** — aka the "Runner". Implements `_run(...)`, the actual execution logic. Calls internal helpers (e.g., `maite_evaluate`) to produce outputs.

See the **Baseline Evaluation Capability** for the simplest implementation.

---

## Caching

The caching layer is designed to avoid redundant computation. There are two levels of caching:

### 1. Capability-level Cache

When a Capability is executed with `use_cache=True` (the default), it checks whether a Run with the same configuration and inputs has already been completed. If a cache hit is found, the stored Run object is returned immediately — no computation occurs.


### 2. Prediction/Evaluation Cache

At a lower level, individual `predict` and `evaluate` calls (e.g., calls to `maite.evaluate`) are also cached globally. If two different Capabilities within the same pipeline call `evaluate` with the same model, dataset, and metric configuration, the second call will reuse the result from the first.

This cache is controlled by the same `use_cache` flag. When `use_cache=False`, both the capability-level and prediction/evaluation-level caches are bypassed.

> **Note:** It is not currently possible to disable the prediction/evaluation cache independently of the capability cache. Both are toggled together via `use_cache`.

---

### Cache Key Generation

Cache hits are determined by matching on identifiers such as:
- `model_id`
- `dataset_id`
- `metric_id`

**These are user-supplied metadata fields and must be unique.** The cache does not perform content-based hashing (e.g., checksumming image files) for performance reasons. It is the responsibility of the caller to ensure that IDs accurately reflect the data being passed in.

> ⚠️ **Important:** If you run the same model or dataset under the same ID but with different underlying content, you will get incorrect cache hits. When using this library programmatically (e.g., from a notebook), ensure IDs are managed carefully. In a production environment with a model registry or dataset warehouse, these IDs should be derived automatically from versioned artifacts.

---

### Configuring the Cache

The cache behavior is controlled by the `use_cache` parameter on the Capability's `run` method:

```python
# Use cache (default) — will return cached result if available
capability.run(model=my_model, dataset=my_dataset, use_cache=True)

# Bypass cache — always recompute
capability.run(model=my_model, dataset=my_dataset, use_cache=False)
```

---

## Input Flexibility (Type Coercion)

The checkmaite accepts flexible input types at its public API boundary and normalizes them internally. For example, an image can be passed as:

- A file path (`str` or `Path`)
- Raw bytes
- A `BufferedIOBase` object
- A PIL `Image` object

Internally, all images are normalized to PIL `Image` objects before any processing occurs. This coercion is handled automatically via Pydantic validators and follows [Postel's Law](https://en.wikipedia.org/wiki/Robustness_principle): *be flexible in what you accept, strict in what you emit*.

Similarly, PySpark DataFrames are automatically converted to pandas DataFrames at the boundary.

This means internal code never needs to check input types — it can always assume inputs are in the canonical internal format.

---

## Reporting and Visualization

Each Run exposes a `collect_report_consumables()` method that prepares outputs for visualization or reporting. Report generation is handled by pluggable backends located in the `report/` submodule:

- **Gradient-based reports** (legacy, optional dependency) — generates visual outputs using the Gradient library. Will emit a deprecation warning if used.
- **Markdown reports** — generates a structured `.md` file summarizing outputs. This is the recommended approach going forward.

Both are available as separate functions on the Run object, so end users can choose the format appropriate to their context.

---

## Optional Dependencies

UI-related dependencies (Panel, HoloViews, JupyterLab, etc.) are **optional** and not installed by default. This keeps the base package lightweight for use in non-interactive / production environments.

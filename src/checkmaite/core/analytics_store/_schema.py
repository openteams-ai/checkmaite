import types
from datetime import datetime, timezone
from typing import Any, ClassVar, Union, get_args, get_origin

import pydantic
from pydantic import Field

# Scalar types permitted as record fields.  All other types (list, dict, set,
# nested BaseModel, etc.) are rejected at class-definition time so that
# records are always flat and queryable via SQL.
_SCALAR_TYPES: tuple[type, ...] = (str, int, float, bool, bytes, datetime)


def _is_scalar(annotation: Any) -> bool:
    """Return True if *annotation* resolves to a flat, SQL-friendly type.

    Handles ``Optional[X]`` (i.e. ``X | None``) by unwrapping the union
    and checking that every non-None member is scalar.
    """
    if annotation is None or annotation is type(None):
        return True

    origin = get_origin(annotation)

    # X | None  /  Optional[X]  /  Union[X, None]
    if origin is Union or origin is types.UnionType:
        return all(_is_scalar(arg) for arg in get_args(annotation))

    # Any bare generic (list[…], dict[…], set[…], …) is nested
    if origin is not None:
        return False

    return isinstance(annotation, type) and issubclass(annotation, _SCALAR_TYPES)


class BaseRecord(pydantic.BaseModel):
    """Base class for analytics store records.

    Provides common fields that all run-specific records should include.
    Subclasses add their own capability-specific fields.

    **All fields must be scalar types** (str, int, float, bool, bytes,
    datetime, or Optional variants).  Nested types such as ``list``,
    ``dict``, or other Pydantic models are rejected at class-definition
    time so that every record maps directly to a flat SQL row.

    If a capability produces variable-length data (e.g. per-metric
    results), the ``extract()`` method should return multiple records
    — one per logical entity.

    To look up the human-readable context for a ``run_uid`` (dataset IDs,
    model IDs, capability name, etc.), JOIN against the automatically
    populated ``runs`` table.

    Cross-capability conventions
    ---------------------------
    To enable direct JOINs across capability tables without routing
    through the ``runs`` table:

    - **Single-dataset capabilities** should include a field named
      ``dataset_id: str`` containing the dataset identifier.  This
      allows cross-capability queries like::

          SELECT c.exact_duplicate_ratio, m.metric_value
          FROM dataeval_cleaning c
          JOIN maite_evaluation m ON c.dataset_id = m.dataset_id

    - **Multi-dataset capabilities** (two or more input datasets) do
      not currently have a convention for cross-capability JOINs.
      Use the ``runs`` table to correlate by entity ID instead.
      This may change in future.

    Schema evolution
    ----------------
    - **Adding fields**: Supported. Existing data will have ``None`` for
      the new column. Use ``Optional`` types for fields that may not
      exist in historical data.
    - **Removing fields**: Supported. Old data retains the column but
      new records won't populate it.
    - **Renaming fields**: Not supported. Requires manual migration
      at the storage backend level.
    - **Changing field types**: Not supported. The Parquet backend
      rejects writes where a column's type conflicts with existing data.

    Attributes
    ----------
    run_uid
        SHA-256 hash linking to the capability run.
    created_at
        Timestamp when the record was created. Auto-generated.
    """

    model_config = pydantic.ConfigDict(
        extra="forbid",
    )

    #: Set via keyword argument: ``class MyRecord(BaseRecord, table_name="x")``
    table_name: ClassVar[str]

    run_uid: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __init_subclass__(cls, table_name: str = "") -> None:
        if not table_name:
            raise TypeError(
                f"{cls.__qualname__} must pass 'table_name' as a keyword "
                f"argument (e.g. class {cls.__qualname__}(BaseRecord, "
                f'table_name="my_capability")).'
            )
        super().__init_subclass__()
        cls.table_name = table_name

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Validate that all fields are scalar types.

        Called by Pydantic's metaclass after ``model_fields`` is populated,
        so field annotations are available for inspection.  This runs at
        class-definition time (import), not at first instantiation.

        We use ``__pydantic_init_subclass__`` rather than the standard
        ``__init_subclass__`` because ``model_fields`` is not yet populated
        when ``__init_subclass__`` executes in Pydantic v2 — see
        https://github.com/pydantic/pydantic/discussions/7177.
        """
        super().__pydantic_init_subclass__(**kwargs)

        base_fields = set(BaseRecord.model_fields)

        for field_name, field_info in cls.model_fields.items():
            if field_name in base_fields:
                continue
            if not _is_scalar(field_info.annotation):
                raise TypeError(
                    f"Field '{field_name}' on {cls.__qualname__} uses non-scalar "
                    f"type {field_info.annotation!r}. BaseRecord subclasses must use only "
                    f"flat types ({', '.join(t.__name__ for t in _SCALAR_TYPES)}, "
                    f"or Optional variants). If you need variable-length data, "
                    f"return multiple records from extract()."
                )


class RunRecord(BaseRecord, table_name="runs"):
    """A row in the ``runs`` table, emitted automatically by the store.

    Maps a ``run_uid`` to the human-readable identifiers that produced it
    (capability, datasets, models, metrics). One record is written per unique
    ``(run_uid, capability_table, entity_type, entity_id)`` mapping so that
    users can filter on any dimension with plain SQL::

        SELECT c.*
        FROM dataeval_cleaning c
        JOIN runs r ON c.run_uid = r.run_uid
        WHERE r.entity_type = 'dataset' AND r.entity_id = 'CIFAR-10'
    """

    capability_id: str
    capability_table: str  # table_name of the capability's record class
    entity_type: str  # "dataset", "model", or "metric"
    entity_id: str

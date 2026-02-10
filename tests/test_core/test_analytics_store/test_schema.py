from datetime import datetime
from typing import Optional, Union

import pydantic
import pytest

from jatic_ri.core.analytics_store._schema import BaseRecord, RunRecord, _is_scalar


class TestIsScalarAcceptsScalarTypes:
    """Each of the six allowed scalar types should be accepted."""

    @pytest.mark.parametrize("typ", [str, int, float, bool, bytes, datetime])
    def test_bare_scalar(self, typ: type) -> None:
        assert _is_scalar(typ) is True


class TestIsScalarAcceptsOptionalScalars:
    """Optional[X] and X | None should be accepted for each scalar type."""

    @pytest.mark.parametrize("typ", [str, int, float, bool, bytes, datetime])
    def test_optional_typing(self, typ: type) -> None:
        assert _is_scalar(Optional[typ]) is True  # noqa: UP007

    @pytest.mark.parametrize("typ", [str, int, float, bool, bytes, datetime])
    def test_pipe_none(self, typ: type) -> None:
        assert _is_scalar(typ | None) is True


class TestIsScalarAcceptsUnionsOfScalars:
    """Unions containing only scalar types (with or without None) should pass."""

    def test_union_two_scalars(self) -> None:
        assert _is_scalar(Union[str, int]) is True  # noqa: UP007

    def test_pipe_two_scalars(self) -> None:
        assert _is_scalar(str | int) is True

    def test_union_three_scalars_and_none(self) -> None:
        assert _is_scalar(str | int | None) is True

    def test_union_all_scalars(self) -> None:
        assert _is_scalar(Union[str, int, float, bool]) is True  # noqa: UP007


class TestIsScalarRejectsNonScalarTypes:
    """Non-scalar types should be rejected."""

    @pytest.mark.parametrize("typ", [list, dict, set, tuple])
    def test_bare_collection(self, typ: type) -> None:
        assert _is_scalar(typ) is False

    def test_parameterised_list(self) -> None:
        assert _is_scalar(list[str]) is False

    def test_parameterised_dict(self) -> None:
        assert _is_scalar(dict[str, int]) is False

    def test_parameterised_set(self) -> None:
        assert _is_scalar(set[str]) is False

    def test_parameterised_tuple(self) -> None:
        assert _is_scalar(tuple[str, ...]) is False

    def test_pydantic_model(self) -> None:
        class Inner(pydantic.BaseModel):
            x: int

        assert _is_scalar(Inner) is False


class TestIsScalarRejectsUnionsContainingNonScalars:
    """A union is rejected if any non-None member is non-scalar."""

    def test_union_scalar_and_list(self) -> None:
        assert _is_scalar(Union[str, list[int]]) is False  # noqa: UP007

    def test_optional_list(self) -> None:
        assert _is_scalar(Optional[list[str]]) is False  # noqa: UP007

    def test_optional_dict(self) -> None:
        assert _is_scalar(dict[str, int] | None) is False

    def test_pipe_scalar_and_dict(self) -> None:
        assert _is_scalar(str | dict[str, int]) is False


class TestIsScalarNoneEdgeCases:
    """None literal and NoneType are both accepted (defensive)."""

    def test_none_literal(self) -> None:
        assert _is_scalar(None) is True

    def test_nonetype(self) -> None:
        assert _is_scalar(type(None)) is True


def test_missing_table_name_raises() -> None:
    with pytest.raises(TypeError, match="must pass 'table_name'"):

        class NoTableName(BaseRecord):
            value: str


def test_inherited_table_name_not_accepted() -> None:
    """Subclasses without an explicit table_name keyword argument are rejected.

    Although the child inherits the parent's ClassVar, __init_subclass__
    receives table_name="" (the default) when the keyword is omitted,
    which triggers the guard.  This prevents accidentally sharing a
    table between parent and child.
    """

    class Parent(BaseRecord, table_name="parent"):
        foo: str

    with pytest.raises(TypeError, match="must pass 'table_name'"):

        class Child(Parent):
            bar: int


def test_subclass_with_own_table_name_accepted() -> None:
    class Parent(BaseRecord, table_name="parent"):
        foo: str

    class Child(Parent, table_name="child"):
        bar: int

    assert Child.table_name == "child"


def test_non_scalar_field_raises() -> None:
    with pytest.raises(TypeError, match="non-scalar"):

        class BadRecord(BaseRecord, table_name="bad"):
            items: list[str]


def test_nested_model_field_raises() -> None:
    class Inner(pydantic.BaseModel):
        x: int

    with pytest.raises(TypeError, match="non-scalar"):

        class BadRecord(BaseRecord, table_name="bad"):
            nested: Inner


def test_dict_field_raises() -> None:
    with pytest.raises(TypeError, match="non-scalar"):

        class BadRecord(BaseRecord, table_name="bad"):
            mapping: dict[str, int]


def test_optional_list_field_raises() -> None:
    with pytest.raises(TypeError, match="non-scalar"):

        class BadRecord(BaseRecord, table_name="bad"):
            maybe_items: list[str] | None


def test_all_scalar_types_accepted_as_fields() -> None:
    """A record using every allowed scalar type should instantiate successfully."""

    class KitchenSink(BaseRecord, table_name="kitchen_sink"):
        a_str: str
        a_int: int
        a_float: float
        a_bool: bool
        a_bytes: bytes
        a_datetime: datetime
        opt_str: str | None = None
        opt_int: int | None = None

    record = KitchenSink(
        run_uid="uid",
        a_str="hello",
        a_int=1,
        a_float=1.0,
        a_bool=True,
        a_bytes=b"x",
        a_datetime=datetime.now(),
    )
    assert record.a_str == "hello"


def test_run_record_is_base_record_subclass() -> None:
    assert issubclass(RunRecord, BaseRecord)


def test_run_record_has_scalar_enforcement() -> None:
    """RunRecord inherits scalar enforcement from BaseRecord."""
    assert RunRecord.table_name == "runs"
    # Verify it has the expected fields
    rec = RunRecord(
        run_uid="r1",
        capability_id="cap",
        capability_table="tbl",
        entity_type="dataset",
        entity_id="ds1",
    )
    assert rec.run_uid == "r1"
    assert rec.created_at is not None


def test_run_record_forbids_extra_fields() -> None:
    with pytest.raises(pydantic.ValidationError):
        RunRecord(
            run_uid="r1",
            capability_id="cap",
            capability_table="tbl",
            entity_type="dataset",
            entity_id="ds1",
            surprise="extra",  # type: ignore[call-arg]
        )


def test_model_dump_produces_flat_dict() -> None:
    class SimpleRecord(BaseRecord, table_name="simple"):
        name: str
        score: float
        label: int | None = None

    rec = SimpleRecord(run_uid="uid1", name="test", score=0.95)
    dumped = rec.model_dump(mode="python")

    assert isinstance(dumped, dict)
    assert dumped["run_uid"] == "uid1"
    assert dumped["name"] == "test"
    assert dumped["score"] == 0.95
    assert dumped["label"] is None
    assert isinstance(dumped["created_at"], datetime)


def test_extra_fields_forbidden() -> None:
    class StrictRecord(BaseRecord, table_name="strict"):
        value: int

    with pytest.raises(pydantic.ValidationError):
        StrictRecord(run_uid="uid1", value=1, extra_field="nope")  # type: ignore[call-arg]

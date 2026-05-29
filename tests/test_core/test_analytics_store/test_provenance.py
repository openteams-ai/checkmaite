from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import pydantic
import pytest

from checkmaite import configure_provenance, get_provenance_defaults, reset_provenance
from checkmaite.core.analytics_store import Provenance, RunRecord
from checkmaite.core.analytics_store import _provenance as provenance_module


@pytest.fixture(autouse=True)
def _reset_provenance_defaults() -> Iterator[None]:
    reset_provenance()
    yield
    reset_provenance()


def test_configure_provenance_sets_and_resets_process_defaults() -> None:
    reset_provenance()

    configured = configure_provenance(
        user_id="alice@company.com",
        workspace_id="workspace-a",
        environment="databricks",
        executor="ray",
        cluster_id="cluster-42",
        request_id="req-123",
    )

    assert configured.user_id == "alice@company.com"
    assert configured.workspace_id == "workspace-a"
    assert configured.environment == "databricks"
    assert get_provenance_defaults().request_id == "req-123"

    reset = reset_provenance()

    assert reset == Provenance()
    assert get_provenance_defaults() == Provenance()


def test_configure_provenance_rejects_per_run_event_fields() -> None:
    with pytest.raises(TypeError):
        configure_provenance(run_event_id="invoke-1")  # type: ignore[call-arg]


def test_environment_provenance_loader_freezes_present_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHECKMAITE_PROVENANCE_USER_ID", "env-alice")
    monkeypatch.setenv("CHECKMAITE_PROVENANCE_WORKSPACE_ID", "workspace-a")

    loaded, frozen_fields = provenance_module._load_environment_provenance()

    assert loaded.user_id == "env-alice"
    assert loaded.workspace_id == "workspace-a"
    assert loaded.environment is None
    assert frozen_fields == {"user_id", "workspace_id"}


def test_environment_provenance_loader_ignores_empty_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHECKMAITE_PROVENANCE_USER_ID", "")

    loaded, frozen_fields = provenance_module._load_environment_provenance()

    assert loaded.user_id is None
    assert "user_id" not in frozen_fields


def test_configure_provenance_preserves_and_freezes_environment_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance_module, "_ENV_PROVENANCE", Provenance(user_id="env-alice"))
    monkeypatch.setattr(provenance_module, "_FROZEN_PROVENANCE_FIELDS", frozenset({"user_id"}))

    configured = configure_provenance(workspace_id="workspace-a")

    assert configured.user_id == "env-alice"
    assert configured.workspace_id == "workspace-a"

    with pytest.raises(ValueError, match="user_id"):
        configure_provenance(user_id="bob")


def test_provenance_instances_are_immutable() -> None:
    provenance = Provenance(user_id="alice")

    with pytest.raises(pydantic.ValidationError, match="frozen"):
        provenance.user_id = "bob"


def test_provenance_rejects_naive_timestamps() -> None:
    with pytest.raises(pydantic.ValidationError, match="timezone-aware"):
        Provenance(submitted_at=datetime(2026, 1, 1, 12, 0, 0))


def test_provenance_normalizes_aware_timestamps_to_utc() -> None:
    submitted_at = datetime(2026, 1, 1, 7, 0, 0, tzinfo=timezone(timedelta(hours=-5)))

    provenance = Provenance(submitted_at=submitted_at)

    assert provenance.submitted_at == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_provenance_fields_are_present_on_run_record() -> None:
    assert set(Provenance.model_fields) <= set(RunRecord.model_fields)


def test_provenance_merge_keeps_defaults_and_applies_non_none_overrides() -> None:
    base = Provenance(user_id="alice", workspace_id="workspace-a", backend="local")

    merged = base.merge({"backend": "ray", "job_id": "job-1", "workspace_id": None})

    assert merged.user_id == "alice"
    assert merged.workspace_id == "workspace-a"
    assert merged.backend == "ray"
    assert merged.job_id == "job-1"

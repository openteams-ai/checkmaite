from types import SimpleNamespace

import pytest
from ray import serve

from checkmaite.core.serving.rayserve import print_serve_status


def test_print_serve_status_smoke() -> None:
    print_serve_status()


def test_print_serve_status_handles_runtime_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def raise_runtime_error():
        raise RuntimeError("no ray")

    monkeypatch.setattr(serve, "status", raise_runtime_error)

    print_serve_status()

    assert "Could not query cluster status" in capsys.readouterr().out


def test_print_serve_status_prints_deployments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    deployment = SimpleNamespace(status=SimpleNamespace(value="HEALTHY"), replica_states={"RUNNING": 2})
    app = SimpleNamespace(status="RUNNING", deployments={"model": deployment})
    monkeypatch.setattr(serve, "status", lambda: SimpleNamespace(applications={"app": app}))

    print_serve_status()

    output = capsys.readouterr().out
    assert "Application: app" in output
    assert "Deployment: model" in output
    assert "Replicas: 2" in output

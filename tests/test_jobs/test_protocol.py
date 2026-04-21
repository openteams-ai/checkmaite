from checkmaite.jobs import JobError, JobStatus, JobTimeoutError


def test_job_status_terminal_flags() -> None:
    assert JobStatus.PENDING.is_terminal is False
    assert JobStatus.RUNNING.is_terminal is False
    assert JobStatus.COMPLETED.is_terminal is True
    assert JobStatus.FAILED.is_terminal is True
    assert JobStatus.CANCELLED.is_terminal is True


def test_job_errors_include_job_id() -> None:
    err = JobError("abc123", "boom")
    assert "abc123" in str(err)
    assert "boom" in str(err)

    timeout = JobTimeoutError("job-1", 3.5)
    assert "job-1" in str(timeout)
    assert "3.500" in str(timeout)

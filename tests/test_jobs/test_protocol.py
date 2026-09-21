from checkmaite.jobs import JobError, JobStatus, JobSubmissionError, JobTimeoutError


def test_job_status_terminal_flags() -> None:
    assert JobStatus.PENDING.is_terminal is False
    assert JobStatus.SCHEDULING.is_terminal is False
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


def test_job_submission_error_exposes_phase_and_cause_details() -> None:
    error = JobSubmissionError("starting controller", ValueError("bad configuration"), "job-1")

    assert error.phase == "starting controller"
    assert error.job_id == "job-1"
    assert error.error_type == "ValueError"
    assert error.detail == "bad configuration"

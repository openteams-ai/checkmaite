"""Typed capability report models."""

from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from typing_extensions import TypedDict

MAX_INLINE_REPORT_BYTES = 256 * 1024
"""Maximum UTF-8 size allowed when a report is embedded in job-result metadata."""

_NonEmptyString: TypeAlias = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _ReportBase(BaseModel):
    """Validation shared by public report variants."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    media_type: _NonEmptyString
    filename: _NonEmptyString

    @field_validator("filename")
    @classmethod
    def _validate_filename(cls, filename: str) -> str:
        if (
            filename in {".", ".."}
            or PurePosixPath(filename).name != filename
            or PureWindowsPath(filename).name != filename
        ):
            raise ValueError("filename must be a basename without directory components")
        return filename


class InlineTextReport(_ReportBase):
    """Small, self-contained textual report returned in capability job metadata."""

    kind: Literal["inline_text"] = "inline_text"
    content: str

    @field_validator("content")
    @classmethod
    def _validate_content(cls, content: str) -> str:
        if not content.strip():
            raise ValueError("inline report content must not be empty")

        return content


class ArtifactReport(_ReportBase):
    """Producer-published report stored durably outside capability job metadata."""

    kind: Literal["artifact"] = "artifact"
    uri: _NonEmptyString


CapabilityReport: TypeAlias = Annotated[
    InlineTextReport | ArtifactReport,
    Field(discriminator="kind"),
]


class InlineTextReportPayload(TypedDict):
    """JSON-compatible serialized form of :class:`InlineTextReport`."""

    kind: Literal["inline_text"]
    media_type: str
    content: str
    filename: str


class ArtifactReportPayload(TypedDict):
    """JSON-compatible serialized form of :class:`ArtifactReport`."""

    kind: Literal["artifact"]
    media_type: str
    uri: str
    filename: str


CapabilityReportPayload: TypeAlias = InlineTextReportPayload | ArtifactReportPayload

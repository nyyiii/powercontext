# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Captured row Sources for offline CSV documents."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document and its row identity configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    document_id: str
    primary_key: str
    content: bytes

    @model_validator(mode="after")
    def validate_identity_configuration(self) -> CSVDocumentInput:
        _validate_document_id(self.document_id)
        if not self.primary_key:
            raise ValueError("primary_key must not be empty")  # noqa: TRY003
        return self


class CSVRowInput(BaseModel):
    """One validated CSV row with values in the original header order."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_row(self) -> CSVRowInput:
        _validate_document_id(self.document_id)
        _validate_columns(self.columns)
        if not self.primary_key:
            raise ValueError("primary_key must not be empty")  # noqa: TRY003
        if self.primary_key not in self.columns:
            raise ValueError("primary_key must exactly match a column")  # noqa: TRY003
        if len(self.values) != len(self.columns):
            raise ValueError("row width must match the header width")  # noqa: TRY003
        if not self.primary_key_value.strip():
            raise ValueError("primary_key_value must not be blank")  # noqa: TRY003
        key_index = self.columns.index(self.primary_key)
        if self.values[key_index] != self.primary_key_value:
            raise ValueError("primary_key_value must match the row value")  # noqa: TRY003
        csv_row_source_id(self.document_id, self.primary_key_value)
        return self


class CSVRowSource(Source):
    """Captured evidence for one exact row of a CSV document."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Return the stable, delimiter-safe identity for one CSV row."""

    _validate_document_id(document_id)
    if not isinstance(primary_key_value, str) or not primary_key_value.strip():
        raise ValueError("primary_key_value must be a non-blank string")  # noqa: TRY003
    identity = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"CSV row source identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def parse_csv_document(value: CSVDocumentInput, /) -> tuple[CSVRowInput, ...]:
    """Parse and atomically validate all data rows in an offline CSV document."""

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be valid UTF-8") from error  # noqa: TRY003

    try:
        records = list(csv.reader(io.StringIO(text, newline=""), delimiter=",", quotechar='"', strict=True))
    except csv.Error as error:
        raise ValueError("CSV content is malformed") from error  # noqa: TRY003

    if not records:
        raise ValueError("CSV document must contain a header")  # noqa: TRY003

    columns = tuple(records[0])
    _validate_columns(columns)
    if value.primary_key not in columns:
        raise ValueError("primary_key must exactly match a column")  # noqa: TRY003

    primary_key_index = columns.index(value.primary_key)
    primary_key_values: set[str] = set()
    rows: list[CSVRowInput] = []
    for record in records[1:]:
        if not record:
            continue
        if len(record) != len(columns):
            raise ValueError("row width must match the header width")  # noqa: TRY003
        primary_key_value = record[primary_key_index]
        if not primary_key_value.strip():
            raise ValueError("primary key values must not be blank")  # noqa: TRY003
        if primary_key_value in primary_key_values:
            raise ValueError("primary key values must be unique within a document")  # noqa: TRY003
        primary_key_values.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=value.document_id,
                primary_key=value.primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(rows)


def _validate_document_id(document_id: object) -> None:
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003


def _validate_columns(columns: tuple[str, ...]) -> None:
    if not columns:
        raise ValueError("CSV header must not be empty")  # noqa: TRY003
    if any(not column.strip() for column in columns):
        raise ValueError("CSV header fields must not be blank")  # noqa: TRY003
    if len(set(columns)) != len(columns):
        raise ValueError("CSV header fields must be unique")  # noqa: TRY003


class CSVRowSourceAdapter:
    """Resolve and read captured CSV row Sources."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        identity = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=identity,
            source_id=identity,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Project a CSV row as canonical text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_ADAPTER",
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]

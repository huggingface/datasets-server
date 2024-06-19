# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from typing import Any, Optional, TypedDict

Row = dict[str, Any]


@dataclass
class RowsContent:
    rows: list[Row]
    all_fetched: bool
    truncated_columns: list[str]


class Status(str, enum.Enum):
    WAITING = "waiting"
    STARTED = "started"


class Priority(str, enum.Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class WorkerSize(str, enum.Enum):
    heavy = "heavy"
    medium = "medium"
    light = "light"


class JobParams(TypedDict):
    dataset: str
    revision: str
    config: Optional[str]
    split: Optional[str]


class JobInfo(TypedDict):
    job_id: str
    type: str
    params: JobParams
    priority: Priority
    difficulty: int
    started_at: Optional[datetime]


class FlatJobInfo(TypedDict):
    job_id: str
    type: str
    dataset: str
    revision: str
    config: Optional[str]
    split: Optional[str]
    priority: str
    status: str
    difficulty: int
    created_at: datetime


class JobOutput(TypedDict):
    content: Mapping[str, Any]
    http_status: HTTPStatus
    error_code: Optional[str]
    details: Optional[Mapping[str, Any]]
    progress: Optional[float]


class JobResult(TypedDict):
    job_info: JobInfo
    job_runner_version: int
    is_success: bool
    output: Optional[JobOutput]


class SplitHubFile(TypedDict):
    dataset: str
    config: str
    split: str
    url: str
    filename: str
    size: int


class RowItem(TypedDict):
    row_idx: int
    row: Row
    truncated_cells: list[str]


class FeatureItem(TypedDict):
    feature_idx: int
    name: str
    type: dict[str, Any]


class PaginatedResponse(TypedDict):
    features: list[FeatureItem]
    rows: list[RowItem]
    num_rows_total: int
    num_rows_per_page: int
    partial: bool


class DatasetItem(TypedDict):
    dataset: str


class ConfigItem(DatasetItem):
    config: Optional[str]


class SplitItem(ConfigItem):
    split: Optional[str]


class FullConfigItem(DatasetItem):
    config: str


class FullSplitItem(FullConfigItem):
    split: str


class SplitFirstRowsResponse(FullSplitItem):
    features: list[FeatureItem]
    rows: list[RowItem]
    truncated: bool

# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import pytest
from datasets import Features, Image, Sequence, Value
from libcommon.dtos import Priority
from libcommon.exceptions import PreviousStepFormatError
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response

from worker.config import AppConfig
from worker.dtos import DatasetModalitiesResponse
from worker.job_runners.dataset.modalities import DatasetModalitiesJobRunner
from worker.resources import LibrariesResource

from ..utils import REVISION_NAME, UpstreamResponse


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig], DatasetModalitiesJobRunner]

TEXT_DATASET = "text-dataset"
TABULAR_DATASET = "tabular-dataset"
IMAGE_TEXT_DATASET = "image-text-dataset"
IMAGE_DATASET = "image-dataset"
TIME_SERIES_DATASET = "time-series-dataset"
ERROR_DATASET = "error-dataset"

text_features = Features({"conversations": [{"from": Value("string"), "value": Value("string")}]})
image_text_features = Features({"image": Image(), "caption": Value("string")})
tabular_features = Features({"col1": Value("int8"), "col2": Value("float32")})
not_tabular_features_1 = Features({"col1": Value("int8"), "col2": Value("float32"), "image": Image()})
not_tabular_features_2 = Features({"col1": Value("int8"), "col2": Value("string")})
time_series_features = Features({"window": Sequence(Value("float32")), "target": Value("float32")})

UPSTREAM_RESPONSE_INFO_TEXT: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=TEXT_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={
        "dataset_info": {"default": {"config_name": "default", "features": text_features.to_dict()}},
        "partial": False,
    },
    progress=1.0,
)
UPSTREAM_RESPONSE_INFO_IMAGE_TEXT: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=IMAGE_TEXT_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={
        "dataset_info": {"default": {"config_name": "default", "features": image_text_features.to_dict()}},
        "partial": False,
    },
    progress=1.0,
)
UPSTREAM_RESPONSE_INFO_TABULAR: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=TABULAR_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={
        "dataset_info": {"default": {"config_name": "default", "features": tabular_features.to_dict()}},
        "partial": False,
    },
    progress=1.0,
)
UPSTREAM_RESPONSE_INFO_NOT_TABULAR_1: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=IMAGE_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={
        "dataset_info": {"default": {"config_name": "default", "features": not_tabular_features_1.to_dict()}},
        "partial": False,
    },
    progress=1.0,
)
UPSTREAM_RESPONSE_INFO_NOT_TABULAR_2: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=TEXT_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={
        "dataset_info": {"default": {"config_name": "default", "features": not_tabular_features_2.to_dict()}},
        "partial": False,
    },
    progress=1.0,
)
UPSTREAM_RESPONSE_INFO_TIME_SERIES: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=TIME_SERIES_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={
        "dataset_info": {"default": {"config_name": "default", "features": time_series_features.to_dict()}},
        "partial": False,
    },
    progress=1.0,
)
UPSTREAM_RESPONSE_INFO_ERROR: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=ERROR_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
    progress=0.0,
)
UPSTREAM_RESPONSE_INFO_MALFORMED: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=ERROR_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    # The content is missing the "dataset_info" key
    content={"bad": "content"},
    progress=0.0,
)

UPSTREAM_RESPONSE_FILETYPES_TEXT: UpstreamResponse = UpstreamResponse(
    kind="dataset-filetypes",
    dataset=TEXT_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={
        "filetypes": [
            {"extension": ".txt", "count": 1, "compressed_in": ".gz"},
            {"extension": ".gz", "count": 1},
        ],
        "partial": False,
    },
    progress=1.0,
)
UPSTREAM_RESPONSE_FILETYPES_ALL: UpstreamResponse = UpstreamResponse(
    kind="dataset-filetypes",
    dataset=TEXT_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={
        "filetypes": [
            {"extension": ".txt", "count": 1, "compressed_in": ".gz"},
            {"extension": ".avi", "count": 1},
            {"extension": ".geoparquet", "count": 1, "archived_in": ".zip"},
            {"extension": ".gz", "count": 1},
            {"extension": ".zip", "count": 1},
            {"extension": ".jpg", "count": 1},
            {"extension": ".wav", "count": 1},
            {"extension": ".gltf", "count": 1},
        ],
        "partial": False,
    },
    progress=1.0,
)

EXPECTED_TEXT: tuple[DatasetModalitiesResponse, float] = (
    {"modalities": ["text"]},
    1.0,
)
EXPECTED_TABULAR: tuple[DatasetModalitiesResponse, float] = (
    {"modalities": ["tabular"]},
    1.0,
)
EXPECTED_IMAGE: tuple[DatasetModalitiesResponse, float] = (
    {"modalities": ["image"]},
    1.0,
)
EXPECTED_IMAGE_TEXT: tuple[DatasetModalitiesResponse, float] = (
    {"modalities": ["image", "text"]},
    1.0,
)
EXPECTED_ALL_MODALITIES: tuple[DatasetModalitiesResponse, float] = (
    {
        "modalities": [
            "3d",
            "audio",
            "geospatial",
            "image",
            "text",
            "video",
        ]
    },
    1.0,
)
EXPECTED_EMPTY: tuple[DatasetModalitiesResponse, float] = (
    {"modalities": []},
    1.0,
)
EXPECTED_TIME_SERIES: tuple[DatasetModalitiesResponse, float] = (
    {"modalities": ["timeseries"]},
    1.0,
)


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetModalitiesJobRunner:
        return DatasetModalitiesJobRunner(
            job_info={
                "type": DatasetModalitiesJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "config": None,
                    "split": None,
                    "revision": REVISION_NAME,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 20,
                "started_at": None,
            },
            app_config=app_config,
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,upstream_responses,expected",
    [
        (
            TEXT_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_TEXT,
            ],
            EXPECTED_TEXT,
        ),
        (
            IMAGE_TEXT_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_IMAGE_TEXT,
            ],
            EXPECTED_IMAGE_TEXT,
        ),
        (
            TEXT_DATASET,
            [
                UPSTREAM_RESPONSE_FILETYPES_TEXT,
            ],
            EXPECTED_TEXT,
        ),
        (
            TEXT_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_TEXT,
                UPSTREAM_RESPONSE_FILETYPES_TEXT,
            ],
            EXPECTED_TEXT,
        ),
        (
            TEXT_DATASET,
            [
                UPSTREAM_RESPONSE_FILETYPES_ALL,
            ],
            EXPECTED_ALL_MODALITIES,
        ),
        (
            ERROR_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_ERROR,
            ],
            EXPECTED_EMPTY,
        ),
        (
            TABULAR_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_TABULAR,
            ],
            EXPECTED_TABULAR,
        ),
        (
            IMAGE_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_NOT_TABULAR_1,
            ],
            EXPECTED_IMAGE,
        ),
        (
            TEXT_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_NOT_TABULAR_2,
            ],
            EXPECTED_TEXT,
        ),
        (
            TIME_SERIES_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_TIME_SERIES,
            ],
            EXPECTED_TIME_SERIES,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    upstream_responses: list[UpstreamResponse],
    expected: Any,
) -> None:
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config)
    compute_result = job_runner.compute()
    assert compute_result.content == expected[0]
    assert compute_result.progress == expected[1]


@pytest.mark.parametrize(
    "dataset,upstream_responses,expectation",
    [
        (
            ERROR_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_MALFORMED,
            ],
            pytest.raises(PreviousStepFormatError),
        )
    ],
)
def test_compute_error(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    upstream_responses: list[UpstreamResponse],
    expectation: Any,
) -> None:
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config)
    with expectation:
        job_runner.compute()

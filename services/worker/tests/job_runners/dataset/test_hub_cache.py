# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import pytest
from libcommon.dtos import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedArtifactError, CachedArtifactNotFoundError, upsert_response

from worker.config import AppConfig
from worker.dtos import DatasetFormat, DatasetHubCacheResponse, DatasetLibrary, DatasetModality, DatasetTag
from worker.job_runners.dataset.hub_cache import DatasetHubCacheJobRunner

from ..utils import REVISION_NAME, UpstreamResponse


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig], DatasetHubCacheJobRunner]

DATASET = "dataset"
TAG: DatasetTag = "croissant"
LIBRARY: DatasetLibrary = "mlcroissant"
FORMAT: DatasetFormat = "json"
MODALITY: DatasetModality = "image"
UPSTREAM_RESPONSE_IS_VALID_OK: UpstreamResponse = UpstreamResponse(
    kind="dataset-is-valid",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={"preview": True, "viewer": False, "search": True},
    progress=0.5,
)
UPSTREAM_RESPONSE_IS_VALID_ERROR: UpstreamResponse = UpstreamResponse(
    kind="dataset-is-valid",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
    progress=0.0,
)
UPSTREAM_RESPONSE_SIZE_OK: UpstreamResponse = UpstreamResponse(
    kind="dataset-size",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={"size": {"dataset": {"num_rows": 1000}}, "partial": False},
    progress=0.2,
)
UPSTREAM_RESPONSE_SIZE_ERROR: UpstreamResponse = UpstreamResponse(
    kind="dataset-size",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
    progress=0.0,
)
UPSTREAM_RESPONSE_SIZE_NO_PROGRESS: UpstreamResponse = UpstreamResponse(
    kind="dataset-size",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={"size": {"dataset": {"num_rows": 1000}}, "partial": True},
    progress=None,
)
UPSTREAM_RESPONSE_COMPATIBLE_LIBRARIES_OK: UpstreamResponse = UpstreamResponse(
    kind="dataset-compatible-libraries",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={"tags": [TAG], "libraries": [{"library": LIBRARY}], "formats": [FORMAT]},
    progress=1.0,
)
UPSTREAM_RESPONSE_COMPATIBLE_LIBRARIES_ERROR: UpstreamResponse = UpstreamResponse(
    kind="dataset-compatible-libraries",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
    progress=0.0,
)
UPSTREAM_RESPONSE_MODALITIES_OK: UpstreamResponse = UpstreamResponse(
    kind="dataset-modalities",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={"tags": [TAG], "modalities": [MODALITY]},
    progress=1.0,
)
UPSTREAM_RESPONSE_MODALITIES_ERROR: UpstreamResponse = UpstreamResponse(
    kind="dataset-modalities",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
    progress=0.0,
)
EXPECTED_OK: tuple[DatasetHubCacheResponse, float] = (
    {
        "viewer": False,
        "preview": True,
        "partial": False,
        "num_rows": 1000,
        "tags": [],
        "libraries": [],
        "modalities": [],
        "formats": [],
    },
    0.2,
)
EXPECTED_NO_PROGRESS: tuple[DatasetHubCacheResponse, float] = (
    {
        "viewer": False,
        "preview": True,
        "partial": True,
        "num_rows": 1000,
        "tags": [],
        "libraries": [],
        "modalities": [],
        "formats": [],
    },
    0.5,
)
EXPECTED_NO_SIZE: tuple[DatasetHubCacheResponse, float] = (
    {
        "viewer": False,
        "preview": True,
        "partial": False,
        "num_rows": None,
        "tags": [],
        "libraries": [],
        "modalities": [],
        "formats": [],
    },
    0.0,
)
EXPECTED_OK_WITH_LIBRARIES_AND_FORMATS: tuple[DatasetHubCacheResponse, float] = (
    {
        "viewer": False,
        "preview": True,
        "partial": True,
        "num_rows": 1000,
        "tags": [TAG],
        "libraries": [LIBRARY],
        "modalities": [],
        "formats": [FORMAT],
    },
    0.5,
)
EXPECTED_OK_WITH_MODALITIES: tuple[DatasetHubCacheResponse, float] = (
    {
        "viewer": False,
        "preview": True,
        "partial": True,
        "num_rows": 1000,
        "tags": [],
        "libraries": [],
        "modalities": [MODALITY],
        "formats": [],
    },
    0.5,
)


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetHubCacheJobRunner:
        return DatasetHubCacheJobRunner(
            job_info={
                "type": DatasetHubCacheJobRunner.get_job_type(),
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
    "upstream_responses,expected",
    [
        (
            [
                UPSTREAM_RESPONSE_IS_VALID_OK,
                UPSTREAM_RESPONSE_SIZE_OK,
            ],
            EXPECTED_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_IS_VALID_OK,
                UPSTREAM_RESPONSE_SIZE_NO_PROGRESS,
            ],
            EXPECTED_NO_PROGRESS,
        ),
        (
            [
                UPSTREAM_RESPONSE_IS_VALID_OK,
                UPSTREAM_RESPONSE_SIZE_NO_PROGRESS,
                UPSTREAM_RESPONSE_COMPATIBLE_LIBRARIES_OK,
            ],
            EXPECTED_OK_WITH_LIBRARIES_AND_FORMATS,
        ),
        (
            [
                UPSTREAM_RESPONSE_IS_VALID_OK,
                UPSTREAM_RESPONSE_SIZE_NO_PROGRESS,
                UPSTREAM_RESPONSE_MODALITIES_OK,
            ],
            EXPECTED_OK_WITH_MODALITIES,
        ),
        (
            [
                UPSTREAM_RESPONSE_IS_VALID_OK,
                UPSTREAM_RESPONSE_SIZE_ERROR,
            ],
            EXPECTED_NO_SIZE,
        ),
        (
            [
                UPSTREAM_RESPONSE_IS_VALID_OK,
                UPSTREAM_RESPONSE_SIZE_OK,
                UPSTREAM_RESPONSE_COMPATIBLE_LIBRARIES_ERROR,
            ],
            EXPECTED_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_IS_VALID_OK,
                UPSTREAM_RESPONSE_SIZE_OK,
                UPSTREAM_RESPONSE_MODALITIES_ERROR,
            ],
            EXPECTED_OK,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    upstream_responses: list[UpstreamResponse],
    expected: Any,
) -> None:
    dataset = DATASET
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config)
    compute_result = job_runner.compute()
    assert compute_result.content == expected[0]
    assert compute_result.progress == expected[1]


@pytest.mark.parametrize(
    "upstream_responses,expectation",
    [
        (
            [],
            pytest.raises(CachedArtifactNotFoundError),
        ),
        (
            [
                UPSTREAM_RESPONSE_IS_VALID_ERROR,
                UPSTREAM_RESPONSE_SIZE_OK,
            ],
            pytest.raises(CachedArtifactError),
        ),
    ],
)
def test_compute_error(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    upstream_responses: list[UpstreamResponse],
    expectation: Any,
) -> None:
    dataset = DATASET
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config)
    with expectation:
        job_runner.compute()

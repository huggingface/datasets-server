# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import pytest
from libcommon.dtos import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response

from worker.config import AppConfig
from worker.job_runners.dataset.is_valid import DatasetIsValidJobRunner

from ..utils import REVISION_NAME, UpstreamResponse


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig], DatasetIsValidJobRunner]

DATASET = "dataset"
CONFIG_1 = "config1"
CONFIG_2 = "config2"

UPSTREAM_RESPONSE_CONFIG_NAMES: UpstreamResponse = UpstreamResponse(
    kind="dataset-config-names",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={
        "config_names": [
            {"dataset": DATASET, "config": CONFIG_1},
            {"dataset": DATASET, "config": CONFIG_2},
        ]
    },
)
UPSTREAM_RESPONSE_CONFIG_1_OK: UpstreamResponse = UpstreamResponse(
    kind="config-is-valid",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    config=CONFIG_1,
    http_status=HTTPStatus.OK,
    content={"viewer": True, "preview": True, "search": True, "filter": True, "statistics": True},
)
UPSTREAM_RESPONSE_CONFIG_1_OK_VIEWER: UpstreamResponse = UpstreamResponse(
    kind="config-is-valid",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    config=CONFIG_1,
    http_status=HTTPStatus.OK,
    content={"viewer": True, "preview": False, "search": False, "filter": False, "statistics": False},
)
UPSTREAM_RESPONSE_CONFIG_2_OK_SEARCH: UpstreamResponse = UpstreamResponse(
    kind="config-is-valid",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    config=CONFIG_2,
    http_status=HTTPStatus.OK,
    content={"viewer": False, "preview": False, "search": True, "filter": True, "statistics": False},
)
UPSTREAM_RESPONSE_CONFIG_2_OK: UpstreamResponse = UpstreamResponse(
    kind="config-is-valid",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    config=CONFIG_2,
    http_status=HTTPStatus.OK,
    content={"viewer": True, "preview": True, "search": True, "filter": True, "statistics": True},
)
UPSTREAM_RESPONSE_CONFIG_1_ERROR: UpstreamResponse = UpstreamResponse(
    kind="config-is-valid",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    config=CONFIG_1,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
)
UPSTREAM_RESPONSE_CONFIG_2_ERROR: UpstreamResponse = UpstreamResponse(
    kind="config-is-valid",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    config=CONFIG_2,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
)
EXPECTED_COMPLETED_ALL_FALSE = (
    {"viewer": False, "preview": False, "search": False, "filter": False, "statistics": False},
    1.0,
)
EXPECTED_ALL_MIXED = (
    {"viewer": True, "preview": False, "search": True, "filter": True, "statistics": False},
    1.0,
)
EXPECTED_COMPLETED_ALL_TRUE = (
    {"viewer": True, "preview": True, "search": True, "filter": True, "statistics": True},
    1.0,
)
EXPECTED_PENDING_ALL_TRUE = (
    {"viewer": True, "preview": True, "search": True, "filter": True, "statistics": True},
    0.5,
)
EXPECTED_PENDING_ALL_FALSE = (
    {"viewer": False, "preview": False, "search": False, "filter": False, "statistics": False},
    0.0,
)


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetIsValidJobRunner:
        return DatasetIsValidJobRunner(
            job_info={
                "type": DatasetIsValidJobRunner.get_job_type(),
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
                UPSTREAM_RESPONSE_CONFIG_1_OK,
                UPSTREAM_RESPONSE_CONFIG_2_OK,
            ],
            EXPECTED_COMPLETED_ALL_TRUE,
        ),
        (
            [
                UPSTREAM_RESPONSE_CONFIG_1_OK,
            ],
            EXPECTED_PENDING_ALL_TRUE,
        ),
        (
            [
                UPSTREAM_RESPONSE_CONFIG_1_ERROR,
                UPSTREAM_RESPONSE_CONFIG_2_ERROR,
            ],
            EXPECTED_COMPLETED_ALL_FALSE,
        ),
        ([UPSTREAM_RESPONSE_CONFIG_1_OK_VIEWER, UPSTREAM_RESPONSE_CONFIG_2_OK_SEARCH], EXPECTED_ALL_MIXED),
        (
            [],
            EXPECTED_PENDING_ALL_FALSE,
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
    upsert_response(**UPSTREAM_RESPONSE_CONFIG_NAMES)
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config)
    compute_result = job_runner.compute()
    assert compute_result.content == expected[0]
    assert compute_result.progress == expected[1]

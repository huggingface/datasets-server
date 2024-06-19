# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import pytest
from libcommon.dtos import Priority
from libcommon.exceptions import PreviousStepFormatError
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import (
    CachedArtifactError,
    CachedArtifactNotFoundError,
    upsert_response,
)

from worker.config import AppConfig
from worker.dtos import PreviousJob
from worker.job_runners.dataset.info import DatasetInfoJobRunner

from ..config.test_info import CONFIG_INFO_1, CONFIG_INFO_2, DATASET_INFO_OK
from ..utils import REVISION_NAME, UpstreamResponse


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig], DatasetInfoJobRunner]


UPSTREAM_RESPONSE_CONFIG_NAMES: UpstreamResponse = UpstreamResponse(
    kind="dataset-config-names",
    dataset="dataset_ok",
    dataset_git_revision=REVISION_NAME,
    config=None,
    http_status=HTTPStatus.OK,
    content={
        "config_names": [
            {"dataset": "dataset_ok", "config": "config_1"},
            {"dataset": "dataset_ok", "config": "config_2"},
        ],
    },
)

UPSTREAM_RESPONSE_CONFIG_INFO_1: UpstreamResponse = UpstreamResponse(
    kind="config-info",
    dataset="dataset_ok",
    dataset_git_revision=REVISION_NAME,
    config="config_1",
    http_status=HTTPStatus.OK,
    content={"dataset_info": CONFIG_INFO_1, "partial": False},
)

UPSTREAM_RESPONSE_CONFIG_INFO_2: UpstreamResponse = UpstreamResponse(
    kind="config-info",
    dataset="dataset_ok",
    dataset_git_revision=REVISION_NAME,
    config="config_2",
    http_status=HTTPStatus.OK,
    content={"dataset_info": CONFIG_INFO_2, "partial": False},
)

EXPECTED_OK = (
    {
        "dataset_info": DATASET_INFO_OK,
        "pending": [],
        "failed": [],
        "partial": False,
    },
    1.0,
)

EXPECTED_PARTIAL_PENDING = (
    {
        "dataset_info": {
            "config_1": CONFIG_INFO_1,
        },
        "pending": [
            PreviousJob(
                kind="config-info",
                dataset="dataset_ok",
                config="config_2",
                split=None,
            )
        ],
        "failed": [],
        "partial": False,
    },
    0.5,
)

EXPECTED_PARTIAL_FAILED = (
    {
        "dataset_info": {
            "config_1": CONFIG_INFO_1,
        },
        "pending": [],
        "failed": [
            PreviousJob(
                kind="config-info",
                dataset="dataset_ok",
                config="config_2",
                split=None,
            )
        ],
        "partial": False,
    },
    1.0,
)


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetInfoJobRunner:
        return DatasetInfoJobRunner(
            job_info={
                "type": DatasetInfoJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": REVISION_NAME,
                    "config": None,
                    "split": None,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 50,
                "started_at": None,
            },
            app_config=app_config,
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,upstream_responses,expected_error_code,expected,should_raise",
    [
        (
            "dataset_ok",
            [
                UPSTREAM_RESPONSE_CONFIG_NAMES,
                UPSTREAM_RESPONSE_CONFIG_INFO_1,
                UPSTREAM_RESPONSE_CONFIG_INFO_2,
            ],
            None,
            EXPECTED_OK,
            False,
        ),
        (
            "dataset_ok",
            [UPSTREAM_RESPONSE_CONFIG_NAMES, UPSTREAM_RESPONSE_CONFIG_INFO_1],
            None,
            EXPECTED_PARTIAL_PENDING,
            False,
        ),
        (
            "dataset_ok",
            [
                UPSTREAM_RESPONSE_CONFIG_NAMES,
                UPSTREAM_RESPONSE_CONFIG_INFO_1,
                UpstreamResponse(
                    kind="config-info",
                    dataset="dataset_ok",
                    dataset_git_revision=REVISION_NAME,
                    config="config_2",
                    http_status=HTTPStatus.NOT_FOUND,
                    content={"error": "error"},
                ),
            ],
            None,
            EXPECTED_PARTIAL_FAILED,
            False,
        ),
        (
            "status_error",
            [
                UpstreamResponse(
                    kind="dataset-config-names",
                    dataset="status_error",
                    dataset_git_revision=REVISION_NAME,
                    config=None,
                    http_status=HTTPStatus.NOT_FOUND,
                    content={"error": "error"},
                )
            ],
            CachedArtifactError.__name__,
            None,
            True,
        ),
        (
            "format_error",
            [
                UpstreamResponse(
                    kind="dataset-config-names",
                    dataset="format_error",
                    dataset_git_revision=REVISION_NAME,
                    config=None,
                    http_status=HTTPStatus.OK,
                    content={"not_dataset_info": "wrong_format"},
                )
            ],
            PreviousStepFormatError.__name__,
            None,
            True,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    upstream_responses: list[UpstreamResponse],
    expected_error_code: str,
    expected: Any,
    should_raise: bool,
) -> None:
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config)
    if should_raise:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.typename == expected_error_code
    else:
        compute_result = job_runner.compute()
        assert compute_result.content == expected[0]
        assert compute_result.progress == expected[1]


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "doesnotexist"
    job_runner = get_job_runner(dataset, app_config)
    with pytest.raises(CachedArtifactNotFoundError):
        job_runner.compute()

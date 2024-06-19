# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

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
from worker.job_runners.dataset.duckdb_index_size import DatasetDuckdbIndexSizeJobRunner

from ..utils import REVISION_NAME, UpstreamResponse


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig], DatasetDuckdbIndexSizeJobRunner]


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetDuckdbIndexSizeJobRunner:
        return DatasetDuckdbIndexSizeJobRunner(
            job_info={
                "type": DatasetDuckdbIndexSizeJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
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
    "dataset,upstream_responses,expected_error_code,expected_content,should_raise",
    [
        (
            "dataset_ok",
            [
                UpstreamResponse(
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
                ),
                UpstreamResponse(
                    kind="config-duckdb-index-size",
                    dataset="dataset_ok",
                    dataset_git_revision=REVISION_NAME,
                    config="config_1",
                    http_status=HTTPStatus.OK,
                    content={
                        "size": {
                            "config": {
                                "dataset": "dataset_ok",
                                "config": "config_1",
                                "has_fts": True,
                                "num_rows": 7,
                                "num_bytes": 56,
                            },
                            "splits": [
                                {
                                    "dataset": "dataset_ok",
                                    "config": "config_1",
                                    "split": "train",
                                    "has_fts": True,
                                    "num_rows": 5,
                                    "num_bytes": 40,
                                },
                                {
                                    "dataset": "dataset_ok",
                                    "config": "config_1",
                                    "split": "test",
                                    "has_fts": True,
                                    "num_rows": 2,
                                    "num_bytes": 16,
                                },
                            ],
                        },
                        "partial": False,
                    },
                ),
                UpstreamResponse(
                    kind="config-duckdb-index-size",
                    dataset="dataset_ok",
                    dataset_git_revision=REVISION_NAME,
                    config="config_2",
                    http_status=HTTPStatus.OK,
                    content={
                        "size": {
                            "config": {
                                "dataset": "dataset_ok",
                                "config": "config_2",
                                "has_fts": True,
                                "num_rows": 5,
                                "num_bytes": 40,
                            },
                            "splits": [
                                {
                                    "dataset": "dataset_ok",
                                    "config": "config_2",
                                    "split": "train",
                                    "has_fts": True,
                                    "num_rows": 5,
                                    "num_bytes": 40,
                                },
                            ],
                        },
                        "partial": False,
                    },
                ),
            ],
            None,
            {
                "size": {
                    "dataset": {
                        "dataset": "dataset_ok",
                        "has_fts": True,
                        "num_rows": 12,
                        "num_bytes": 96,
                    },
                    "configs": [
                        {
                            "dataset": "dataset_ok",
                            "config": "config_1",
                            "has_fts": True,
                            "num_rows": 7,
                            "num_bytes": 56,
                        },
                        {
                            "dataset": "dataset_ok",
                            "config": "config_2",
                            "has_fts": True,
                            "num_rows": 5,
                            "num_bytes": 40,
                        },
                    ],
                    "splits": [
                        {
                            "dataset": "dataset_ok",
                            "config": "config_1",
                            "split": "train",
                            "has_fts": True,
                            "num_rows": 5,
                            "num_bytes": 40,
                        },
                        {
                            "dataset": "dataset_ok",
                            "config": "config_1",
                            "split": "test",
                            "has_fts": True,
                            "num_rows": 2,
                            "num_bytes": 16,
                        },
                        {
                            "dataset": "dataset_ok",
                            "config": "config_2",
                            "split": "train",
                            "has_fts": True,
                            "num_rows": 5,
                            "num_bytes": 40,
                        },
                    ],
                },
                "failed": [],
                "pending": [],
                "partial": False,
            },
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
    expected_content: Any,
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
        assert job_runner.compute().content == expected_content


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "doesnotexist"
    job_runner = get_job_runner(dataset, app_config)
    with pytest.raises(CachedArtifactNotFoundError):
        job_runner.compute()

# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import pytest
from libcommon.dtos import Priority, SplitHubFile
from libcommon.exceptions import PreviousStepFormatError
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import (
    CachedArtifactError,
    CachedArtifactNotFoundError,
    upsert_response,
)

from worker.config import AppConfig
from worker.dtos import ConfigParquetResponse, DatasetParquetResponse
from worker.job_runners.dataset.parquet import DatasetParquetJobRunner

from ..utils import REVISION_NAME, UpstreamResponse


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig], DatasetParquetJobRunner]


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetParquetJobRunner:
        return DatasetParquetJobRunner(
            job_info={
                "type": DatasetParquetJobRunner.get_job_type(),
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
    "dataset,upstream_responses,expected_error_code,expected_content,should_raise",
    [
        (
            "ok",
            [
                UpstreamResponse(
                    kind="dataset-config-names",
                    dataset="ok",
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
                    kind="config-parquet",
                    dataset="ok",
                    dataset_git_revision=REVISION_NAME,
                    config="config_1",
                    http_status=HTTPStatus.OK,
                    content=ConfigParquetResponse(
                        parquet_files=[
                            SplitHubFile(
                                dataset="ok",
                                config="config_1",
                                split="train",
                                url="url1",
                                filename="filename1",
                                size=0,
                            ),
                        ],
                        partial=False,
                        features=None,
                    ),
                ),
                UpstreamResponse(
                    kind="config-parquet",
                    dataset="ok",
                    dataset_git_revision=REVISION_NAME,
                    config="config_2",
                    http_status=HTTPStatus.OK,
                    content=ConfigParquetResponse(
                        parquet_files=[
                            SplitHubFile(
                                dataset="ok",
                                config="config_2",
                                split="train",
                                url="url2",
                                filename="filename2",
                                size=0,
                            ),
                        ],
                        partial=False,
                        features=None,
                    ),
                ),
            ],
            None,
            DatasetParquetResponse(
                parquet_files=[
                    SplitHubFile(
                        dataset="ok", config="config_1", split="train", url="url1", filename="filename1", size=0
                    ),
                    SplitHubFile(
                        dataset="ok", config="config_2", split="train", url="url2", filename="filename2", size=0
                    ),
                ],
                pending=[],
                failed=[],
                partial=False,
            ),
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
                    content={"not_parquet_files": "wrong_format"},
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

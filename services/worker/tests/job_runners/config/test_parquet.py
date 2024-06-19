# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import pytest
from datasets import Features, Value
from libcommon.dtos import Priority, SplitHubFile
from libcommon.exceptions import PreviousStepFormatError
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import (
    CachedArtifactError,
    CachedArtifactNotFoundError,
    upsert_response,
)

from worker.config import AppConfig
from worker.dtos import ConfigParquetAndInfoResponse, ConfigParquetResponse
from worker.job_runners.config.parquet import ConfigParquetJobRunner

from ..utils import REVISION_NAME


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, str, AppConfig], ConfigParquetJobRunner]


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigParquetJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        return ConfigParquetJobRunner(
            job_info={
                "type": ConfigParquetJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": REVISION_NAME,
                    "config": config,
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
    "dataset,config,upstream_status,upstream_content,expected_error_code,expected_content,should_raise",
    [
        (
            "ok",
            "config_1",
            HTTPStatus.OK,
            ConfigParquetAndInfoResponse(
                parquet_files=[
                    SplitHubFile(
                        dataset="ok", config="config_1", split="train", url="url1", filename="filename1", size=0
                    ),
                    SplitHubFile(
                        dataset="ok", config="config_1", split="train", url="url2", filename="filename2", size=0
                    ),
                ],
                dataset_info={"description": "value", "dataset_size": 10},
                partial=False,
            ),
            None,
            ConfigParquetResponse(
                parquet_files=[
                    SplitHubFile(
                        dataset="ok", config="config_1", split="train", url="url1", filename="filename1", size=0
                    ),
                    SplitHubFile(
                        dataset="ok", config="config_1", split="train", url="url2", filename="filename2", size=0
                    ),
                ],
                partial=False,
                features=None,
            ),
            False,
        ),
        (
            "status_error",
            "config_1",
            HTTPStatus.NOT_FOUND,
            {"error": "error"},
            CachedArtifactError.__name__,
            None,
            True,
        ),
        (
            "format_error",
            "config_1",
            HTTPStatus.OK,
            {"not_parquet_files": "wrong_format"},
            PreviousStepFormatError.__name__,
            None,
            True,
        ),
        (
            "shards_order",
            "config_1",
            HTTPStatus.OK,
            ConfigParquetAndInfoResponse(
                parquet_files=[
                    SplitHubFile(
                        dataset="ok",
                        config="config_1",
                        split="train",
                        url="url1",
                        filename="0000.parquet",
                        size=0,
                    ),
                    SplitHubFile(
                        dataset="ok",
                        config="config_1",
                        split="train",
                        url="url2",
                        filename="0001.parquet",
                        size=0,
                    ),
                    SplitHubFile(
                        dataset="ok",
                        config="config_1",
                        split="test",
                        url="url2",
                        filename="0000.parquet",
                        size=0,
                    ),
                ],
                dataset_info={"description": "value", "dataset_size": 10},
                partial=False,
            ),
            None,
            ConfigParquetResponse(
                parquet_files=[
                    SplitHubFile(
                        dataset="ok",
                        config="config_1",
                        split="test",
                        url="url2",
                        filename="0000.parquet",
                        size=0,
                    ),
                    SplitHubFile(
                        dataset="ok",
                        config="config_1",
                        split="train",
                        url="url1",
                        filename="0000.parquet",
                        size=0,
                    ),
                    SplitHubFile(
                        dataset="ok",
                        config="config_1",
                        split="train",
                        url="url2",
                        filename="0001.parquet",
                        size=0,
                    ),
                ],
                partial=False,
                features=None,
            ),
            False,
        ),
        (
            "with_features",
            "config_1",
            HTTPStatus.OK,
            ConfigParquetAndInfoResponse(
                parquet_files=[
                    SplitHubFile(
                        dataset="with_features",
                        config="config_1",
                        split="train",
                        url="url1",
                        filename="filename1",
                        size=0,
                    ),
                    SplitHubFile(
                        dataset="with_features",
                        config="config_1",
                        split="train",
                        url="url2",
                        filename="filename2",
                        size=0,
                    ),
                ],
                dataset_info={
                    "description": "value",
                    "dataset_size": 10,
                    "features": Features({"a": Value("string")}).to_dict(),
                },
                partial=False,
            ),
            None,
            ConfigParquetResponse(
                parquet_files=[
                    SplitHubFile(
                        dataset="with_features",
                        config="config_1",
                        split="train",
                        url="url1",
                        filename="filename1",
                        size=0,
                    ),
                    SplitHubFile(
                        dataset="with_features",
                        config="config_1",
                        split="train",
                        url="url2",
                        filename="filename2",
                        size=0,
                    ),
                ],
                partial=False,
                features=Features({"a": Value("string")}).to_dict(),
            ),
            False,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    config: str,
    upstream_status: HTTPStatus,
    upstream_content: Any,
    expected_error_code: str,
    expected_content: Any,
    should_raise: bool,
) -> None:
    upsert_response(
        kind="config-parquet-and-info",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        content=upstream_content,
        http_status=upstream_status,
    )
    job_runner = get_job_runner(dataset, config, app_config)
    if should_raise:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.typename == expected_error_code
    else:
        assert job_runner.compute().content == expected_content


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = config = "doesnotexist"
    job_runner = get_job_runner(dataset, config, app_config)
    with pytest.raises(CachedArtifactNotFoundError):
        job_runner.compute()

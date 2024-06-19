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
from worker.job_runners.config.duckdb_index_size import ConfigDuckdbIndexSizeJobRunner

from ..utils import REVISION_NAME


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, str, AppConfig], ConfigDuckdbIndexSizeJobRunner]


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigDuckdbIndexSizeJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        return ConfigDuckdbIndexSizeJobRunner(
            job_info={
                "type": ConfigDuckdbIndexSizeJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
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
    "dataset,config,upstream_status,upstream_contents,expected_error_code,expected_content,should_raise",
    [
        (
            "dataset_ok",
            "config_1",
            HTTPStatus.OK,
            [
                {
                    "dataset": "dataset_ok",
                    "config": "config_1",
                    "split": "train",
                    "url": "https://foo.bar/config_1/split_1/index.duckdb",
                    "filename": "index.duckdb",
                    "size": 1234,
                    "features": {},
                    "has_fts": True,
                    "partial": False,
                    "num_rows": 5,
                    "num_bytes": 40,
                },
                {
                    "dataset": "dataset_ok",
                    "config": "config_1",
                    "split": "test",
                    "url": "https://foo.bar/config_1/split_1/index.duckdb",
                    "filename": "index.duckdb",
                    "size": 5678,
                    "features": {},
                    "has_fts": True,
                    "partial": False,
                    "num_rows": 2,
                    "num_bytes": 16,
                },
            ],
            None,
            {
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
            False,
        ),
        (
            "status_error",
            "config_1",
            HTTPStatus.INTERNAL_SERVER_ERROR,
            [{"error": "error"}],
            CachedArtifactError.__name__,
            None,
            True,
        ),
        (
            "status_not_found",
            "config_1",
            HTTPStatus.NOT_FOUND,
            [{"error": "error"}],
            CachedArtifactNotFoundError.__name__,
            None,
            True,
        ),
        (
            "format_error",
            "config_1",
            HTTPStatus.OK,
            [{"not_dataset_info": "wrong_format"}],
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
    config: str,
    upstream_status: HTTPStatus,
    upstream_contents: list[Any],
    expected_error_code: str,
    expected_content: Any,
    should_raise: bool,
) -> None:
    if upstream_status != HTTPStatus.NOT_FOUND:
        splits = [{"split": upstream_content.get("split", "train")} for upstream_content in upstream_contents]
        upsert_response(
            kind="config-split-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            config=config,
            content={"splits": splits},
            http_status=upstream_status,
        )
        upsert_response(
            kind="config-info",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            config=config,
            content={},
            http_status=upstream_status,
        )
        for upstream_content in upstream_contents:
            upsert_response(
                kind="split-duckdb-index",
                dataset=dataset,
                dataset_git_revision=REVISION_NAME,
                config=config,
                split=upstream_content.get("split", "train"),
                content=upstream_content,
                http_status=upstream_status,
            )
    job_runner = get_job_runner(dataset, config, app_config)
    if should_raise:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.typename == expected_error_code
    else:
        assert sorted(job_runner.compute().content) == sorted(expected_content)


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = config = "doesnotexist"
    job_runner = get_job_runner(dataset, config, app_config)
    with pytest.raises(CachedArtifactNotFoundError):
        job_runner.compute()

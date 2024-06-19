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
from worker.job_runners.split.is_valid import SplitIsValidJobRunner

from ..utils import REVISION_NAME, UpstreamResponse


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, str, str, AppConfig], SplitIsValidJobRunner]

DATASET = "dataset"
CONFIG = "config"
SPLIT = "split"

UPSTREAM_RESPONSE_CONFIG_SIZE: UpstreamResponse = UpstreamResponse(
    kind="config-size",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    config=CONFIG,
    http_status=HTTPStatus.OK,
    content={},
)
UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS: UpstreamResponse = UpstreamResponse(
    kind="split-first-rows",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    config=CONFIG,
    split=SPLIT,
    http_status=HTTPStatus.OK,
    content={},
)
UPSTREAM_RESPONSE_SPLIT_DUCKDB_INDEX: UpstreamResponse = UpstreamResponse(
    kind="split-duckdb-index",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    config=CONFIG,
    split=SPLIT,
    http_status=HTTPStatus.OK,
    content={"has_fts": True},
)
UPSTREAM_RESPONSE_SPLIT_DUCKDB_INDEX_ONLY_DATA: UpstreamResponse = UpstreamResponse(
    kind="split-duckdb-index",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    config=CONFIG,
    split=SPLIT,
    http_status=HTTPStatus.OK,
    content={"has_fts": False},
)
UPSTREAM_RESPONSE_SPLIT_DESCRIPTIVE_STATISTICS: UpstreamResponse = UpstreamResponse(
    kind="split-descriptive-statistics",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    config=CONFIG,
    split=SPLIT,
    http_status=HTTPStatus.OK,
    content={},
)
UPSTREAM_RESPONSE_CONFIG_SIZE_ERROR: UpstreamResponse = UpstreamResponse(
    kind="config-size",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    config=CONFIG,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
)
UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_ERROR: UpstreamResponse = UpstreamResponse(
    kind="split-first-rows",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    config=CONFIG,
    split=SPLIT,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
)
UPSTREAM_RESPONSE_SPLIT_DUCKDB_INDEX_ERROR: UpstreamResponse = UpstreamResponse(
    kind="split-duckdb-index",
    dataset=DATASET,
    dataset_git_revision=REVISION_NAME,
    config=CONFIG,
    split=SPLIT,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
)
EXPECTED_ERROR = (
    {"viewer": False, "preview": False, "search": False, "filter": False, "statistics": False},
    1.0,
)
EXPECTED_VIEWER_OK = (
    {"viewer": True, "preview": False, "search": False, "filter": False, "statistics": False},
    1.0,
)
EXPECTED_PREVIEW_OK = (
    {"viewer": False, "preview": True, "search": False, "filter": False, "statistics": False},
    1.0,
)
EXPECTED_FILTER_OK = (
    {"viewer": False, "preview": False, "search": False, "filter": True, "statistics": False},
    1.0,
)
EXPECTED_SEARCH_OK = (
    {"viewer": False, "preview": False, "search": True, "filter": True, "statistics": False},
    1.0,
)
EXPECTED_STATISTICS_OK = (
    {"viewer": False, "preview": False, "search": False, "filter": False, "statistics": True},
    1.0,
)
EXPECTED_ALL_OK = (
    {"viewer": True, "preview": True, "search": True, "filter": True, "statistics": True},
    1.0,
)


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        split: str,
        app_config: AppConfig,
    ) -> SplitIsValidJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        upsert_response(
            kind="config-split-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            config=config,
            content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
            http_status=HTTPStatus.OK,
        )

        return SplitIsValidJobRunner(
            job_info={
                "type": SplitIsValidJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
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
                UPSTREAM_RESPONSE_CONFIG_SIZE,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS,
                UPSTREAM_RESPONSE_SPLIT_DUCKDB_INDEX,
                UPSTREAM_RESPONSE_SPLIT_DESCRIPTIVE_STATISTICS,
            ],
            EXPECTED_ALL_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_CONFIG_SIZE_ERROR,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS,
            ],
            EXPECTED_PREVIEW_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS,
            ],
            EXPECTED_PREVIEW_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_CONFIG_SIZE,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_ERROR,
            ],
            EXPECTED_VIEWER_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_CONFIG_SIZE,
            ],
            EXPECTED_VIEWER_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_SPLIT_DUCKDB_INDEX,
            ],
            EXPECTED_SEARCH_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_SPLIT_DUCKDB_INDEX_ONLY_DATA,
            ],
            EXPECTED_FILTER_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_SPLIT_DESCRIPTIVE_STATISTICS,
            ],
            EXPECTED_STATISTICS_OK,
        ),
        (
            [
                UPSTREAM_RESPONSE_CONFIG_SIZE_ERROR,
                UPSTREAM_RESPONSE_SPLIT_FIRST_ROWS_ERROR,
                UPSTREAM_RESPONSE_SPLIT_DUCKDB_INDEX_ERROR,
            ],
            EXPECTED_ERROR,
        ),
        (
            [],
            EXPECTED_ERROR,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    upstream_responses: list[UpstreamResponse],
    expected: Any,
) -> None:
    dataset, config, split = DATASET, CONFIG, SPLIT
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, config, split, app_config)
    compute_result = job_runner.compute()
    assert compute_result.content == expected[0]
    assert compute_result.progress == expected[1]


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset, config, split = "doesnotexist", "doesnotexist", "doesnotexist"
    job_runner = get_job_runner(dataset, config, split, app_config)
    compute_result = job_runner.compute()
    assert compute_result.content == {
        "viewer": False,
        "preview": False,
        "search": False,
        "filter": False,
        "statistics": False,
    }

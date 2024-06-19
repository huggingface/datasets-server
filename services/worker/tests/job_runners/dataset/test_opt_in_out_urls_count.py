# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import pytest
from libcommon.dtos import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedArtifactNotFoundError, upsert_response

from worker.config import AppConfig
from worker.job_runners.dataset.opt_in_out_urls_count import (
    DatasetOptInOutUrlsCountJobRunner,
)

from ..utils import REVISION_NAME


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig], DatasetOptInOutUrlsCountJobRunner]


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetOptInOutUrlsCountJobRunner:
        return DatasetOptInOutUrlsCountJobRunner(
            job_info={
                "type": DatasetOptInOutUrlsCountJobRunner.get_job_type(),
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
    "dataset,config_names_status,config_names_content,config_upstream_status"
    + ",config_upstream_content,expected_error_code,expected_content,should_raise",
    [
        (
            "dataset_ok_full_scan",
            HTTPStatus.OK,
            {
                "config_names": [
                    {"dataset": "dataset_ok_full_scan", "config": "config1"},
                    {"dataset": "dataset_ok_full_scan", "config": "config2"},
                ]
            },
            [HTTPStatus.OK, HTTPStatus.OK],
            [
                {
                    "urls_columns": ["image_url", "url"],
                    "num_opt_in_urls": 10,
                    "num_opt_out_urls": 20,
                    "num_urls": 100,
                    "num_scanned_rows": 100,
                    "has_urls_columns": True,
                    "full_scan": True,
                },
                {
                    "urls_columns": ["image_url", "label", "url"],
                    "num_opt_in_urls": 10,
                    "num_opt_out_urls": 0,
                    "num_urls": 50,
                    "num_scanned_rows": 300,
                    "has_urls_columns": True,
                    "full_scan": True,
                },
            ],
            None,
            {
                "urls_columns": ["image_url", "label", "url"],
                "num_opt_in_urls": 20,
                "num_opt_out_urls": 20,
                "num_urls": 150,
                "num_scanned_rows": 400,
                "has_urls_columns": True,
                "full_scan": True,
            },
            False,
        ),
        (
            "dataset_ok_not_full_scan",
            HTTPStatus.OK,
            {
                "config_names": [
                    {"dataset": "dataset_ok_not_full_scan", "config": "config1"},
                    {"dataset": "dataset_ok_not_full_scan", "config": "config2"},
                ]
            },
            [HTTPStatus.OK, HTTPStatus.OK],
            [
                {
                    "urls_columns": ["image_url", "url"],
                    "num_opt_in_urls": 10,
                    "num_opt_out_urls": 20,
                    "num_urls": 100,
                    "num_scanned_rows": 100,
                    "has_urls_columns": True,
                    "full_scan": False,
                },
                {
                    "urls_columns": ["image_url", "label", "url"],
                    "num_opt_in_urls": 10,
                    "num_opt_out_urls": 0,
                    "num_urls": 50,
                    "num_scanned_rows": 300,
                    "has_urls_columns": True,
                    "full_scan": True,
                },
            ],
            None,
            {
                "urls_columns": ["image_url", "label", "url"],
                "num_opt_in_urls": 20,
                "num_opt_out_urls": 20,
                "num_urls": 150,
                "num_scanned_rows": 400,
                "has_urls_columns": True,
                "full_scan": False,
            },
            False,
        ),
        (
            "previos_step_error",
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {},
            [],
            [],
            "CachedArtifactError",
            None,
            True,
        ),
        (
            "previous_step_format_error",
            HTTPStatus.OK,
            {
                "config_names": [
                    {"dataset": "dataset_ok", "config": "config1"},
                    {"dataset": "dataset_ok", "config": "config2"},
                ]
            },
            [HTTPStatus.OK],
            [{"wrong_format": None}],
            "PreviousStepFormatError",
            None,
            True,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    config_names_status: HTTPStatus,
    config_names_content: Any,
    config_upstream_status: list[HTTPStatus],
    config_upstream_content: list[Any],
    expected_error_code: str,
    expected_content: Any,
    should_raise: bool,
) -> None:
    upsert_response(
        kind="dataset-config-names",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        content=config_names_content,
        http_status=config_names_status,
    )

    if config_names_status == HTTPStatus.OK:
        for split_item, status, content in zip(
            config_names_content["config_names"], config_upstream_status, config_upstream_content
        ):
            upsert_response(
                kind="config-opt-in-out-urls-count",
                dataset=dataset,
                dataset_git_revision=REVISION_NAME,
                config=split_item["config"],
                content=content,
                http_status=status,
            )

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

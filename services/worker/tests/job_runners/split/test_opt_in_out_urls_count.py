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
from worker.job_runners.split.opt_in_out_urls_count import (
    SplitOptInOutUrlsCountJobRunner,
)

from ..utils import REVISION_NAME


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, str, str, AppConfig], SplitOptInOutUrlsCountJobRunner]


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
    ) -> SplitOptInOutUrlsCountJobRunner:
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

        return SplitOptInOutUrlsCountJobRunner(
            job_info={
                "type": SplitOptInOutUrlsCountJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": REVISION_NAME,
                    "config": config,
                    "split": split,
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
    "dataset,config,split,upstream_status,upstream_content,expected_error_code,expected_content,should_raise",
    [
        (
            "dataset_ok",
            "config_ok",
            "split_ok",
            HTTPStatus.OK,
            {
                "has_urls_columns": True,
                "num_scanned_rows": 4,
                "opt_in_urls": [
                    {"url": "http://testurl.test/test_image3-optIn.jpg", "row_idx": 3, "column_name": "col"}
                ],
                "opt_out_urls": [
                    {"url": "http://testurl.test/test_image-optOut.jpg", "row_idx": 0, "column_name": "col"}
                ],
                "urls_columns": ["col"],
                "num_opt_out_urls": 1,
                "num_opt_in_urls": 1,
                "num_urls": 4,
                "full_scan": True,
            },
            None,
            {
                "has_urls_columns": True,
                "num_scanned_rows": 4,
                "urls_columns": ["col"],
                "num_opt_out_urls": 1,
                "num_opt_in_urls": 1,
                "num_urls": 4,
                "full_scan": True,
            },
            False,
        ),
        (
            "dataset_previous_step_error",
            "config_previous_step_error",
            "split_previous_step_error",
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {},
            "CachedArtifactError",
            None,
            True,
        ),
        (
            "dataset_format_error",
            "config_format_error",
            "split_format_error",
            HTTPStatus.OK,
            {"wrong_format": None},
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
    config: str,
    split: str,
    upstream_status: HTTPStatus,
    upstream_content: Any,
    expected_error_code: str,
    expected_content: Any,
    should_raise: bool,
) -> None:
    upsert_response(
        kind="split-opt-in-out-urls-scan",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        split=split,
        content=upstream_content,
        http_status=upstream_status,
    )
    job_runner = get_job_runner(dataset, config, split, app_config)
    if should_raise:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.typename == expected_error_code
    else:
        assert job_runner.compute().content == expected_content


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = config = split = "doesnotexist"
    job_runner = get_job_runner(dataset, config, split, app_config)
    with pytest.raises(CachedArtifactNotFoundError):
        job_runner.compute()

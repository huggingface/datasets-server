# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from asyncio import Semaphore
from collections.abc import Callable, Mapping
from dataclasses import replace
from http import HTTPStatus
from typing import Any
from unittest.mock import patch

import pytest
from aiohttp import ClientSession
from aiolimiter import AsyncLimiter
from libcommon.dtos import Priority
from libcommon.exceptions import ExternalServerError
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response

from worker.config import AppConfig
from worker.dtos import ImageUrlColumnsResponse
from worker.job_runners.split.opt_in_out_urls_scan_from_streaming import (
    SplitOptInOutUrlsScanJobRunner,
    check_spawning,
)
from worker.resources import LibrariesResource

from ...constants import CI_SPAWNING_TOKEN
from ...fixtures.hub import HubDatasetTest, get_default_config_split
from ..utils import REVISION_NAME

GetJobRunner = Callable[[str, str, str, AppConfig], SplitOptInOutUrlsScanJobRunner]


async def mock_check_spawning(
    image_urls: list[str], session: ClientSession, semaphore: Semaphore, limiter: AsyncLimiter, url: str
) -> Any:
    return {"urls": [{"url": url, "optIn": "optIn" in url, "optOut": "optOut" in url} for url in image_urls]}


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        split: str,
        app_config: AppConfig,
    ) -> SplitOptInOutUrlsScanJobRunner:
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

        return SplitOptInOutUrlsScanJobRunner(
            job_info={
                "type": SplitOptInOutUrlsScanJobRunner.get_job_type(),
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
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


IMAGE_URL_COLUMNS_RESPONSE_EMPTY: ImageUrlColumnsResponse = {"columns": []}


IMAGE_URL_COLUMNS_RESPONSE_WITH_DATA: ImageUrlColumnsResponse = {"columns": ["col"]}


DEFAULT_EMPTY_RESPONSE = {
    "has_urls_columns": False,
    "num_scanned_rows": 0,
    "opt_in_urls": [],
    "opt_out_urls": [],
    "urls_columns": [],
    "num_opt_out_urls": 0,
    "num_opt_in_urls": 0,
    "num_urls": 0,
    "full_scan": None,
}


@pytest.mark.parametrize(
    "name,rows_max_number,upstream_content,expected_content",
    [
        (
            "public",
            100_000,
            IMAGE_URL_COLUMNS_RESPONSE_EMPTY,
            DEFAULT_EMPTY_RESPONSE,
        ),
        (
            "spawning_opt_in_out",
            100_000,  # dataset has less rows
            IMAGE_URL_COLUMNS_RESPONSE_WITH_DATA,
            {
                "has_urls_columns": True,
                "num_scanned_rows": 5,
                "opt_in_urls": [
                    {"url": "http://testurl.test/test_image3-optIn.png", "row_idx": 3, "column_name": "col"}
                ],
                "opt_out_urls": [
                    {"url": "http://testurl.test/test_image-optOut.jpg", "row_idx": 0, "column_name": "col"}
                ],
                "urls_columns": ["col"],
                "num_opt_out_urls": 1,
                "num_opt_in_urls": 1,
                "num_urls": 5,
                "full_scan": True,
            },
        ),
        (
            "spawning_opt_in_out",
            3,  # dataset has more rows
            IMAGE_URL_COLUMNS_RESPONSE_WITH_DATA,
            {
                "has_urls_columns": True,
                "num_scanned_rows": 3,
                "opt_in_urls": [],
                "opt_out_urls": [
                    {"url": "http://testurl.test/test_image-optOut.jpg", "row_idx": 0, "column_name": "col"}
                ],
                "urls_columns": ["col"],
                "num_opt_out_urls": 1,
                "num_opt_in_urls": 0,
                "num_urls": 3,
                "full_scan": False,
            },
        ),
        (
            "spawning_opt_in_out",
            5,  # dataset has same amount of rows
            IMAGE_URL_COLUMNS_RESPONSE_WITH_DATA,
            {
                "has_urls_columns": True,
                "num_scanned_rows": 5,
                "opt_in_urls": [
                    {"url": "http://testurl.test/test_image3-optIn.png", "row_idx": 3, "column_name": "col"}
                ],
                "opt_out_urls": [
                    {"url": "http://testurl.test/test_image-optOut.jpg", "row_idx": 0, "column_name": "col"}
                ],
                "urls_columns": ["col"],
                "num_opt_out_urls": 1,
                "num_opt_in_urls": 1,
                "num_urls": 5,
                "full_scan": True,
            },
        ),
    ],
)
def test_compute(
    hub_responses_public: HubDatasetTest,
    hub_responses_spawning_opt_in_out: HubDatasetTest,
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    name: str,
    rows_max_number: int,
    upstream_content: Mapping[str, Any],
    expected_content: Mapping[str, Any],
) -> None:
    hub_datasets = {"public": hub_responses_public, "spawning_opt_in_out": hub_responses_spawning_opt_in_out}
    dataset = hub_datasets[name]["name"]
    config, split = get_default_config_split()
    job_runner = get_job_runner(
        dataset,
        config,
        split,
        replace(app_config, urls_scan=replace(app_config.urls_scan, rows_max_number=rows_max_number)),
    )
    upsert_response(
        kind="split-image-url-columns",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        split=split,
        content=upstream_content,
        job_runner_version=1,
        progress=1.0,
        http_status=HTTPStatus.OK,
    )
    with patch("worker.job_runners.split.opt_in_out_urls_scan_from_streaming.check_spawning", mock_check_spawning):
        response = job_runner.compute()
    assert response
    assert response.content == expected_content


@pytest.mark.parametrize(
    "dataset,columns_max_number,upstream_content,upstream_status,exception_name",
    [
        ("doesnotexist", 10, {}, HTTPStatus.OK, "CachedArtifactNotFoundError"),
        ("wrong_format", 10, {}, HTTPStatus.OK, "PreviousStepFormatError"),
        (
            "upstream_failed",
            10,
            {},
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "CachedArtifactError",
        ),
        (
            "info_error",
            10,
            IMAGE_URL_COLUMNS_RESPONSE_EMPTY,
            HTTPStatus.OK,
            "InfoError",
        ),
        (
            "too_many_columns",
            0,
            IMAGE_URL_COLUMNS_RESPONSE_WITH_DATA,
            HTTPStatus.OK,
            "TooManyColumnsError",
        ),
    ],
)
def test_compute_failed(
    app_config: AppConfig,
    hub_responses_spawning_opt_in_out: HubDatasetTest,
    get_job_runner: GetJobRunner,
    dataset: str,
    columns_max_number: int,
    upstream_content: Mapping[str, Any],
    upstream_status: HTTPStatus,
    exception_name: str,
) -> None:
    if dataset == "too_many_columns":
        dataset = hub_responses_spawning_opt_in_out["name"]
    config, split = get_default_config_split()
    job_runner = get_job_runner(
        dataset,
        config,
        split,
        replace(app_config, urls_scan=replace(app_config.urls_scan, columns_max_number=columns_max_number)),
    )
    if dataset != "doesnotexist":
        upsert_response(
            kind="split-image-url-columns",
            dataset=dataset,
            config=config,
            split=split,
            content=upstream_content,
            dataset_git_revision=REVISION_NAME,
            job_runner_version=1,
            progress=1.0,
            http_status=upstream_status,
        )
    with pytest.raises(Exception) as exc_info:
        job_runner.compute()
    assert exc_info.typename == exception_name


def test_compute_error_from_spawning(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    hub_public_spawning_opt_in_out: str,
) -> None:
    dataset = hub_public_spawning_opt_in_out
    config, split = get_default_config_split()
    job_runner = get_job_runner(
        dataset,
        config,
        split,
        replace(app_config, urls_scan=replace(app_config.urls_scan, spawning_url="wrong_url")),
    )
    upsert_response(
        kind="split-image-url-columns",
        dataset=dataset,
        config=config,
        split=split,
        content=IMAGE_URL_COLUMNS_RESPONSE_WITH_DATA,
        dataset_git_revision=REVISION_NAME,
        job_runner_version=1,
        progress=1.0,
        http_status=HTTPStatus.OK,
    )
    with pytest.raises(ExternalServerError):
        job_runner.compute()


@pytest.mark.skip(
    reason=(
        "Temporarily disabled, we can't use secrets on fork repos. See"
        " https://github.com/huggingface/dataset-viewer/issues/1085"
    )
)
@pytest.mark.asyncio
async def test_real_check_spawning_response(app_config: AppConfig) -> None:
    semaphore = Semaphore(value=10)
    limiter = AsyncLimiter(10, time_period=1)

    headers = {"Authorization": f"API {CI_SPAWNING_TOKEN}"}
    async with ClientSession(headers=headers) as session:
        image_url = "http://testurl.test/test_image.jpg"
        image_urls = [image_url]
        spawning_url = app_config.urls_scan.spawning_url
        spawning_response = await check_spawning(image_urls, session, semaphore, limiter, spawning_url)
        assert spawning_response and isinstance(spawning_response, dict)
        assert spawning_response["urls"] and isinstance(spawning_response["urls"], list)
        assert len(spawning_response["urls"]) == 2  # the API requires >1 urls
        first_url = spawning_response["urls"][0]
        assert first_url and isinstance(first_url, dict)
        assert first_url["url"] and isinstance(first_url["url"], str)
        assert first_url["url"] == image_url

# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Callable
from dataclasses import replace
from http import HTTPStatus
from typing import Any

import pytest
from libcommon.dtos import Priority
from libcommon.exceptions import (
    CustomError,
    DatasetManualDownloadError,
    PreviousStepFormatError,
)
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import (
    CachedArtifactError,
    CachedArtifactNotFoundError,
    upsert_response,
)

from worker.config import AppConfig
from worker.job_runners.config.split_names import (
    ConfigSplitNamesJobRunner,
    compute_split_names_from_info_response,
    compute_split_names_from_streaming_response,
)
from worker.resources import LibrariesResource

from ...fixtures.hub import HubDatasetTest, get_default_config_split
from ..utils import REVISION_NAME

GetJobRunner = Callable[[str, str, AppConfig], ConfigSplitNamesJobRunner]


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigSplitNamesJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        return ConfigSplitNamesJobRunner(
            job_info={
                "type": ConfigSplitNamesJobRunner.get_job_type(),
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
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,upstream_status,upstream_content,error_code,content",
    [
        (
            "ok",
            HTTPStatus.OK,
            {
                "dataset_info": {
                    "splits": {
                        "train": {"name": "train", "dataset_name": "ok"},
                        "validation": {"name": "validation", "dataset_name": "ok"},
                        "test": {"name": "test", "dataset_name": "ok"},
                    },
                }
            },
            None,
            {
                "splits": [
                    {"dataset": "ok", "config": "config_name", "split": "train"},
                    {"dataset": "ok", "config": "config_name", "split": "validation"},
                    {"dataset": "ok", "config": "config_name", "split": "test"},
                ]
            },
        ),
        (
            "upstream_fail",
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {"error": "error"},
            CachedArtifactError.__name__,
            None,
        ),
        (
            "without_dataset_info",
            HTTPStatus.OK,
            {"some_column": "wrong_format"},
            PreviousStepFormatError.__name__,
            None,
        ),
        (
            "without_config_name",
            HTTPStatus.OK,
            {"dataset_info": "wrong_format"},
            PreviousStepFormatError.__name__,
            None,
        ),
        (
            "without_splits",
            HTTPStatus.OK,
            {"dataset_info": {"config_name": "wrong_format"}},
            PreviousStepFormatError.__name__,
            None,
        ),
    ],
)
def test_compute_split_names_from_info_response(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
    dataset: str,
    upstream_status: HTTPStatus,
    upstream_content: Any,
    error_code: str,
    content: Any,
) -> None:
    config = "config_name"
    upsert_response(
        kind="config-info",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        content=upstream_content,
        http_status=upstream_status,
    )

    if error_code:
        with pytest.raises(Exception) as e:
            compute_split_names_from_info_response(dataset, config, max_number=999)
        assert e.typename == error_code
    else:
        assert compute_split_names_from_info_response(dataset, config, max_number=999) == content


def test_doesnotexist(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> None:
    dataset = "non_existent"
    config = "non_existent"
    with pytest.raises(CachedArtifactNotFoundError):
        compute_split_names_from_info_response(dataset, config, max_number=999)


@pytest.mark.parametrize(
    "name,use_token,error_code,cause",
    [
        ("public", False, None, None),
        ("audio", False, None, None),
        ("gated", True, None, None),
        ("private", True, None, None),
        ("empty", False, "EmptyDatasetError", "EmptyDatasetError"),
        # should we really test the following cases?
        # The assumption is that the dataset exists and is accessible with the token
        ("does_not_exist", False, "SplitNamesFromStreamingError", "DatasetNotFoundError"),
        ("gated", False, "SplitNamesFromStreamingError", "DatasetNotFoundError"),
        ("private", False, "SplitNamesFromStreamingError", "DatasetNotFoundError"),
    ],
)
def test_compute_split_names_from_streaming_response(
    hub_responses_public: HubDatasetTest,
    hub_responses_audio: HubDatasetTest,
    hub_responses_gated: HubDatasetTest,
    hub_responses_private: HubDatasetTest,
    hub_responses_empty: HubDatasetTest,
    hub_responses_does_not_exist: HubDatasetTest,
    get_job_runner: GetJobRunner,
    name: str,
    use_token: bool,
    error_code: str,
    cause: str,
    app_config: AppConfig,
) -> None:
    hub_datasets = {
        "public": hub_responses_public,
        "audio": hub_responses_audio,
        "gated": hub_responses_gated,
        "private": hub_responses_private,
        "empty": hub_responses_empty,
        "does_not_exist": hub_responses_does_not_exist,
    }
    dataset = hub_datasets[name]["name"]
    config, _ = get_default_config_split()
    expected_configs_response = hub_datasets[name]["splits_response"]
    job_runner = get_job_runner(
        dataset,
        config,
        app_config if use_token else replace(app_config, common=replace(app_config.common, hf_token=None)),
    )
    if error_code is None:
        result = job_runner.compute().content
        assert result == expected_configs_response
        return

    with pytest.raises(CustomError) as exc_info:
        job_runner.compute()
    assert exc_info.value.code == error_code
    assert exc_info.value.cause_exception == cause
    if exc_info.value.disclose_cause:
        response = exc_info.value.as_response()
        assert set(response.keys()) == {"error", "cause_exception", "cause_message", "cause_traceback"}
        response_dict = dict(response)
        # ^ to remove mypy warnings
        assert response_dict["cause_exception"] == cause
        assert isinstance(response_dict["cause_traceback"], list)
        assert response_dict["cause_traceback"][0] == "Traceback (most recent call last):\n"


def test_compute_split_names_from_streaming_response_raises(
    hub_public_manual_download: str, app_config: AppConfig
) -> None:
    with pytest.raises(DatasetManualDownloadError):
        compute_split_names_from_streaming_response(
            hub_public_manual_download,
            "default",
            max_number=999,
            hf_token=app_config.common.hf_token,
            dataset_scripts_allow_list=[hub_public_manual_download],
        )


def test_compute(app_config: AppConfig, get_job_runner: GetJobRunner, hub_public_csv: str) -> None:
    dataset = hub_public_csv
    config, _ = get_default_config_split()
    job_runner = get_job_runner(dataset, config, app_config)
    response = job_runner.compute()
    content = response.content
    assert len(content["splits"]) == 1

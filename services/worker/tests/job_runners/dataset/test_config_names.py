# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Callable
from dataclasses import replace
from unittest.mock import patch

import pytest
from libcommon.dtos import Priority
from libcommon.exceptions import CustomError
from libcommon.resources import CacheMongoResource, QueueMongoResource

from worker.config import AppConfig
from worker.job_runners.dataset.config_names import DatasetConfigNamesJobRunner
from worker.resources import LibrariesResource

from ...fixtures.hub import HubDatasetTest
from ..utils import REVISION_NAME

GetJobRunner = Callable[[str, AppConfig], DatasetConfigNamesJobRunner]


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetConfigNamesJobRunner:
        return DatasetConfigNamesJobRunner(
            job_info={
                "type": DatasetConfigNamesJobRunner.get_job_type(),
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
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


def test_compute(app_config: AppConfig, hub_public_csv: str, get_job_runner: GetJobRunner) -> None:
    dataset = hub_public_csv
    job_runner = get_job_runner(dataset, app_config)
    response = job_runner.compute()
    content = response.content
    assert len(content["config_names"]) == 1


@pytest.mark.parametrize(
    "max_number_of_configs,error_code",
    [
        (1, "DatasetWithTooManyConfigsError"),
        (2, None),
        (3, None),
    ],
)
def test_compute_too_many_configs(
    app_config: AppConfig, get_job_runner: GetJobRunner, max_number_of_configs: int, error_code: str
) -> None:
    dataset = "dataset"
    configs = ["config_1", "config_2"]
    job_runner = get_job_runner(
        dataset,
        replace(app_config, config_names=replace(app_config.config_names, max_number=max_number_of_configs)),
    )

    with patch("worker.job_runners.dataset.config_names.get_dataset_config_names", return_value=configs):
        with patch("worker.job_runners.dataset.config_names.get_dataset_default_config_name", return_value=None):
            if error_code:
                with pytest.raises(CustomError) as exc_info:
                    job_runner.compute()
                assert exc_info.value.code == error_code
            else:
                assert job_runner.compute() is not None


@pytest.mark.parametrize(
    "name,use_token,error_code,cause",
    [
        ("public", False, None, None),
        ("audio", False, None, None),
        ("gated", True, None, None),
        ("private", True, None, None),
        ("empty", False, "EmptyDatasetError", "EmptyDatasetError"),
        ("n_configs_with_default", False, None, None),
        # should we really test the following cases?
        # The assumption is that the dataset exists and is accessible with the token
        ("does_not_exist", False, "ConfigNamesError", "DatasetNotFoundError"),
        ("gated", False, "ConfigNamesError", "DatasetNotFoundError"),
        ("private", False, "ConfigNamesError", "DatasetNotFoundError"),
    ],
)
def test_compute_splits_response(
    hub_responses_public: HubDatasetTest,
    hub_responses_audio: HubDatasetTest,
    hub_responses_gated: HubDatasetTest,
    hub_responses_private: HubDatasetTest,
    hub_responses_empty: HubDatasetTest,
    hub_responses_does_not_exist: HubDatasetTest,
    hub_responses_n_configs_with_default: HubDatasetTest,
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
        "n_configs_with_default": hub_responses_n_configs_with_default,
    }
    dataset = hub_datasets[name]["name"]
    expected_configs_response = hub_datasets[name]["config_names_response"]
    job_runner = get_job_runner(
        dataset,
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

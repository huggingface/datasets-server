# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import pytest
from libcommon.dtos import Priority
from libcommon.exceptions import PreviousStepFormatError
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedArtifactNotFoundError, upsert_response

from worker.config import AppConfig
from worker.job_runners.dataset.split_names import DatasetSplitNamesJobRunner

from ..utils import REVISION_NAME

GetJobRunner = Callable[[str, AppConfig], DatasetSplitNamesJobRunner]


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetSplitNamesJobRunner:
        return DatasetSplitNamesJobRunner(
            job_info={
                "type": DatasetSplitNamesJobRunner.get_job_type(),
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
    "dataset,split_names,expected_content,progress",
    [
        (
            "pending_response",
            [
                {
                    "config": "config_a",
                    "response": {
                        "splits": [
                            {
                                "dataset": "pending_response",
                                "config": "config_a",
                                "split": "split_a",
                            }
                        ]
                    },
                }
            ],
            {
                "splits": [
                    {
                        "dataset": "pending_response",
                        "config": "config_a",
                        "split": "split_a",
                    },
                ],
                "pending": [{"dataset": "pending_response", "config": "config_b"}],
                "failed": [],
            },
            0.5,
        ),
        (
            "complete",
            [
                {
                    "config": "config_a",
                    "response": {
                        "splits": [
                            {
                                "dataset": "complete",
                                "config": "config_a",
                                "split": "split_a",
                            }
                        ]
                    },
                },
                {
                    "config": "config_b",
                    "response": {
                        "splits": [
                            {
                                "dataset": "complete",
                                "config": "config_b",
                                "split": "split_b",
                            }
                        ]
                    },
                },
            ],
            {
                "splits": [
                    {
                        "dataset": "complete",
                        "config": "config_a",
                        "split": "split_a",
                    },
                    {
                        "dataset": "complete",
                        "config": "config_b",
                        "split": "split_b",
                    },
                ],
                "pending": [],
                "failed": [],
            },
            1,
        ),
    ],
)
def test_compute_progress(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    split_names: Any,
    expected_content: Any,
    progress: float,
) -> None:
    # we could also have tested if dataset-info has a response (it's one case among many, see
    # libcommon.simple_cache.get_response -> CachedArtifactNotFoundError)
    upsert_response(
        kind="dataset-config-names",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        content={
            "config_names": [
                {
                    "dataset": dataset,
                    "config": "config_a",
                },
                {"dataset": dataset, "config": "config_b"},
            ]
        },
        http_status=HTTPStatus.OK,
    )
    for config in split_names:
        upsert_response(
            kind="config-split-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            config=config["config"],
            content=config["response"],
            http_status=HTTPStatus.OK,
        )
    job_runner = get_job_runner(dataset, app_config)
    response = job_runner.compute()
    assert response.content == expected_content
    assert response.progress == progress


def test_compute_error(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "error"
    config = "error"
    # we could also have tested if dataset-info has a response (it's one case among many, see
    # libcommon.simple_cache.get_response -> CachedArtifactNotFoundError)
    upsert_response(
        kind="dataset-config-names",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        content={
            "config_names": [
                {
                    "dataset": dataset,
                    "config": config,
                }
            ]
        },
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind="config-split-names",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        content={},
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    )
    job_runner = get_job_runner(dataset, app_config)
    response = job_runner.compute()
    assert response.content == {
        "splits": [],
        "failed": [{"dataset": dataset, "config": config, "error": {}}],
        "pending": [],
    }
    assert response.progress == 1.0


def test_compute_format_error(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "error"
    config = "error"
    upsert_response(
        kind="dataset-config-names",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        content={
            "config_names": [
                {
                    "dataset": dataset,
                    "config": config,
                }
            ]
        },
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind="config-split-names",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        content={"wrong_format": []},
        http_status=HTTPStatus.OK,
    )
    job_runner = get_job_runner(dataset, app_config)
    with pytest.raises(PreviousStepFormatError):
        job_runner.compute()


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "doesnotexist"
    job_runner = get_job_runner(dataset, app_config)
    with pytest.raises(CachedArtifactNotFoundError):
        job_runner.compute()

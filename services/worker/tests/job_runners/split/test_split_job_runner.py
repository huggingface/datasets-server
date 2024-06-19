# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus
from typing import Optional

import pytest
from libcommon.dtos import Priority
from libcommon.exceptions import CustomError
from libcommon.resources import CacheMongoResource
from libcommon.simple_cache import upsert_response

from worker.config import AppConfig
from worker.dtos import CompleteJobResult
from worker.job_runners.split.split_job_runner import SplitJobRunner

from ..utils import REVISION_NAME


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


class DummySplitJobRunner(SplitJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "/dummy"

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult({"key": "value"})


@pytest.mark.parametrize("config,split", [(None, None), (None, "split"), ("config", None)])
def test_failed_creation(app_config: AppConfig, config: str, split: str) -> None:
    upsert_response(
        kind="dataset-config-names",
        dataset="dataset",
        dataset_git_revision=REVISION_NAME,
        content={"config_names": [{"dataset": "dataset", "config": config}]},
        http_status=HTTPStatus.OK,
    )

    with pytest.raises(CustomError) as exc_info:
        DummySplitJobRunner(
            job_info={
                "job_id": "job_id",
                "type": "dummy",
                "params": {
                    "dataset": "dataset",
                    "revision": REVISION_NAME,
                    "config": config,
                    "split": split,
                },
                "priority": Priority.NORMAL,
                "difficulty": 50,
                "started_at": None,
            },
            app_config=app_config,
        ).validate()
    assert exc_info.value.code == "ParameterMissingError"


@pytest.mark.parametrize(
    "upsert_config,upsert_split,exception_name",
    [
        ("config", "split", None),
        ("config", "other_split", "SplitNotFoundError"),
        ("other_config", "split", "ConfigNotFoundError"),
    ],
)
def test_creation(
    app_config: AppConfig,
    upsert_config: str,
    upsert_split: str,
    exception_name: Optional[str],
) -> None:
    dataset, config, split = "dataset", "config", "split"

    upsert_response(
        kind="dataset-config-names",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        content={"config_names": [{"dataset": dataset, "config": upsert_config}]},
        http_status=HTTPStatus.OK,
    )

    upsert_response(
        kind="config-split-names",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        content={"splits": [{"dataset": dataset, "config": upsert_config, "split": upsert_split}]},
        http_status=HTTPStatus.OK,
    )

    if exception_name is None:
        DummySplitJobRunner(
            job_info={
                "job_id": "job_id",
                "type": "dummy",
                "params": {
                    "dataset": dataset,
                    "revision": REVISION_NAME,
                    "config": config,
                    "split": split,
                },
                "priority": Priority.NORMAL,
                "difficulty": 50,
                "started_at": None,
            },
            app_config=app_config,
        ).validate()
    else:
        with pytest.raises(CustomError) as exc_info:
            DummySplitJobRunner(
                job_info={
                    "job_id": "job_id",
                    "type": "dummy",
                    "params": {
                        "dataset": dataset,
                        "revision": REVISION_NAME,
                        "config": config,
                        "split": split,
                    },
                    "priority": Priority.NORMAL,
                    "difficulty": 50,
                    "started_at": None,
                },
                app_config=app_config,
            ).validate()
        assert exc_info.value.code == exception_name

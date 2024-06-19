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
from worker.job_runners.config.config_job_runner import ConfigJobRunner

from ..utils import REVISION_NAME


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


class DummyConfigJobRunner(ConfigJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "/dummy"

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult({"key": "value"})


def test_failed_creation(app_config: AppConfig) -> None:
    with pytest.raises(CustomError) as exc_info:
        DummyConfigJobRunner(
            job_info={
                "job_id": "job_id",
                "type": "dummy",
                "params": {
                    "dataset": "dataset",
                    "revision": REVISION_NAME,
                    "config": None,
                    "split": None,
                },
                "priority": Priority.NORMAL,
                "difficulty": 50,
                "started_at": None,
            },
            app_config=app_config,
        ).validate()
    assert exc_info.value.code == "ParameterMissingError"


@pytest.mark.parametrize(
    "upsert_config,exception_name",
    [
        ("config", None),
        ("other_config", "ConfigNotFoundError"),
    ],
)
def test_creation(
    app_config: AppConfig,
    upsert_config: str,
    exception_name: Optional[str],
) -> None:
    dataset, config = "dataset", "config"

    upsert_response(
        kind="dataset-config-names",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        content={"config_names": [{"dataset": dataset, "config": upsert_config}]},
        http_status=HTTPStatus.OK,
    )

    if exception_name is None:
        DummyConfigJobRunner(
            job_info={
                "job_id": "job_id",
                "type": "dummy",
                "params": {
                    "dataset": dataset,
                    "revision": REVISION_NAME,
                    "config": config,
                    "split": None,
                },
                "priority": Priority.NORMAL,
                "difficulty": 50,
                "started_at": None,
            },
            app_config=app_config,
        ).validate()
    else:
        with pytest.raises(CustomError) as exc_info:
            DummyConfigJobRunner(
                job_info={
                    "job_id": "job_id",
                    "type": "dummy",
                    "params": {
                        "dataset": dataset,
                        "revision": REVISION_NAME,
                        "config": config,
                        "split": None,
                    },
                    "priority": Priority.NORMAL,
                    "difficulty": 50,
                    "started_at": None,
                },
                app_config=app_config,
            ).validate()
        assert exc_info.value.code == exception_name

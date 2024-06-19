# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import pytest
from libcommon.dtos import Priority
from libcommon.exceptions import CustomError

from worker.config import AppConfig
from worker.dtos import CompleteJobResult
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner


class DummyDatasetJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "/dummy"

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult({"key": "value"})


def test_failed_creation(app_config: AppConfig) -> None:
    with pytest.raises(CustomError) as exc_info:
        DummyDatasetJobRunner(
            job_info={
                "job_id": "job_id",
                "type": "dummy",
                "params": {
                    "dataset": None,  # type: ignore
                    # ^ Needed to raise error
                    "revision": "revision",
                    "config": None,
                    "split": None,
                },
                "priority": Priority.NORMAL,
                "difficulty": 50,
                "started_at": None,
            },
            app_config=app_config,
        )
    assert exc_info.value.code == "ParameterMissingError"


def test_success_creation(app_config: AppConfig) -> None:
    assert (
        DummyDatasetJobRunner(
            job_info={
                "job_id": "job_id",
                "type": "dummy",
                "params": {
                    "dataset": "dataset",
                    "revision": "revision",
                    "config": None,
                    "split": None,
                },
                "priority": Priority.NORMAL,
                "difficulty": 50,
                "started_at": None,
            },
            app_config=app_config,
        )
        is not None
    )

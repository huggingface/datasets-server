# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import random
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import pytest
from libcommon.dtos import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource

from worker.config import AppConfig
from worker.dtos import CompleteJobResult
from worker.job_runners._job_runner_with_cache import JobRunnerWithCache
from worker.resources import LibrariesResource

from ..fixtures.hub import get_default_config_split


class DummyJobRunner(JobRunnerWithCache):
    @staticmethod
    def get_job_type() -> str:
        return "dummy-job-runner"

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult({"col1": "a" * 200})


GetJobRunner = Callable[[str, Optional[str], Optional[str], AppConfig], DummyJobRunner]


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: Optional[str],
        split: Optional[str],
        app_config: AppConfig,
    ) -> DummyJobRunner:
        return DummyJobRunner(
            job_info={
                "type": DummyJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
                    "config": config,
                    "split": split,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 50,
                "started_at": None,
            },
            app_config=app_config,
            cache_directory=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,config,split,expected",
    [
        ("user/dataset", "config", "split", "64218998941645-dummy-job-runner-user-dataset-da67625f"),
        # Every parameter variation changes the hash, hence the subdirectory
        ("user/dataset", None, "split", "64218998941645-dummy-job-runner-user-dataset-498c21fa"),
        ("user/dataset", "config2", "split", "64218998941645-dummy-job-runner-user-dataset-1c4f24f2"),
        ("user/dataset", "config", None, "64218998941645-dummy-job-runner-user-dataset-a87e8dc2"),
        ("user/dataset", "config", "split2", "64218998941645-dummy-job-runner-user-dataset-f169bd48"),
        # The subdirectory length is truncated, and it always finishes with the hash
        (
            "very_long_dataset_name_0123456789012345678901234567890123456789012345678901234567890123456789",
            "config",
            "split",
            "64218998941645-dummy-job-runner-very_long_dataset_name_012345678-25cb8442",
        ),
    ],
)
def test_get_cache_subdirectory(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    config: Optional[str],
    split: Optional[str],
    expected: str,
) -> None:
    job_runner = get_job_runner(dataset, config, split, app_config)
    random.seed(0)
    assert job_runner.get_cache_subdirectory() == expected


def test_pre_compute_post_compute(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "user/dataset"
    config, split = get_default_config_split()
    job_runner = get_job_runner(dataset, config, split, app_config)
    datasets_base_path = job_runner.base_cache_directory
    job_runner.pre_compute()
    datasets_cache_subdirectory = job_runner.cache_subdirectory
    assert_datasets_cache_path(path=datasets_cache_subdirectory, exists=True)
    job_runner.post_compute()
    assert_datasets_cache_path(path=datasets_base_path, exists=True)
    assert_datasets_cache_path(path=datasets_cache_subdirectory, exists=False)


def assert_datasets_cache_path(path: Optional[Path], exists: bool) -> None:
    assert path is not None
    assert path.exists() is exists

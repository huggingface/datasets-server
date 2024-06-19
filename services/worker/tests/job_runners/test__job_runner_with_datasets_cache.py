# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Callable
from pathlib import Path
from typing import Optional

import datasets.config
import pytest
from libcommon.dtos import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource

from worker.config import AppConfig
from worker.dtos import CompleteJobResult
from worker.job_runners._job_runner_with_datasets_cache import (
    JobRunnerWithDatasetsCache,
)
from worker.resources import LibrariesResource

from ..fixtures.hub import get_default_config_split


class DummyJobRunner(JobRunnerWithDatasetsCache):
    @staticmethod
    def get_job_type() -> str:
        return "dummy-job-runner"
        # ^ borrowing the type, so that the processing step exists and the job runner can be initialized
        # refactoring libcommon.processing_graph might help avoiding this

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
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


def test_set_datasets_cache(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "dataset"
    config, split = get_default_config_split()
    job_runner = get_job_runner(dataset, config, split, app_config)
    base_path = job_runner.base_cache_directory
    dummy_path = base_path / "dummy"
    job_runner.set_datasets_cache(dummy_path)
    assert str(datasets.config.HF_DATASETS_CACHE).startswith(str(dummy_path))


def test_pre_compute_post_compute(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = "user/dataset"
    config, split = get_default_config_split()
    job_runner = get_job_runner(dataset, config, split, app_config)
    datasets_base_path = job_runner.base_cache_directory
    job_runner.pre_compute()
    datasets_cache_subdirectory = job_runner.cache_subdirectory
    assert_datasets_cache_path(path=datasets_cache_subdirectory, exists=True)
    assert str(datasets.config.HF_DATASETS_CACHE).startswith(str(datasets_base_path))
    assert "dummy-job-runner-user-dataset" in str(datasets.config.HF_DATASETS_CACHE)
    job_runner.post_compute()
    assert_datasets_cache_path(path=datasets_base_path, exists=True)
    assert_datasets_cache_path(path=datasets_cache_subdirectory, exists=False, equals=False)


def assert_datasets_cache_path(path: Optional[Path], exists: bool, equals: bool = True) -> None:
    assert path is not None
    assert path.exists() is exists
    assert (datasets.config.HF_DATASETS_CACHE == path) is equals
    assert (datasets.config.DOWNLOADED_DATASETS_PATH == path / datasets.config.DOWNLOADED_DATASETS_DIR) is equals
    assert (datasets.config.EXTRACTED_DATASETS_PATH == path / datasets.config.EXTRACTED_DATASETS_DIR) is equals

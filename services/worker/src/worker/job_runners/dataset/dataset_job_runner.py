# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pathlib import Path

from libcommon.dtos import JobInfo
from libcommon.exceptions import ParameterMissingError

from worker.config import AppConfig
from worker.job_runner import JobRunner
from worker.job_runners._job_runner_with_datasets_cache import (
    JobRunnerWithDatasetsCache,
)


class DatasetJobRunner(JobRunner):
    dataset: str
    dataset_git_revision: str

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
    ) -> None:
        super().__init__(job_info=job_info, app_config=app_config)
        if job_info["params"]["dataset"] is None:
            raise ParameterMissingError("'dataset' parameter is required")
        if job_info["params"]["revision"] is None:
            raise ParameterMissingError("'revision' parameter is required")
        self.dataset = job_info["params"]["dataset"]
        self.dataset_git_revision = job_info["params"]["revision"]


class DatasetJobRunnerWithDatasetsCache(JobRunnerWithDatasetsCache, DatasetJobRunner):
    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        hf_datasets_cache: Path,
    ) -> None:
        JobRunnerWithDatasetsCache.__init__(
            self=self,
            job_info=job_info,
            app_config=app_config,
            hf_datasets_cache=hf_datasets_cache,
        )
        DatasetJobRunner.__init__(
            self=self,
            job_info=job_info,
            app_config=app_config,
        )

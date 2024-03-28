# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from libcommon.dtos import JobInfo
from libcommon.storage import StrPath
from libcommon.storage_client import StorageClient

from worker.config import AppConfig
from worker.job_runner import JobRunner
from worker.job_runners.config.duckdb_index_size import (
    ConfigDuckdbIndexSizeJobRunner,
)
from worker.job_runners.config.info import ConfigInfoJobRunner
from worker.job_runners.config.is_valid import ConfigIsValidJobRunner
from worker.job_runners.config.opt_in_out_urls_count import (
    ConfigOptInOutUrlsCountJobRunner,
)
from worker.job_runners.config.parquet import ConfigParquetJobRunner
from worker.job_runners.config.parquet_and_info import ConfigParquetAndInfoJobRunner
from worker.job_runners.config.parquet_metadata import ConfigParquetMetadataJobRunner
from worker.job_runners.config.size import ConfigSizeJobRunner
from worker.job_runners.config.split_names import ConfigSplitNamesJobRunner
from worker.job_runners.dataset.compatible_libraries import DatasetCompatibleLibrariesJobRunner
from worker.job_runners.dataset.config_names import DatasetConfigNamesJobRunner
from worker.job_runners.dataset.croissant_crumbs import DatasetCroissantCrumbsJobRunner
from worker.job_runners.dataset.duckdb_index_size import (
    DatasetDuckdbIndexSizeJobRunner,
)
from worker.job_runners.dataset.hub_cache import DatasetHubCacheJobRunner
from worker.job_runners.dataset.info import DatasetInfoJobRunner
from worker.job_runners.dataset.is_valid import DatasetIsValidJobRunner
from worker.job_runners.dataset.modalities import DatasetModalitiesJobRunner
from worker.job_runners.dataset.opt_in_out_urls_count import (
    DatasetOptInOutUrlsCountJobRunner,
)
from worker.job_runners.dataset.parquet import DatasetParquetJobRunner
from worker.job_runners.dataset.size import DatasetSizeJobRunner
from worker.job_runners.dataset.split_names import DatasetSplitNamesJobRunner
from worker.job_runners.split.descriptive_statistics import (
    SplitDescriptiveStatisticsJobRunner,
)
from worker.job_runners.split.duckdb_index import SplitDuckDbIndexJobRunner
from worker.job_runners.split.first_rows import SplitFirstRowsJobRunner
from worker.job_runners.split.image_url_columns import SplitImageUrlColumnsJobRunner
from worker.job_runners.split.is_valid import SplitIsValidJobRunner
from worker.job_runners.split.opt_in_out_urls_count import (
    SplitOptInOutUrlsCountJobRunner,
)
from worker.job_runners.split.opt_in_out_urls_scan_from_streaming import (
    SplitOptInOutUrlsScanJobRunner,
)


class BaseJobRunnerFactory(ABC):
    """
    Base class for job runner factories. A job runner factory is a class that creates a job runner.

    It cannot be instantiated directly, but must be subclassed.

    Note that this class is only implemented once in the code, but we need it for the tests.
    """

    def create_job_runner(self, job_info: JobInfo) -> JobRunner:
        return self._create_job_runner(job_info=job_info)

    @abstractmethod
    def _create_job_runner(self, job_info: JobInfo) -> JobRunner:
        pass


@dataclass
class JobRunnerFactory(BaseJobRunnerFactory):
    app_config: AppConfig
    hf_datasets_cache: Path
    parquet_metadata_directory: StrPath
    duckdb_index_cache_directory: StrPath
    statistics_cache_directory: StrPath
    storage_client: StorageClient

    def _create_job_runner(self, job_info: JobInfo) -> JobRunner:
        job_type = job_info["type"]
        if job_type == DatasetConfigNamesJobRunner.get_job_type():
            return DatasetConfigNamesJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                hf_datasets_cache=self.hf_datasets_cache,
            )
        if job_type == ConfigSplitNamesJobRunner.get_job_type():
            return ConfigSplitNamesJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                hf_datasets_cache=self.hf_datasets_cache,
            )
        if job_type == SplitFirstRowsJobRunner.get_job_type():
            return SplitFirstRowsJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                hf_datasets_cache=self.hf_datasets_cache,
                parquet_metadata_directory=self.parquet_metadata_directory,
                storage_client=self.storage_client,
            )
        if job_type == ConfigParquetAndInfoJobRunner.get_job_type():
            return ConfigParquetAndInfoJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                hf_datasets_cache=self.hf_datasets_cache,
            )
        if job_type == ConfigParquetJobRunner.get_job_type():
            return ConfigParquetJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == ConfigParquetMetadataJobRunner.get_job_type():
            return ConfigParquetMetadataJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                parquet_metadata_directory=self.parquet_metadata_directory,
            )
        if job_type == DatasetParquetJobRunner.get_job_type():
            return DatasetParquetJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == DatasetInfoJobRunner.get_job_type():
            return DatasetInfoJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == ConfigInfoJobRunner.get_job_type():
            return ConfigInfoJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == DatasetSizeJobRunner.get_job_type():
            return DatasetSizeJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == ConfigSizeJobRunner.get_job_type():
            return ConfigSizeJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == DatasetSplitNamesJobRunner.get_job_type():
            return DatasetSplitNamesJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == SplitIsValidJobRunner.get_job_type():
            return SplitIsValidJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == ConfigIsValidJobRunner.get_job_type():
            return ConfigIsValidJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == DatasetIsValidJobRunner.get_job_type():
            return DatasetIsValidJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == SplitImageUrlColumnsJobRunner.get_job_type():
            return SplitImageUrlColumnsJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == SplitOptInOutUrlsScanJobRunner.get_job_type():
            return SplitOptInOutUrlsScanJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                hf_datasets_cache=self.hf_datasets_cache,
            )
        if job_type == ConfigOptInOutUrlsCountJobRunner.get_job_type():
            return ConfigOptInOutUrlsCountJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == DatasetOptInOutUrlsCountJobRunner.get_job_type():
            return DatasetOptInOutUrlsCountJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )

        if job_type == SplitOptInOutUrlsCountJobRunner.get_job_type():
            return SplitOptInOutUrlsCountJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == SplitDescriptiveStatisticsJobRunner.get_job_type():
            return SplitDescriptiveStatisticsJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                statistics_cache_directory=self.statistics_cache_directory,
                parquet_metadata_directory=self.parquet_metadata_directory,
            )
        if job_type == SplitDuckDbIndexJobRunner.get_job_type():
            return SplitDuckDbIndexJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                duckdb_index_cache_directory=self.duckdb_index_cache_directory,
                parquet_metadata_directory=self.parquet_metadata_directory,
            )
        if job_type == ConfigDuckdbIndexSizeJobRunner.get_job_type():
            return ConfigDuckdbIndexSizeJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == DatasetDuckdbIndexSizeJobRunner.get_job_type():
            return DatasetDuckdbIndexSizeJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == DatasetHubCacheJobRunner.get_job_type():
            return DatasetHubCacheJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == DatasetCompatibleLibrariesJobRunner.get_job_type():
            return DatasetCompatibleLibrariesJobRunner(
                job_info=job_info,
                app_config=self.app_config,
                hf_datasets_cache=self.hf_datasets_cache,
            )
        if job_type == DatasetModalitiesJobRunner.get_job_type():
            return DatasetModalitiesJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        if job_type == DatasetCroissantCrumbsJobRunner.get_job_type():
            return DatasetCroissantCrumbsJobRunner(
                job_info=job_info,
                app_config=self.app_config,
            )
        supported_job_types = [
            DatasetConfigNamesJobRunner.get_job_type(),
            ConfigSplitNamesJobRunner.get_job_type(),
            SplitFirstRowsJobRunner.get_job_type(),
            ConfigParquetAndInfoJobRunner.get_job_type(),
            ConfigParquetJobRunner.get_job_type(),
            DatasetParquetJobRunner.get_job_type(),
            DatasetInfoJobRunner.get_job_type(),
            ConfigInfoJobRunner.get_job_type(),
            DatasetSizeJobRunner.get_job_type(),
            ConfigSizeJobRunner.get_job_type(),
            SplitIsValidJobRunner.get_job_type(),
            ConfigIsValidJobRunner.get_job_type(),
            DatasetIsValidJobRunner.get_job_type(),
            SplitImageUrlColumnsJobRunner.get_job_type(),
            SplitOptInOutUrlsScanJobRunner.get_job_type(),
            SplitOptInOutUrlsCountJobRunner.get_job_type(),
            ConfigOptInOutUrlsCountJobRunner.get_job_type(),
            DatasetOptInOutUrlsCountJobRunner.get_job_type(),
            SplitDuckDbIndexJobRunner.get_job_type(),
            SplitDescriptiveStatisticsJobRunner.get_job_type(),
            ConfigDuckdbIndexSizeJobRunner.get_job_type(),
            DatasetDuckdbIndexSizeJobRunner.get_job_type(),
            DatasetHubCacheJobRunner.get_job_type(),
            DatasetCompatibleLibrariesJobRunner.get_job_type(),
            DatasetModalitiesJobRunner.get_job_type(),
            DatasetCroissantCrumbsJobRunner.get_job_type(),
        ]
        raise KeyError(f"Unsupported job type: '{job_type}'. The supported job types are: {supported_job_types}")

# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import pytest
from libcommon.dtos import Priority
from libcommon.exceptions import PreviousStepFormatError
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import (
    CachedArtifactError,
    CachedArtifactNotFoundError,
    upsert_response,
)

from worker.config import AppConfig
from worker.job_runners.config.info import ConfigInfoJobRunner

from ..utils import REVISION_NAME


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, str, AppConfig], ConfigInfoJobRunner]


CONFIG_INFO_1 = {
    "description": "_DESCRIPTION",
    "citation": "_CITATION",
    "homepage": "_HOMEPAGE",
    "license": "_LICENSE",
    "features": {
        "image": {"_type": "Image"},
        "label": {
            "names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "_type": "ClassLabel",
        },
    },
    "splits": {
        "train": {
            "name": "train",
            "num_bytes": 17470800,
            "num_examples": 60000,
            "dataset_name": "dataset_ok",
        },
        "test": {
            "name": "test",
            "num_bytes": 2916432,
            "num_examples": 10000,
            "dataset_name": "dataset_ok",
        },
    },
    "builder_name": "dataset_ok",
    "config_name": "config_1",
    "version": {"version_str": "0.0.0", "major": 0, "minor": 0, "patch": 0},
    "download_checksums": {
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz": {
            "num_bytes": 9912422,
            "checksum": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        },
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz": {
            "num_bytes": 28881,
            "checksum": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
        },
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz": {
            "num_bytes": 1648877,
            "checksum": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        },
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz": {
            "num_bytes": 4542,
            "checksum": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
        },
    },
    "download_size": 11594722,
    "dataset_size": 20387232,
    "size_in_bytes": 31981954,
}

CONFIG_INFO_2 = {
    "description": "_DESCRIPTION",
    "citation": "_CITATION",
    "homepage": "_HOMEPAGE",
    "license": "_LICENSE",
    "features": {
        "image": {"_type": "Image"},
        "image2": {"_type": "Image"},
        "label": {
            "names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "_type": "ClassLabel",
        },
    },
    "splits": {
        "train": {
            "name": "train",
            "num_bytes": 5678,
            "num_examples": 3000,
            "dataset_name": "dataset_ok",
        },
        "test": {
            "name": "test",
            "num_bytes": 1234,
            "num_examples": 1000,
            "dataset_name": "dataset_ok",
        },
    },
    "builder_name": "dataset_ok",
    "config_name": "config_2",
    "version": {"version_str": "0.0.0", "major": 0, "minor": 0, "patch": 0},
    "download_checksums": {
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz": {
            "num_bytes": 9912422,
            "checksum": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        },
    },
    "download_size": 9912422,
    "dataset_size": 6912,
    "size_in_bytes": 9919334,
}

DATASET_INFO_OK = {
    "config_1": CONFIG_INFO_1,
    "config_2": CONFIG_INFO_2,
}

PARQUET_FILES = [
    {"dataset": "dataset_ok", "config": "config_1", "split": "train", "size": 14281188},
    {"dataset": "dataset_ok", "config": "config_1", "split": "test", "size": 2383903},
    {"dataset": "dataset_ok", "config": "config_2", "split": "train", "size": 1234},
    {"dataset": "dataset_ok", "config": "config_2", "split": "train", "size": 6789},
    {"dataset": "dataset_ok", "config": "config_2", "split": "test", "size": 2383903},
]


@pytest.fixture
def get_job_runner(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigInfoJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        return ConfigInfoJobRunner(
            job_info={
                "type": ConfigInfoJobRunner.get_job_type(),
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
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,config,upstream_status,upstream_content,expected_error_code,expected_content,should_raise",
    [
        (
            "dataset_ok",
            "config_1",
            HTTPStatus.OK,
            {"parquet_files": PARQUET_FILES, "dataset_info": CONFIG_INFO_1, "partial": False},
            None,
            {"dataset_info": CONFIG_INFO_1, "partial": False},
            False,
        ),
        (
            "status_error",
            "config_1",
            HTTPStatus.NOT_FOUND,
            {"error": "error"},
            CachedArtifactError.__name__,
            None,
            True,
        ),
        (
            "format_error",
            "config_1",
            HTTPStatus.OK,
            {"not_dataset_info": "wrong_format"},
            PreviousStepFormatError.__name__,
            None,
            True,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    dataset: str,
    config: str,
    upstream_status: HTTPStatus,
    upstream_content: Any,
    expected_error_code: str,
    expected_content: Any,
    should_raise: bool,
) -> None:
    upsert_response(
        kind="config-parquet-and-info",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        content=upstream_content,
        http_status=upstream_status,
    )
    job_runner = get_job_runner(dataset, config, app_config)
    if should_raise:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.typename == expected_error_code
    else:
        assert job_runner.compute().content == expected_content


def test_doesnotexist(app_config: AppConfig, get_job_runner: GetJobRunner) -> None:
    dataset = config = "doesnotexist"
    job_runner = get_job_runner(dataset, config, app_config)
    with pytest.raises(CachedArtifactNotFoundError):
        job_runner.compute()

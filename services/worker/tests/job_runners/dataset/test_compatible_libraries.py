# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from collections.abc import Callable, Iterator
from http import HTTPStatus
from typing import Any
from unittest.mock import patch

import fsspec
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from libcommon.dtos import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedArtifactError, upsert_response
from pytest import TempPathFactory

from worker.config import AppConfig
from worker.job_runners.dataset.compatible_libraries import (
    LOGIN_COMMENT,
    DatasetCompatibleLibrariesJobRunner,
    get_builder_configs_with_simplified_data_files,
    get_compatible_library_for_builder,
)
from worker.resources import LibrariesResource

from ..utils import REVISION_NAME, UpstreamResponse


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, AppConfig], DatasetCompatibleLibrariesJobRunner]

PARQUET_DATASET = "parquet-dataset"
PARQUET_DATASET_LOGIN_REQUIRED = "parquet-dataset-login_required"
WEBDATASET_DATASET = "webdataset-dataset"
ERROR_DATASET = "error-dataset"

UPSTREAM_RESPONSE_INFO_PARQUET: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=PARQUET_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={"dataset_info": {"default": {"config_name": "default", "builder_name": "parquet"}}, "partial": False},
    progress=1.0,
)
UPSTREAM_RESPONSE_INFO_PARQUET_LOGIN_REQUIRED: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=PARQUET_DATASET_LOGIN_REQUIRED,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={"dataset_info": {"default": {"config_name": "default", "builder_name": "parquet"}}, "partial": False},
    progress=1.0,
)
UPSTREAM_RESPONSE_INFO_WEBDATASET: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=WEBDATASET_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.OK,
    content={"dataset_info": {"default": {"config_name": "default", "builder_name": "webdataset"}}, "partial": False},
    progress=1.0,
)
UPSTREAM_RESPONSE_INFO_ERROR: UpstreamResponse = UpstreamResponse(
    kind="dataset-info",
    dataset=ERROR_DATASET,
    dataset_git_revision=REVISION_NAME,
    http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    content={},
    progress=0.0,
)
EXPECTED_PARQUET = (
    {
        "tags": ["croissant"],
        "formats": ["parquet"],
        "libraries": [
            {
                "function": "load_dataset",
                "language": "python",
                "library": "datasets",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {},
                        "code": ('from datasets import load_dataset\n\nds = load_dataset("parquet-dataset")'),
                    }
                ],
            },
            {
                "function": "pd.read_parquet",
                "language": "python",
                "library": "pandas",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {
                            "splits": {
                                "test": "test.parquet",
                                "train": "train.parquet",
                            }
                        },
                        "code": (
                            "import pandas as pd\n"
                            "\n"
                            "splits = {'train': 'train.parquet', 'test': 'test.parquet'}\n"
                            'df = pd.read_parquet("hf://datasets/parquet-dataset/" + splits["train"])'
                        ),
                    }
                ],
            },
            {
                "function": "Dataset",
                "language": "python",
                "library": "mlcroissant",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {"record_set": "default", "partial": False},
                        "code": (
                            "from mlcroissant "
                            "import Dataset\n"
                            "\n"
                            'ds = Dataset(jsonld="https://huggingface.co/api/datasets/parquet-dataset/croissant")\n'
                            'records = ds.records("default")'
                        ),
                    }
                ],
            },
        ],
    },
    1.0,
)

EXPECTED_PARQUET_LOGIN_REQUIRED = (
    {
        "tags": ["croissant"],
        "formats": ["parquet"],
        "libraries": [
            {
                "function": "load_dataset",
                "language": "python",
                "library": "datasets",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {},
                        "code": (
                            f'from datasets import load_dataset\n{LOGIN_COMMENT}\nds = load_dataset("parquet-dataset-login_required")'
                        ),
                    }
                ],
            },
            {
                "function": "pd.read_parquet",
                "language": "python",
                "library": "pandas",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {
                            "splits": {
                                "test": "test.parquet",
                                "train": "train.parquet",
                            }
                        },
                        "code": (
                            "import pandas as pd\n"
                            f"{LOGIN_COMMENT}\n"
                            "splits = {'train': 'train.parquet', 'test': 'test.parquet'}\n"
                            'df = pd.read_parquet("hf://datasets/parquet-dataset-login_required/" + splits["train"])'
                        ),
                    }
                ],
            },
            {
                "function": "Dataset",
                "language": "python",
                "library": "mlcroissant",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {"record_set": "default", "partial": False},
                        "code": (
                            "import requests\n"
                            "from huggingface_hub.file_download import build_hf_headers\n"
                            "from mlcroissant import Dataset\n"
                            f"{LOGIN_COMMENT}\n"
                            "headers = build_hf_headers()  # handles authentication\n"
                            'jsonld = requests.get("https://huggingface.co/api/datasets/parquet-dataset-login_required/croissant", headers=headers).json()\n'
                            "ds = Dataset(jsonld=jsonld)\n"
                            'records = ds.records("default")'
                        ),
                    }
                ],
            },
        ],
    },
    1.0,
)
EXPECTED_WEBDATASET = (
    {
        "tags": ["croissant"],
        "formats": ["webdataset"],
        "libraries": [
            {
                "function": "load_dataset",
                "language": "python",
                "library": "datasets",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {},
                        "code": ('from datasets import load_dataset\n\nds = load_dataset("webdataset-dataset")'),
                    }
                ],
            },
            {
                "function": "wds.WebDataset",
                "language": "python",
                "library": "webdataset",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {"splits": {"train": "**/*.tar"}},
                        "code": (
                            "import webdataset as wds\n"
                            "from huggingface_hub import HfFileSystem, get_token, hf_hub_url\n"
                            "\n"
                            "fs = HfFileSystem()\n"
                            'files = [fs.resolve_path(path) for path in fs.glob("hf://datasets/webdataset-dataset/**/*.tar")]\n'
                            'urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]\n'
                            "urls = f\"pipe: curl -s -L -H 'Authorization:Bearer {get_token()}' {'::'.join(urls)}\"\n"
                            "\n"
                            "ds = wds.WebDataset(urls).decode()"
                        ),
                    }
                ],
            },
            {
                "function": "Dataset",
                "language": "python",
                "library": "mlcroissant",
                "loading_codes": [
                    {
                        "config_name": "default",
                        "arguments": {"record_set": "default", "partial": False},
                        "code": (
                            "from mlcroissant import Dataset\n"
                            "\n"
                            'ds = Dataset(jsonld="https://huggingface.co/api/datasets/webdataset-dataset/croissant")\n'
                            'records = ds.records("default")'
                        ),
                    }
                ],
            },
        ],
    },
    1.0,
)


@pytest.fixture()
def mock_hffs(tmp_path_factory: TempPathFactory) -> Iterator[fsspec.AbstractFileSystem]:
    hf = tmp_path_factory.mktemp("hf")

    (hf / "datasets" / PARQUET_DATASET).mkdir(parents=True)
    (hf / "datasets" / PARQUET_DATASET / "train.parquet").touch()
    (hf / "datasets" / PARQUET_DATASET / "test.parquet").touch()

    (hf / "datasets" / PARQUET_DATASET_LOGIN_REQUIRED).mkdir(parents=True)
    (hf / "datasets" / PARQUET_DATASET_LOGIN_REQUIRED / "train.parquet").touch()
    (hf / "datasets" / PARQUET_DATASET_LOGIN_REQUIRED / "test.parquet").touch()

    (hf / "datasets" / WEBDATASET_DATASET).mkdir(parents=True)
    (hf / "datasets" / WEBDATASET_DATASET / "0000.tar").touch()
    (hf / "datasets" / WEBDATASET_DATASET / "0001.tar").touch()
    (hf / "datasets" / WEBDATASET_DATASET / "0002.tar").touch()
    (hf / "datasets" / WEBDATASET_DATASET / "0003.tar").touch()

    class MockHfFileSystem(DirFileSystem):  # type: ignore[misc]
        protocol = "hf"

        def __init__(self, path: str = str(hf), target_protocol: str = "local", **kwargs: Any) -> None:
            super().__init__(path=path, target_protocol=target_protocol, **kwargs)
            self.logged_in = kwargs.get("token") != "no_token"

        def isdir(self, path: str, **kwargs: Any) -> bool:
            if "login_required" in path and not self.logged_in:
                return False
            return bool(super().isdir(path, **kwargs))

    HfFileSystem = fsspec.get_filesystem_class("hf")
    fsspec.register_implementation("hf", MockHfFileSystem, clobber=True)
    with patch("worker.job_runners.dataset.compatible_libraries.HfFileSystem", MockHfFileSystem):
        yield MockHfFileSystem()
    fsspec.register_implementation("hf", HfFileSystem, clobber=True)


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetCompatibleLibrariesJobRunner:
        return DatasetCompatibleLibrariesJobRunner(
            job_info={
                "type": DatasetCompatibleLibrariesJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "config": None,
                    "split": None,
                    "revision": REVISION_NAME,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 20,
                "started_at": None,
            },
            app_config=app_config,
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,upstream_responses,expected",
    [
        (
            PARQUET_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_PARQUET,
            ],
            EXPECTED_PARQUET,
        ),
        (
            PARQUET_DATASET_LOGIN_REQUIRED,
            [
                UPSTREAM_RESPONSE_INFO_PARQUET_LOGIN_REQUIRED,
            ],
            EXPECTED_PARQUET_LOGIN_REQUIRED,
        ),
        (
            WEBDATASET_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_WEBDATASET,
            ],
            EXPECTED_WEBDATASET,
        ),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    mock_hffs: fsspec.AbstractFileSystem,
    dataset: str,
    upstream_responses: list[UpstreamResponse],
    expected: Any,
) -> None:
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config)
    compute_result = job_runner.compute()
    assert compute_result.content == expected[0]
    assert compute_result.progress == expected[1]


@pytest.mark.parametrize(
    "dataset,upstream_responses,expectation",
    [
        (
            ERROR_DATASET,
            [
                UPSTREAM_RESPONSE_INFO_ERROR,
            ],
            pytest.raises(CachedArtifactError),
        )
    ],
)
def test_compute_error(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    mock_hffs: fsspec.AbstractFileSystem,
    dataset: str,
    upstream_responses: list[UpstreamResponse],
    expectation: Any,
) -> None:
    for upstream_response in upstream_responses:
        upsert_response(**upstream_response)
    job_runner = get_job_runner(dataset, app_config)
    with expectation:
        job_runner.compute()


@pytest.mark.real_dataset
@pytest.mark.parametrize(
    "dataset,module_name,expected_simplified_data_files",
    [
        (
            "Anthropic/hh-rlhf",
            "json",
            {"default": {"test": ["**/test.jsonl.gz"], "train": ["**/train.jsonl.gz"]}},
        ),
        ("laion/conceptual-captions-12m-webdataset", "webdataset", {"default": {"train": ["**/*.tar"]}}),
        (
            "tatsu-lab/alpaca",
            "parquet",
            {"default": {"train": ["data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"]}},
        ),
        (
            "christopherthompson81/quant_exploration",
            "json",
            {"default": {"train": ["quant_exploration.json"]}},
        ),
        (
            "teknium/openhermes",
            "json",
            {"default": {"train": ["openhermes.json"]}},
        ),
        (
            "cnn_dailymail",
            "parquet",
            {
                "1.0.0": {
                    "test": ["1.0.0/test-00000-of-00001.parquet"],
                    "train": ["1.0.0/train-*.parquet"],
                    "validation": ["1.0.0/validation-00000-of-00001.parquet"],
                },
                "2.0.0": {
                    "test": ["2.0.0/test-00000-of-00001.parquet"],
                    "train": ["2.0.0/train-*.parquet"],
                    "validation": ["2.0.0/validation-00000-of-00001.parquet"],
                },
                "3.0.0": {
                    "test": ["3.0.0/test-00000-of-00001.parquet"],
                    "train": ["3.0.0/train-*.parquet"],
                    "validation": ["3.0.0/validation-00000-of-00001.parquet"],
                },
            },
        ),
        (
            "lmsys/toxic-chat",
            "csv",
            {
                "toxicchat0124": {
                    "test": ["data/0124/toxic-chat_annotation_test.csv"],
                    "train": ["data/0124/toxic-chat_annotation_train.csv"],
                },
                "toxicchat1123": {
                    "test": ["data/1123/toxic-chat_annotation_test.csv"],
                    "train": ["data/1123/toxic-chat_annotation_train.csv"],
                },
            },
        ),
    ],
)
def test_simplify_data_files_patterns(
    use_hub_prod_endpoint: pytest.MonkeyPatch,
    dataset: str,
    module_name: str,
    expected_simplified_data_files: dict[str, dict[str, list[str]]],
) -> None:
    configs = get_builder_configs_with_simplified_data_files(dataset, module_name=module_name)
    simplified_data_files: dict[str, dict[str, list[str]]] = {config.name: config.data_files for config in configs}
    assert simplified_data_files == expected_simplified_data_files


@pytest.mark.integration
@pytest.mark.parametrize(
    "dataset,module_name,expected_data_files,expected_library",
    [
        (
            "rajpurkar/squad",
            "parquet",
            {
                "train": ["plain_text/train-00000-of-00001.parquet"],
                "validation": ["plain_text/validation-00000-of-00001.parquet"],
            },
            "pandas",
        ),
        (
            "Anthropic/hh-rlhf",
            "json",
            {
                "train": ["**/train.jsonl.gz"],
                "test": ["**/test.jsonl.gz"],
            },
            "dask",
        ),
    ],
)
def test_get_builder_configs_with_simplified_data_files(
    use_hub_prod_endpoint: pytest.MonkeyPatch,
    dataset: str,
    module_name: str,
    expected_data_files: dict[str, list[str]],
    expected_library: str,
) -> None:
    hf_token = None
    login_required = False
    configs = get_builder_configs_with_simplified_data_files(dataset, module_name=module_name, hf_token=hf_token)
    assert len(configs) == 1
    config = configs[0]
    assert config.data_files == expected_data_files
    assert module_name in get_compatible_library_for_builder
    compatible_library = get_compatible_library_for_builder[module_name](dataset, hf_token, login_required)
    assert compatible_library["library"] == expected_library

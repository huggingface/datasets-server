# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import os
from collections.abc import Callable, Generator
from dataclasses import replace
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
import pytest
from datasets import Dataset
from datasets.packaged_modules import csv
from fsspec import AbstractFileSystem
from libcommon.dtos import Priority
from libcommon.exceptions import CustomError, TooLongColumnNameError
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.storage import StrPath
from libcommon.storage_client import StorageClient
from libcommon.utils import get_json_size

from worker.config import AppConfig
from worker.job_runners.split.first_rows import SplitFirstRowsJobRunner
from worker.resources import LibrariesResource

from ...constants import ASSETS_BASE_URL
from ...fixtures.hub import HubDatasetTest, get_default_config_split
from ..utils import REVISION_NAME

GetJobRunner = Callable[[str, str, str, AppConfig], SplitFirstRowsJobRunner]


@pytest.fixture
def ds() -> Dataset:
    return Dataset.from_dict({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})


@pytest.fixture
def ds_fs(ds: Dataset, tmpfs: AbstractFileSystem) -> Generator[AbstractFileSystem, None, None]:
    with tmpfs.open("config/train/0000.parquet", "wb") as f:
        ds.to_parquet(f)
    yield tmpfs


@pytest.fixture
def get_job_runner(
    parquet_metadata_directory: StrPath,
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
    tmp_path: Path,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        split: str,
        app_config: AppConfig,
    ) -> SplitFirstRowsJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        upsert_response(
            kind="config-split-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            config=config,
            content={"splits": [{"dataset": dataset, "config": config, "split": split}]},
            http_status=HTTPStatus.OK,
        )

        return SplitFirstRowsJobRunner(
            job_info={
                "type": SplitFirstRowsJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": REVISION_NAME,
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
            parquet_metadata_directory=parquet_metadata_directory,
            storage_client=StorageClient(
                protocol="file",
                storage_root=str(tmp_path / "assets"),
                base_url=ASSETS_BASE_URL,
                overwrite=True,  # all the job runners will overwrite the files
            ),
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "rows_max_bytes,columns_max_number,has_parquet_files,error_code",
    [
        (0, 10, True, "TooBigContentError"),  # too small limit, even with truncation
        (1_000, 1, True, "TooManyColumnsError"),  # too small columns limit
        (1_000, 10, True, None),
        # "ParquetResponseEmptyError" -> triggers first-rows from streaming which fails with a
        # different error ("InfoError")
        (1_000, 10, False, "InfoError"),
    ],
)
def test_compute_from_parquet(
    ds: Dataset,
    ds_fs: AbstractFileSystem,
    parquet_metadata_directory: StrPath,
    get_job_runner: GetJobRunner,
    app_config: AppConfig,
    rows_max_bytes: int,
    columns_max_number: int,
    has_parquet_files: bool,
    error_code: str,
) -> None:
    dataset, config, split = "dataset", "config", "split"
    parquet_file = ds_fs.open("config/train/0000.parquet")
    fake_url = (
        "https://fake.huggingface.co/datasets/dataset/resolve/refs%2Fconvert%2Fparquet/config/train/0000.parquet"
    )
    fake_metadata_subpath = "fake-parquet-metadata/dataset/config/train/0000.parquet"

    config_parquet_metadata_content = {
        "parquet_files_metadata": [
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "url": fake_url,  # noqa: E501
                "filename": "0000.parquet",
                "size": parquet_file.size,
                "num_rows": len(ds),
                "parquet_metadata_subpath": fake_metadata_subpath,
            }
        ]
        if has_parquet_files
        else []
    }

    upsert_response(
        kind="config-parquet-metadata",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        content=config_parquet_metadata_content,
        http_status=HTTPStatus.OK,
    )

    parquet_metadata = pq.read_metadata(ds_fs.open("config/train/0000.parquet"))
    with (
        patch("libcommon.parquet_utils.HTTPFile", return_value=parquet_file) as mock_http_file,
        patch("pyarrow.parquet.read_metadata", return_value=parquet_metadata) as mock_read_metadata,
        patch("pyarrow.parquet.read_schema", return_value=ds.data.schema) as mock_read_schema,
    ):
        job_runner = get_job_runner(
            dataset,
            config,
            split,
            replace(
                app_config,
                common=replace(app_config.common, hf_token=None),
                first_rows=replace(
                    app_config.first_rows,
                    min_number=10,
                    max_bytes=rows_max_bytes,
                    min_cell_bytes=10,
                    columns_max_number=columns_max_number,
                ),
            ),
        )

        if error_code:
            with pytest.raises(CustomError) as error_info:
                job_runner.compute()
            assert error_info.value.code == error_code
        else:
            response = job_runner.compute().content
            assert get_json_size(response) <= rows_max_bytes
            assert response
            assert response["rows"]
            assert response["features"]
            assert len(response["rows"]) == 3  # testing file has 3 rows see config/train/0000.parquet file
            assert len(response["features"]) == 2  # testing file has 2 columns see config/train/0000.parquet file
            assert response["features"][0]["feature_idx"] == 0
            assert response["features"][0]["name"] == "col1"
            assert response["features"][0]["type"]["_type"] == "Value"
            assert response["features"][0]["type"]["dtype"] == "int64"
            assert response["features"][1]["feature_idx"] == 1
            assert response["features"][1]["name"] == "col2"
            assert response["features"][1]["type"]["_type"] == "Value"
            assert response["features"][1]["type"]["dtype"] == "string"
            assert response["rows"][0]["row_idx"] == 0
            assert response["rows"][0]["truncated_cells"] == []
            assert response["rows"][0]["row"] == {"col1": 1, "col2": "a"}
            assert response["rows"][1]["row_idx"] == 1
            assert response["rows"][1]["truncated_cells"] == []
            assert response["rows"][1]["row"] == {"col1": 2, "col2": "b"}
            assert response["rows"][2]["row_idx"] == 2
            assert response["rows"][2]["truncated_cells"] == []
            assert response["rows"][2]["row"] == {"col1": 3, "col2": "c"}

            assert len(mock_http_file.call_args_list) == 1
            assert mock_http_file.call_args_list[0][0][1] == fake_url
            assert len(mock_read_metadata.call_args_list) == 1
            assert mock_read_metadata.call_args_list[0][0][0] == os.path.join(
                parquet_metadata_directory, fake_metadata_subpath
            )
            assert len(mock_read_schema.call_args_list) == 1
            assert mock_read_schema.call_args_list[0][0][0] == os.path.join(
                parquet_metadata_directory, fake_metadata_subpath
            )


@pytest.mark.parametrize(
    "name,use_token,exception_name",
    [
        ("public", False, None),
        ("audio", False, None),
        ("image", False, None),
        ("images_list", False, None),
        ("jsonl", False, None),
        ("gated", True, None),
        ("private", True, None),
        # should we really test the following cases?
        # The assumption is that the dataset exists and is accessible with the token
        ("gated", False, "InfoError"),
        ("private", False, "InfoError"),
    ],
)
def test_number_rows(
    hub_responses_public: HubDatasetTest,
    hub_responses_audio: HubDatasetTest,
    hub_responses_image: HubDatasetTest,
    hub_responses_images_list: HubDatasetTest,
    hub_reponses_jsonl: HubDatasetTest,
    hub_responses_gated: HubDatasetTest,
    hub_responses_private: HubDatasetTest,
    hub_responses_empty: HubDatasetTest,
    hub_responses_does_not_exist_config: HubDatasetTest,
    hub_responses_does_not_exist_split: HubDatasetTest,
    get_job_runner: GetJobRunner,
    name: str,
    use_token: bool,
    exception_name: str,
    app_config: AppConfig,
) -> None:
    # no parquet-metadata entry available -> the job runner will use the streaming approach

    # temporary patch to remove the effect of
    # https://github.com/huggingface/datasets/issues/4875#issuecomment-1280744233
    # note: it fixes the tests, but it does not fix the bug in the "real world"
    if hasattr(csv, "_patched_for_streaming") and csv._patched_for_streaming:
        csv._patched_for_streaming = False

    hub_datasets = {
        "public": hub_responses_public,
        "audio": hub_responses_audio,
        "image": hub_responses_image,
        "images_list": hub_responses_images_list,
        "jsonl": hub_reponses_jsonl,
        "gated": hub_responses_gated,
        "private": hub_responses_private,
        "empty": hub_responses_empty,
        "does_not_exist_config": hub_responses_does_not_exist_config,
        "does_not_exist_split": hub_responses_does_not_exist_split,
    }
    dataset = hub_datasets[name]["name"]
    expected_first_rows_response = hub_datasets[name]["first_rows_response"]
    config, split = get_default_config_split()
    job_runner = get_job_runner(
        dataset,
        config,
        split,
        app_config if use_token else replace(app_config, common=replace(app_config.common, hf_token=None)),
    )

    if exception_name is None:
        job_runner.validate()
        result = job_runner.compute().content
        assert result == expected_first_rows_response
    else:
        with pytest.raises(Exception) as exc_info:
            job_runner.validate()
            job_runner.compute()
        assert exc_info.typename == exception_name


def test_compute(app_config: AppConfig, get_job_runner: GetJobRunner, hub_public_csv: str) -> None:
    # no parquet-metadata entry available -> the job runner will use the streaming approach

    dataset = hub_public_csv
    config, split = get_default_config_split()
    job_runner = get_job_runner(dataset, config, split, app_config)
    response = job_runner.compute()
    assert response
    content = response.content
    assert content
    assert content["features"][0]["feature_idx"] == 0
    assert content["features"][0]["name"] == "col_1"
    assert content["features"][0]["type"]["_type"] == "Value"
    assert content["features"][0]["type"]["dtype"] == "int64"  # <---|
    assert content["features"][1]["type"]["dtype"] == "int64"  # <---|- auto-detected by the datasets library
    assert content["features"][2]["type"]["dtype"] == "float64"  # <-|


@pytest.mark.parametrize(
    "max_column_name_length,raises",
    [
        (1, True),
        (500, False),
    ],
)
def test_long_column_name(
    app_config: AppConfig, get_job_runner: GetJobRunner, hub_public_csv: str, max_column_name_length: int, raises: bool
) -> None:
    # no parquet-metadata entry available -> the job runner will use the streaming approach

    dataset = hub_public_csv
    config, split = get_default_config_split()
    job_runner = get_job_runner(dataset, config, split, app_config)
    with patch("worker.utils.MAX_COLUMN_NAME_LENGTH", max_column_name_length):
        if raises:
            with pytest.raises(TooLongColumnNameError):
                job_runner.compute()
        else:
            job_runner.compute()

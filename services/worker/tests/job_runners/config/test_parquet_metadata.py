# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import io
from collections.abc import Callable, Mapping
from http import HTTPStatus
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset, Features, Value
from fsspec.implementations.http import HTTPFile, HTTPFileSystem
from huggingface_hub import hf_hub_url
from libcommon.constants import PARQUET_REVISION
from libcommon.dtos import Priority, SplitHubFile
from libcommon.exceptions import PreviousStepFormatError
from libcommon.parquet_utils import ParquetIndexWithMetadata, extract_split_directory_from_parquet_url
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedArtifactError, upsert_response
from libcommon.storage import StrPath

from worker.config import AppConfig
from worker.dtos import (
    ConfigParquetMetadataResponse,
    ConfigParquetResponse,
    ParquetFileMetadataItem,
)
from worker.job_runners.config.parquet_metadata import ConfigParquetMetadataJobRunner
from worker.utils import hffs_parquet_url

from ...constants import CI_USER_TOKEN
from ...fixtures.hub import hf_api
from ..utils import REVISION_NAME


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(app_config: AppConfig) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


GetJobRunner = Callable[[str, str, AppConfig], ConfigParquetMetadataJobRunner]


def get_dummy_parquet_buffer() -> io.BytesIO:
    dummy_parquet_buffer = io.BytesIO()
    pq.write_table(pa.table({"a": [0, 1, 2]}), dummy_parquet_buffer)
    return dummy_parquet_buffer


@pytest.fixture
def get_job_runner(
    parquet_metadata_directory: StrPath,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigParquetMetadataJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        return ConfigParquetMetadataJobRunner(
            job_info={
                "type": ConfigParquetMetadataJobRunner.get_job_type(),
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
            parquet_metadata_directory=parquet_metadata_directory,
        )

    return _get_job_runner


@pytest.mark.parametrize(
    "dataset,config,upstream_status,upstream_content,expected_error_code,expected_content,should_raise",
    [
        (
            "ok",
            "config_1",
            HTTPStatus.OK,
            ConfigParquetResponse(
                parquet_files=[
                    SplitHubFile(
                        dataset="ok",
                        config="config_1",
                        split="train",
                        url="https://url1/train/0000.parquet",
                        filename="filename1",
                        size=0,
                    ),
                    SplitHubFile(
                        dataset="ok",
                        config="config_1",
                        split="train",
                        url="https://url1/train/0001.parquet",
                        filename="filename2",
                        size=0,
                    ),
                ],
                partial=False,
                features=None,
            ),
            None,
            ConfigParquetMetadataResponse(
                parquet_files_metadata=[
                    ParquetFileMetadataItem(
                        dataset="ok",
                        config="config_1",
                        split="train",
                        url="https://url1/train/0000.parquet",
                        filename="filename1",
                        size=0,
                        num_rows=3,
                        parquet_metadata_subpath="ok/--/config_1/train/filename1",
                    ),
                    ParquetFileMetadataItem(
                        dataset="ok",
                        config="config_1",
                        split="train",
                        url="https://url1/train/0001.parquet",
                        filename="filename2",
                        size=0,
                        num_rows=3,
                        parquet_metadata_subpath="ok/--/config_1/train/filename2",
                    ),
                ],
                partial=False,
                features=None,
            ),
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
            {"not_parquet_files": "wrong_format"},
            PreviousStepFormatError.__name__,
            None,
            True,
        ),
        (
            "with_features",
            "config_1",
            HTTPStatus.OK,
            ConfigParquetResponse(
                parquet_files=[
                    SplitHubFile(
                        dataset="with_features",
                        config="config_1",
                        split="train",
                        url="https://url1/train/0000.parquet",
                        filename="filename1",
                        size=0,
                    ),
                    SplitHubFile(
                        dataset="with_features",
                        config="config_1",
                        split="train",
                        url="https://url1/train/0001.parquet",
                        filename="filename2",
                        size=0,
                    ),
                ],
                partial=False,
                features=Features({"a": Value("string")}).to_dict(),
            ),
            None,
            ConfigParquetMetadataResponse(
                parquet_files_metadata=[
                    ParquetFileMetadataItem(
                        dataset="with_features",
                        config="config_1",
                        split="train",
                        url="https://url1/train/0000.parquet",
                        filename="filename1",
                        size=0,
                        num_rows=3,
                        parquet_metadata_subpath="with_features/--/config_1/train/filename1",
                    ),
                    ParquetFileMetadataItem(
                        dataset="with_features",
                        config="config_1",
                        split="train",
                        url="https://url1/train/0001.parquet",
                        filename="filename2",
                        size=0,
                        num_rows=3,
                        parquet_metadata_subpath="with_features/--/config_1/train/filename2",
                    ),
                ],
                partial=False,
                features=Features({"a": Value("string")}).to_dict(),
            ),
            False,
        ),
        (
            "more_than_10k_files",
            "config_1",
            HTTPStatus.OK,
            ConfigParquetResponse(
                parquet_files=[
                    SplitHubFile(
                        dataset="more_than_10k_files",
                        config="config_1",
                        split="train",
                        url="https://url1/train-part0/0000.parquet",
                        filename="filename1",
                        size=0,
                    ),
                    SplitHubFile(
                        dataset="more_than_10k_files",
                        config="config_1",
                        split="train",
                        url="https://url1/train-part1/0000.parquet",
                        filename="filename2",
                        size=0,
                    ),
                ],
                partial=False,
                features=Features({"a": Value("string")}).to_dict(),
            ),
            None,
            ConfigParquetMetadataResponse(
                parquet_files_metadata=[
                    ParquetFileMetadataItem(
                        dataset="more_than_10k_files",
                        config="config_1",
                        split="train",
                        url="https://url1/train-part0/0000.parquet",
                        filename="filename1",
                        size=0,
                        num_rows=3,
                        parquet_metadata_subpath="more_than_10k_files/--/config_1/train-part0/filename1",
                    ),
                    ParquetFileMetadataItem(
                        dataset="more_than_10k_files",
                        config="config_1",
                        split="train",
                        url="https://url1/train-part1/0000.parquet",
                        filename="filename2",
                        size=0,
                        num_rows=3,
                        parquet_metadata_subpath="more_than_10k_files/--/config_1/train-part1/filename2",
                    ),
                ],
                partial=False,
                features=Features({"a": Value("string")}).to_dict(),
            ),
            False,
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
        kind="config-parquet",
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
        assert e.type.__name__ == expected_error_code
    else:
        with patch("worker.utils.hf_hub_open_file") as mock_OpenFile:
            # create a new buffer within each test run since the file is closed under the hood in .compute()
            mock_OpenFile.return_value = get_dummy_parquet_buffer()
            content = job_runner.compute().content
            assert content == expected_content
            assert mock_OpenFile.call_count == len(upstream_content["parquet_files"])
            for parquet_file_item in upstream_content["parquet_files"]:
                split_directory = extract_split_directory_from_parquet_url(parquet_file_item["url"])
                path = hffs_parquet_url(dataset, config, split_directory, parquet_file_item["filename"])
                mock_OpenFile.assert_any_call(
                    file_url=path,
                    hf_endpoint=app_config.common.hf_endpoint,
                    hf_token=app_config.common.hf_token,
                    revision=PARQUET_REVISION,
                )
        assert content["parquet_files_metadata"]
        for parquet_file_metadata_item in content["parquet_files_metadata"]:
            assert (
                pq.read_metadata(
                    Path(job_runner.parquet_metadata_directory)
                    / parquet_file_metadata_item["parquet_metadata_subpath"]
                )
                == pq.ParquetFile(get_dummy_parquet_buffer()).metadata
            )


class AuthenticatedHTTPFile(HTTPFile):  # type: ignore
    last_url: Optional[str] = None

    def __init__(  # type: ignore
        self,
        fs,
        url,
        session=None,
        block_size=None,
        mode="rb",
        cache_type="bytes",
        cache_options=None,
        size=None,
        loop=None,
        asynchronous=False,
        **kwargs,
    ) -> None:
        super().__init__(
            fs,
            url,
            session=session,
            block_size=block_size,
            mode=mode,
            cache_type=cache_type,
            cache_options=cache_options,
            size=size,
            loop=loop,
            asynchronous=asynchronous,
            **kwargs,
        )
        assert self.kwargs == {"headers": {"authorization": f"Bearer {CI_USER_TOKEN}"}}
        AuthenticatedHTTPFile.last_url = url


def test_ParquetIndexWithMetadata_query(
    datasets: Mapping[str, Dataset], hub_public_big: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    ds = datasets["big"]
    httpfs = HTTPFileSystem(headers={"authorization": f"Bearer {CI_USER_TOKEN}"})
    filename = next(
        iter(
            repo_file
            for repo_file in hf_api.list_repo_files(repo_id=hub_public_big, repo_type="dataset")
            if repo_file.endswith(".parquet")
        )
    )
    url = hf_hub_url(repo_id=hub_public_big, filename=filename, repo_type="dataset")
    metadata_path = str(tmp_path_factory.mktemp("test_ParquetIndexWithMetadata_query") / "metadata.parquet")
    with httpfs.open(url) as f:
        num_bytes = f.size
        pf = pq.ParquetFile(url, filesystem=httpfs)
        num_rows = pf.metadata.num_rows
        features = Features.from_arrow_schema(pf.schema_arrow)
        pf.metadata.write_metadata_file(metadata_path)
    index = ParquetIndexWithMetadata(
        features=features,
        supported_columns=list(features),
        unsupported_columns=[],
        parquet_files_urls=[url],
        metadata_paths=[metadata_path],
        num_rows=[num_rows],
        num_bytes=[num_bytes],
        httpfs=httpfs,
        hf_token=CI_USER_TOKEN,
        max_arrow_data_in_memory=999999999,
        partial=False,
    )
    with patch("libcommon.parquet_utils.HTTPFile", AuthenticatedHTTPFile):
        out = index.query(offset=0, length=2).to_pydict()
    assert out == ds[:2]
    assert AuthenticatedHTTPFile.last_url == url

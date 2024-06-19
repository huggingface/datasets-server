# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import io
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from fnmatch import fnmatch
from http import HTTPStatus
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional, TypedDict
from unittest.mock import patch

import datasets.builder
import datasets.config
import datasets.info
import pandas as pd
import pyarrow.parquet as pq
import pytest
import requests
from datasets import Audio, Features, Image, Value, load_dataset, load_dataset_builder
from datasets.packaged_modules.generator.generator import (
    Generator as ParametrizedGeneratorBasedBuilder,
)
from datasets.utils.py_utils import asdict
from huggingface_hub.hf_api import CommitOperationAdd, HfApi
from libcommon.dtos import JobInfo, JobParams, Priority
from libcommon.exceptions import (
    DatasetManualDownloadError,
)
from libcommon.queue import Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response

from worker.config import AppConfig
from worker.dtos import CompleteJobResult
from worker.job_manager import JobManager
from worker.job_runners.config.parquet_and_info import (
    ConfigParquetAndInfoJobRunner,
    ParquetFile,
    ParquetFileValidator,
    TooBigRowGroupsError,
    _is_too_big_from_datasets,
    _is_too_big_from_external_data_files,
    _is_too_big_from_hub,
    create_commits,
    fill_builder_info,
    get_delete_operations,
    get_writer_batch_size_from_info,
    get_writer_batch_size_from_row_group_size,
    limit_parquet_writes,
    list_generated_parquet_files,
    parse_repo_filename,
    raise_if_requires_manual_download,
    stream_convert_to_parquet,
)
from worker.job_runners.dataset.config_names import DatasetConfigNamesJobRunner
from worker.resources import LibrariesResource
from worker.utils import resolve_trust_remote_code

from ...constants import CI_HUB_ENDPOINT, CI_USER_TOKEN
from ...fixtures.hub import HubDatasetTest
from ..utils import REVISION_NAME

GetJobRunner = Callable[[str, str, AppConfig], ConfigParquetAndInfoJobRunner]


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigParquetAndInfoJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        return ConfigParquetAndInfoJobRunner(
            job_info={
                "type": ConfigParquetAndInfoJobRunner.get_job_type(),
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
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


def assert_content_is_equal(content: Any, expected: Any) -> None:
    print(content)
    assert set(content) == {"parquet_files", "dataset_info", "partial"}, f"keys: {set(content)}"
    assert content["parquet_files"] == expected["parquet_files"], f"parquet files: {content['parquet_files']}"
    assert len(content["dataset_info"]) == len(
        expected["dataset_info"]
    ), f"length of dataset_info: {content['dataset_info']}"
    content_value = content["dataset_info"]
    expected_value = expected["dataset_info"]
    assert set(content_value.keys()) == set(
        expected_value.keys()
    ), f"keys of dataset_info: {set(content_value.keys())}"
    for key in content_value.keys():
        if key != "download_checksums":
            assert content_value[key] == expected_value[key], f"content of dataset_info['{key}']: {content_value[key]}"
    assert content["partial"] == expected["partial"], f"partial: {content['partial']}"


def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    hub_responses_public: HubDatasetTest,
) -> None:
    dataset = hub_responses_public["name"]
    config = hub_responses_public["config_names_response"]["config_names"][0]["config"]
    job_runner = get_job_runner(dataset, config, app_config)
    response = job_runner.compute()
    assert response
    content = response.content
    assert content
    assert len(content["parquet_files"]) == 1
    assert_content_is_equal(content, hub_responses_public["parquet_and_info_response"])


def test_compute_legacy_configs(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    hub_public_legacy_configs: str,
) -> None:
    app_config = replace(
        app_config, parquet_and_info=replace(app_config.parquet_and_info, max_dataset_size_bytes=20_000)
    )
    app_config = replace(app_config, common=replace(app_config.common, dataset_scripts_allow_list=["*"]))

    dataset_name = hub_public_legacy_configs
    original_configs = {"first", "second"}

    # first compute and push parquet files for each config for dataset with script with two configs
    for config in original_configs:
        job_runner = get_job_runner(dataset_name, config, app_config)
        # needed to overwrite default record when creating job runner
        upsert_response(
            kind="dataset-config-names",
            dataset=hub_public_legacy_configs,
            dataset_git_revision=REVISION_NAME,
            http_status=HTTPStatus.OK,
            content={
                "config_names": [
                    {"dataset": hub_public_legacy_configs, "config": "first"},
                    {"dataset": hub_public_legacy_configs, "config": "second"},
                ],
            },
        )
        assert job_runner.compute()
    hf_api = HfApi(endpoint=CI_HUB_ENDPOINT, token=CI_USER_TOKEN)
    dataset_info = hf_api.dataset_info(
        repo_id=hub_public_legacy_configs, revision=app_config.parquet_and_info.target_revision, files_metadata=False
    )
    repo_files = {f.rfilename for f in dataset_info.siblings}
    # assert that there are only parquet files for dataset's configs and ".gitattributes" in a repo
    # (no files from 'main')
    assert ".gitattributes" in repo_files
    assert all(
        fnmatch(file, "first/*/*.parquet") or fnmatch(file, "second/*/*.parquet")
        for file in repo_files.difference({".gitattributes"})
    )
    orig_repo_configs = {f.rfilename.split("/")[0] for f in dataset_info.siblings if f.rfilename.endswith(".parquet")}
    # assert that both configs are pushed (push of second config didn't delete first config's files)
    assert len(orig_repo_configs) == 2
    assert orig_repo_configs == original_configs
    # then change the set of dataset configs (remove "second")
    job_runner = get_job_runner(dataset_name, "first", app_config)
    assert job_runner.compute()
    dataset_info = hf_api.dataset_info(
        repo_id=hub_public_legacy_configs, revision=app_config.parquet_and_info.target_revision, files_metadata=False
    )
    updated_repo_files = {f.rfilename for f in dataset_info.siblings}
    # assert that legacy config is removed from the repo
    # and there are only files for config that was just pushed and .gitattributes
    assert ".gitattributes" in updated_repo_files
    assert all(fnmatch(file, "first/*/*.parquet") for file in updated_repo_files.difference({".gitattributes"}))
    updated_repo_configs = {
        f.rfilename.split("/")[0] for f in dataset_info.siblings if f.rfilename.endswith(".parquet")
    }
    assert len(updated_repo_configs) == 1
    assert updated_repo_configs == {"first"}


def test_raise_if_requires_manual_download(hub_public_manual_download: str, app_config: AppConfig) -> None:
    builder = load_dataset_builder(hub_public_manual_download)
    with pytest.raises(DatasetManualDownloadError):
        raise_if_requires_manual_download(
            builder=builder,
            hf_endpoint=app_config.common.hf_endpoint,
            hf_token=app_config.common.hf_token,
        )


@pytest.mark.parametrize(
    "name,expected",
    [("public", False), ("big", True)],
)
def test__is_too_big_from_hub(
    hub_public_csv: str,
    hub_public_big: str,
    name: str,
    expected: bool,
    app_config: AppConfig,
) -> None:
    dataset = hub_public_csv if name == "public" else hub_public_big
    dataset_info = HfApi(endpoint=app_config.common.hf_endpoint, token=app_config.common.hf_token).dataset_info(
        repo_id=dataset, revision="main", files_metadata=True
    )
    assert (
        _is_too_big_from_hub(
            dataset_info=dataset_info, max_dataset_size_bytes=app_config.parquet_and_info.max_dataset_size_bytes
        )
        == expected
    )


@pytest.mark.parametrize(
    "name,expected",
    [("public", False), ("big", True)],
)
def test__is_too_big_from_datasets(
    hub_public_csv: str,
    hub_public_big: str,
    name: str,
    expected: bool,
    app_config: AppConfig,
) -> None:
    dataset = hub_public_csv if name == "public" else hub_public_big
    builder = load_dataset_builder(dataset)
    assert (
        _is_too_big_from_datasets(
            info=builder.info,
            max_dataset_size_bytes=app_config.parquet_and_info.max_dataset_size_bytes,
        )
        == expected
    )


@pytest.mark.parametrize(
    "max_dataset_size_bytes,max_external_data_files,expected",
    [
        (None, None, False),
        (10, None, True),
    ],
)
def test__is_too_big_external_files(
    external_files_dataset_builder: "datasets.builder.DatasetBuilder",
    expected: bool,
    max_dataset_size_bytes: Optional[int],
    max_external_data_files: Optional[int],
    app_config: AppConfig,
) -> None:
    max_dataset_size_bytes = max_dataset_size_bytes or app_config.parquet_and_info.max_dataset_size_bytes
    max_external_data_files = max_external_data_files or app_config.parquet_and_info.max_external_data_files
    assert (
        _is_too_big_from_external_data_files(
            builder=external_files_dataset_builder,
            hf_token=app_config.common.hf_token,
            max_dataset_size_bytes=max_dataset_size_bytes,
            max_external_data_files=max_external_data_files,
        )
        == expected
    )


@pytest.mark.parametrize(
    "max_dataset_size_bytes,max_external_data_files,expected",
    [
        (None, None, False),
        (None, 1, True),
    ],
)
def test_raise_if_too_many_external_files(
    external_files_dataset_builder: "datasets.builder.DatasetBuilder",
    expected: bool,
    max_dataset_size_bytes: Optional[int],
    max_external_data_files: Optional[int],
    app_config: AppConfig,
) -> None:
    max_dataset_size_bytes = max_dataset_size_bytes or app_config.parquet_and_info.max_dataset_size_bytes
    max_external_data_files = max_external_data_files or app_config.parquet_and_info.max_external_data_files
    assert (
        _is_too_big_from_external_data_files(
            builder=external_files_dataset_builder,
            hf_token=app_config.common.hf_token,
            max_dataset_size_bytes=max_dataset_size_bytes,
            max_external_data_files=max_external_data_files,
        )
        == expected
    )


def test_supported_if_big_parquet(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    hub_responses_big: HubDatasetTest,
) -> None:
    # Not in the list of supported datasets and bigger than the maximum size
    # but still supported since it's made of parquet files
    # dataset = hub_public_big
    dataset = hub_responses_big["name"]
    config = hub_responses_big["config_names_response"]["config_names"][0]["config"]
    job_runner = get_job_runner(dataset, config, app_config)
    response = job_runner.compute()
    assert response
    content = response.content
    assert content
    assert len(content["parquet_files"]) == 1
    assert_content_is_equal(content, hub_responses_big["parquet_and_info_response"])


def test_partially_converted_if_big_non_parquet(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    hub_responses_big_csv: HubDatasetTest,
) -> None:
    # Not in the list of supported datasets and bigger than the maximum size
    # dataset = hub_public_big_csv
    dataset = hub_responses_big_csv["name"]
    config = hub_responses_big_csv["config_names_response"]["config_names"][0]["config"]
    job_runner = get_job_runner(dataset, config, app_config)
    from datasets.packaged_modules.csv.csv import CsvConfig

    # Set a small chunk size to yield more than one Arrow Table in _generate_tables
    # to be able to stop the generation mid-way.
    with patch.object(CsvConfig, "pd_read_csv_kwargs", {"chunksize": 10}):
        response = job_runner.compute()
    assert response
    content = response.content
    assert content
    assert len(content["parquet_files"]) == 1
    assert_content_is_equal(content, hub_responses_big_csv["parquet_and_info_response"])
    # dataset is partially generated
    assert content["parquet_files"][0]["size"] < app_config.parquet_and_info.max_dataset_size_bytes
    assert content["parquet_files"][0]["url"].endswith("/partial-train/0000.parquet")


def test_supported_if_gated(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    hub_responses_gated: HubDatasetTest,
) -> None:
    # Access must be granted
    dataset = hub_responses_gated["name"]
    config = hub_responses_gated["config_names_response"]["config_names"][0]["config"]
    job_runner = get_job_runner(dataset, config, app_config)
    response = job_runner.compute()
    assert response
    assert response.content


@pytest.mark.parametrize(
    "name",
    ["public", "audio", "gated", "private"],
)
def test_compute_splits_response_simple_csv_ok(
    hub_responses_public: HubDatasetTest,
    hub_responses_audio: HubDatasetTest,
    hub_responses_gated: HubDatasetTest,
    hub_responses_private: HubDatasetTest,
    get_job_runner: GetJobRunner,
    name: str,
    app_config: AppConfig,
    data_df: pd.DataFrame,
) -> None:
    hub_datasets = {
        "public": hub_responses_public,
        "audio": hub_responses_audio,
        "gated": hub_responses_gated,
        "private": hub_responses_private,
        # ^ private dataset can now be accessed as well. The control is not done in the workers anymore.
    }
    dataset = hub_datasets[name]["name"]
    config = hub_datasets[name]["config_names_response"]["config_names"][0]["config"]
    expected_parquet_and_info_response = hub_datasets[name]["parquet_and_info_response"]
    job_runner = get_job_runner(dataset, config, app_config)
    result = job_runner.compute().content
    assert_content_is_equal(result, expected_parquet_and_info_response)

    # download the parquet file and check that it is valid
    if name == "audio":
        return

    if name == "public":
        df = pd.read_parquet(result["parquet_files"][0]["url"], engine="auto")
    else:
        # in all these cases, the parquet files are not accessible without a token
        with pytest.raises(Exception):
            pd.read_parquet(result["parquet_files"][0]["url"], engine="auto")
        r = requests.get(
            result["parquet_files"][0]["url"], headers={"Authorization": f"Bearer {app_config.common.hf_token}"}
        )
        assert r.status_code == HTTPStatus.OK, r.text
        df = pd.read_parquet(io.BytesIO(r.content), engine="auto")
    assert df.equals(data_df), df


@pytest.mark.parametrize(
    "upstream_status,upstream_content,exception_name",
    [
        (HTTPStatus.NOT_FOUND, {"error": "error"}, "CachedArtifactError"),
        (HTTPStatus.OK, {"not_config_names": "wrong_format"}, "PreviousStepFormatError"),
        (HTTPStatus.OK, {"config_names": "not a list"}, "PreviousStepFormatError"),
    ],
)
def test_previous_step_error(
    get_job_runner: GetJobRunner,
    upstream_status: HTTPStatus,
    upstream_content: Any,
    exception_name: str,
    hub_responses_public: HubDatasetTest,
    app_config: AppConfig,
) -> None:
    dataset = hub_responses_public["name"]
    config = hub_responses_public["config_names_response"]["config_names"][0]["config"]
    job_runner = get_job_runner(dataset, config, app_config)
    upsert_response(
        "dataset-config-names",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        http_status=upstream_status,
        content=upstream_content,
    )
    with pytest.raises(Exception) as exc_info:
        job_runner.compute()
    assert exc_info.typename == exception_name


@pytest.mark.parametrize(
    "filename,split,config,raises",
    [
        ("config/split/0000.parquet", "split", "config", False),
        ("config/split.with.dots/0000.parquet", "split.with.dots", "config", False),
        ("config/partial-split/0000.parquet", "split", "config", False),
        ("config/partial-split.with.dots/0000.parquet", "split.with.dots", "config", False),
        ("config/partial-toomanyzeros/00000.parquet", "toomanyzeros", "config", True),
        ("config/builder-split.parquet", "split", "config", True),
        ("plain_text/train/0000.parquet", "train", "plain_text", False),
        ("plain_text/train/0001.parquet", "train", "plain_text", False),
    ],
)
def test_parse_repo_filename(filename: str, split: str, config: str, raises: bool) -> None:
    if raises:
        with pytest.raises(Exception):
            parse_repo_filename(filename)
    else:
        assert parse_repo_filename(filename) == (config, split)


@pytest.mark.parametrize(
    "ds_info, has_big_chunks",
    [
        (datasets.info.DatasetInfo(), False),
        (datasets.info.DatasetInfo(features=Features({"text": Value("string")})), False),
        (datasets.info.DatasetInfo(features=Features({"image": Image()})), True),
        (datasets.info.DatasetInfo(features=Features({"audio": Audio()})), True),
        (datasets.info.DatasetInfo(features=Features({"nested": [{"image": Image()}]})), True),
        (datasets.info.DatasetInfo(features=Features({"blob": Value("binary")})), True),
    ],
)
def test_get_writer_batch_size_from_info(ds_info: datasets.info.DatasetInfo, has_big_chunks: bool) -> None:
    assert get_writer_batch_size_from_info(ds_info) == (100 if has_big_chunks else None)


@pytest.mark.parametrize(
    "max_operations_per_commit,use_parent_commit,expected_num_commits",
    [(2, False, 1), (1, False, 2), (2, True, 1), (1, True, 2)],
)
def test_create_commits(
    hub_public_legacy_configs: str, max_operations_per_commit: int, use_parent_commit: bool, expected_num_commits: int
) -> None:
    NUM_FILES = 2
    repo_id = hub_public_legacy_configs
    hf_api = HfApi(endpoint=CI_HUB_ENDPOINT, token=CI_USER_TOKEN)
    if use_parent_commit:
        target_dataset_info = hf_api.dataset_info(repo_id=repo_id, files_metadata=False)
        parent_commit = target_dataset_info.sha
    else:
        parent_commit = None
    directory = f".test_create_commits_{max_operations_per_commit}_{use_parent_commit}"
    operations: list[CommitOperationAdd] = [
        CommitOperationAdd(path_in_repo=f"{directory}/file{i}.txt", path_or_fileobj=f"content{i}".encode("UTF-8"))
        for i in range(NUM_FILES)
    ]
    commit_infos = create_commits(
        hf_api=hf_api,
        repo_id=repo_id,
        operations=operations,
        commit_message="test",
        max_operations_per_commit=max_operations_per_commit,
        parent_commit=parent_commit,
    )
    assert len(commit_infos) == expected_num_commits
    # check that the files were created
    filenames = hf_api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    for i in range(NUM_FILES):
        assert f"{directory}/file{i}.txt" in filenames


GetDatasetConfigNamesJobRunner = Callable[[str, AppConfig], DatasetConfigNamesJobRunner]


@pytest.fixture
def get_dataset_config_names_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetDatasetConfigNamesJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetConfigNamesJobRunner:
        return DatasetConfigNamesJobRunner(
            job_info={
                "type": DatasetConfigNamesJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": REVISION_NAME,
                    "config": None,
                    "split": None,
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


class JobRunnerArgs(TypedDict):
    dataset: str
    revision: str
    config: str
    app_config: AppConfig
    tmp_path: Path


def launch_job_runner(job_runner_args: JobRunnerArgs) -> CompleteJobResult:
    config = job_runner_args["config"]
    dataset = job_runner_args["dataset"]
    revision = job_runner_args["revision"]
    app_config = job_runner_args["app_config"]
    tmp_path = job_runner_args["tmp_path"]
    job_runner = ConfigParquetAndInfoJobRunner(
        job_info=JobInfo(
            job_id=f"job_{config}",
            type="config-parquet-and-info",
            params=JobParams(dataset=dataset, revision=revision, config=config, split=None),
            priority=Priority.NORMAL,
            difficulty=50,
            started_at=None,
        ),
        app_config=app_config,
        hf_datasets_cache=tmp_path,
    )
    return job_runner.compute()


def test_concurrency(
    hub_public_legacy_n_configs: str,
    app_config: AppConfig,
    tmp_path: Path,
    get_dataset_config_names_job_runner: GetDatasetConfigNamesJobRunner,
    queue_mongo_resource: QueueMongoResource,
    cache_mongo_resource: CacheMongoResource,
) -> None:
    """
    Test that multiple job runners (to compute config-parquet-and-info) can run in parallel,
    without having conflicts when sending commits to the Hub.
    For this test, we need a lot of configs for the same dataset (say 20) and one job runner for each.
    Ideally we would try for both quick and slow jobs.
    """
    app_config = replace(app_config, common=replace(app_config.common, dataset_scripts_allow_list=["*"]))
    repo_id = hub_public_legacy_n_configs
    hf_api = HfApi(endpoint=CI_HUB_ENDPOINT, token=CI_USER_TOKEN)
    revision = hf_api.dataset_info(repo_id=repo_id, files_metadata=False).sha
    if revision is None:
        raise ValueError(f"Could not find revision for dataset {repo_id}")

    # fill the cache for the step dataset-config-names, required by the job_runner
    # it's a lot of code 😅
    job_info = JobInfo(
        job_id="not_used",
        type="dataset-config-names",
        params=JobParams(dataset=repo_id, revision=revision, config=None, split=None),
        priority=Priority.NORMAL,
        difficulty=50,
        started_at=None,
    )
    queue = Queue()
    queue.create_jobs([job_info])
    job_info = queue.start_job(job_types_only=["dataset-config-names"])
    job_manager = JobManager(
        job_info=job_info,
        app_config=app_config,
        job_runner=get_dataset_config_names_job_runner(repo_id, app_config),
    )
    job_result = job_manager.run_job()
    job_manager.finish(job_result=job_result)
    if not job_result["output"]:
        raise ValueError("Could not get config names")
    configs = [str(config_name["config"]) for config_name in job_result["output"]["content"]["config_names"]]

    # launch the job runners
    NUM_JOB_RUNNERS = 10
    with Pool(NUM_JOB_RUNNERS) as pool:
        pool.map(
            launch_job_runner,
            [
                JobRunnerArgs(
                    dataset=repo_id, revision=revision, config=config, app_config=app_config, tmp_path=tmp_path
                )
                for config in configs
            ],
        )


@pytest.mark.parametrize(
    "parquet_files,all_repo_files,config_names,config,deleted_files",
    [
        (
            set(),
            {"dummy", "c1/dummy", "c1/0.parquet", "c2/0.parquet", "c1/index.duckdb"},
            {"c1", "c2"},
            "c1",
            {"dummy", "c1/dummy", "c1/0.parquet"},
        ),
        (
            {"c1/0.parquet"},
            {"dummy", "c1/dummy", "c1/0.parquet", "c2/0.parquet", "c1/index.duckdb"},
            {"c1", "c2"},
            "c1",
            {"dummy", "c1/dummy"},
        ),
    ],
)
def test_get_delete_operations(
    parquet_files: set[str], all_repo_files: set[str], config_names: set[str], config: str, deleted_files: set[str]
) -> None:
    parquet_operations = [
        CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=b"") for path_in_repo in parquet_files
    ]
    delete_operations = get_delete_operations(
        parquet_operations=parquet_operations, all_repo_files=all_repo_files, config_names=config_names, config=config
    )
    assert set(delete_operation.path_in_repo for delete_operation in delete_operations) == deleted_files


@pytest.mark.parametrize(
    "max_dataset_size_bytes,expected_num_shards",
    [
        (1, 1),
        (150, 2),
        (300, 4),
        (9999999, 10),
        (None, 10),
    ],
)
def test_stream_convert_to_parquet_arrowbasedbuilder(
    csv_path: str, max_dataset_size_bytes: int, expected_num_shards: int, tmp_path: Path
) -> None:
    num_data_files = 10
    builder = load_dataset_builder(
        "csv",
        data_files={"train": [csv_path] * num_data_files},
        cache_dir=str(tmp_path / f"test_stream_convert_to_parquet-{max_dataset_size_bytes=}"),
    )
    with patch("worker.job_runners.config.parquet_and_info.get_writer_batch_size_from_info", lambda ds_config_info: 1):
        with patch.object(datasets.config, "MAX_SHARD_SIZE", 1):
            parquet_operations, partial = stream_convert_to_parquet(
                builder, max_dataset_size_bytes=max_dataset_size_bytes
            )
    num_shards = len(parquet_operations)
    assert num_shards == expected_num_shards
    assert partial == (expected_num_shards < num_data_files)
    assert all(isinstance(op.path_or_fileobj, str) for op in parquet_operations)
    parquet_files = list_generated_parquet_files(builder, partial=partial)
    assert len(parquet_files) == expected_num_shards
    assert all(os.path.isfile(parquet_file.local_file) for parquet_file in parquet_files)
    if max_dataset_size_bytes is not None:
        one_sample_max_size = 100
        expected_max_dataset_size_bytes = max_dataset_size_bytes + one_sample_max_size
        assert (
            sum(pq.ParquetFile(parquet_file.local_file).read().nbytes for parquet_file in parquet_files)
            < expected_max_dataset_size_bytes
        )


@pytest.mark.parametrize(
    "max_dataset_size_bytes,expected_num_shards",
    [
        (1, 1),
        (150, 19),
        (300, 38),
        (9999999, 1000),
        (None, 1000),
    ],
)
def test_stream_convert_to_parquet_generatorbasedbuilder(
    max_dataset_size_bytes: int, expected_num_shards: int, tmp_path: Path
) -> None:
    num_rows = 1000

    def long_generator() -> Iterator[dict[str, int]]:
        for i in range(num_rows):
            yield {"foo": i}

    cache_dir = str(tmp_path / "test_limit_parquet_writes_cache_dir")
    builder = ParametrizedGeneratorBasedBuilder(generator=long_generator, cache_dir=cache_dir)
    with patch("worker.job_runners.config.parquet_and_info.get_writer_batch_size_from_info", lambda ds_config_info: 1):
        with patch.object(datasets.config, "MAX_SHARD_SIZE", 1):
            parquet_operations, partial = stream_convert_to_parquet(
                builder, max_dataset_size_bytes=max_dataset_size_bytes
            )
    num_shards = len(parquet_operations)
    assert num_shards == expected_num_shards
    assert partial == (expected_num_shards < num_rows)
    assert all(isinstance(op.path_or_fileobj, str) for op in parquet_operations)
    parquet_files = list_generated_parquet_files(builder, partial=partial)
    assert len(parquet_files) == expected_num_shards
    assert all(os.path.isfile(parquet_file.local_file) for parquet_file in parquet_files)
    if max_dataset_size_bytes is not None:
        one_sample_max_size = 100
        expected_max_dataset_size_bytes = max_dataset_size_bytes + one_sample_max_size
        assert (
            sum(pq.ParquetFile(parquet_file.local_file).read().nbytes for parquet_file in parquet_files)
            < expected_max_dataset_size_bytes
        )


def test_limit_parquet_writes(tmp_path: Path) -> None:
    num_examples = 0

    def long_generator() -> Iterator[dict[str, int]]:
        nonlocal num_examples
        for i in range(10_000_000):
            yield {"foo": i}
            num_examples += 1

    one_sample_size = 8
    max_dataset_size_bytes = 50_000
    expected_max_dataset_size_bytes = max_dataset_size_bytes + datasets.config.DEFAULT_MAX_BATCH_SIZE * one_sample_size
    expected_max_num_examples = 1 + max_dataset_size_bytes // one_sample_size + datasets.config.DEFAULT_MAX_BATCH_SIZE
    cache_dir = str(tmp_path / "test_limit_parquet_writes_cache_dir")
    builder = ParametrizedGeneratorBasedBuilder(generator=long_generator, cache_dir=cache_dir)
    with limit_parquet_writes(builder, max_dataset_size_bytes=max_dataset_size_bytes) as limiter:
        builder.download_and_prepare(file_format="parquet")
        assert builder.info.dataset_size == limiter.total_bytes <= expected_max_dataset_size_bytes
        assert builder.info.splits["train"].num_examples == num_examples < expected_max_num_examples


@pytest.mark.parametrize(
    "validate,too_big_row_groups",
    [
        (None, False),
        (ParquetFileValidator(max_row_group_byte_size=1).validate, True),
        (ParquetFileValidator(max_row_group_byte_size=100_000).validate, False),
    ],
)
def test_fill_builder_info(
    hub_responses_big: HubDatasetTest,
    app_config: AppConfig,
    tmp_path: Path,
    validate: Optional[Callable[[pq.ParquetFile], None]],
    too_big_row_groups: bool,
) -> None:
    cache_dir = str(tmp_path / "test_fill_builder_info")
    name = hub_responses_big["name"]
    builder = load_dataset_builder(name, cache_dir=cache_dir)
    builder.info = datasets.info.DatasetInfo()
    if too_big_row_groups:
        with pytest.raises(TooBigRowGroupsError) as exc_info:
            fill_builder_info(builder, hf_endpoint=app_config.common.hf_endpoint, hf_token=None, validate=validate)
        assert isinstance(exc_info.value, TooBigRowGroupsError)
        assert isinstance(exc_info.value.num_rows, int)
        assert isinstance(exc_info.value.row_group_byte_size, int)
    else:
        fill_builder_info(builder, hf_endpoint=app_config.common.hf_endpoint, hf_token=None, validate=validate)
        expected_info = hub_responses_big["parquet_and_info_response"]["dataset_info"]
        assert expected_info == asdict(builder.info)


def test_fill_builder_info_multiple_parquets(
    hub_public_three_parquet_files_builder: str,
    app_config: AppConfig,
    tmp_path: Path,
    three_parquet_files_paths: list[str],
) -> None:
    cache_dir = str(tmp_path / "test_fill_builder_info_multiple_files")
    name = hub_public_three_parquet_files_builder
    builder = load_dataset_builder(name, cache_dir=cache_dir)
    builder.info = datasets.info.DatasetInfo()
    validate = ParquetFileValidator(max_row_group_byte_size=100_000).validate
    fill_builder_info(builder, hf_endpoint=app_config.common.hf_endpoint, hf_token=None, validate=validate)

    # load dataset with `load_dataset` to generate correct values for dataset info to compare
    dataset = load_dataset("parquet", data_files=three_parquet_files_paths, split="train")
    expected_info = dataset.info
    actual_info = builder.info
    assert expected_info.features == actual_info.features
    assert expected_info.download_size == actual_info.download_size
    # uncompressed dataset size is approximate
    # since we estimate approximate compression ratio by reading only first row group
    assert expected_info.dataset_size == pytest.approx(actual_info.dataset_size, 300)

    assert set(expected_info.splits) == set(actual_info.splits)
    for split_name, split_info in expected_info.splits.items():
        assert split_info.num_examples == actual_info.splits[split_name].num_examples
        assert split_info.num_bytes == pytest.approx(actual_info.splits[split_name].num_bytes, 300)


def test_fill_builder_info_empty_rows(
    hub_public_parquet_splits_empty_rows: str,
    app_config: AppConfig,
    tmp_path: Path,
) -> None:
    cache_dir = str(tmp_path / "test_fill_builder_info_empty_rows")
    name = hub_public_parquet_splits_empty_rows
    builder = load_dataset_builder(name, cache_dir=cache_dir)
    builder.info = datasets.info.DatasetInfo()
    validate = ParquetFileValidator(max_row_group_byte_size=100_000).validate
    fill_builder_info(builder, hf_endpoint=app_config.common.hf_endpoint, hf_token=None, validate=validate)
    assert "train" in builder.info.splits
    assert builder.info.splits["train"].num_examples == 0
    assert builder.info.splits["train"].num_bytes == 0


def test_fill_builder_info_multiple_splits(
    hub_public_three_parquet_splits_builder: str,
    app_config: AppConfig,
    tmp_path: Path,
    three_parquet_splits_paths: Mapping[str, str],
) -> None:
    cache_dir = str(tmp_path / "test_fill_builder_info_multiple_splits")
    name = hub_public_three_parquet_splits_builder
    builder = load_dataset_builder(name, cache_dir=cache_dir)
    builder.info = datasets.info.DatasetInfo()
    validate = ParquetFileValidator(max_row_group_byte_size=100_000).validate
    fill_builder_info(builder, hf_endpoint=app_config.common.hf_endpoint, hf_token=None, validate=validate)

    # load dataset with `load_dataset` to generate correct values for dataset info to compare
    dataset = load_dataset("parquet", data_files=three_parquet_splits_paths)
    actual_info = builder.info
    assert any(expected_info.features == actual_info.features for expected_info in dataset.values())
    assert any(expected_info.download_size == actual_info.download_size for expected_info in dataset.values())
    # uncompressed dataset size is approximate
    # since we estimate approximate compression ratio by reading only first row group
    assert any(
        expected_info.dataset_size == pytest.approx(actual_info.dataset_size, 300)
        for expected_info in dataset.values()
    )

    assert set(dataset.keys()) == set(actual_info.splits)
    assert len(actual_info.splits) == len(three_parquet_splits_paths)
    for split_name, split in dataset.items():
        assert split.info.splits[split_name].num_examples == actual_info.splits[split_name].num_examples
        assert split.info.splits[split_name].num_bytes == pytest.approx(actual_info.splits[split_name].num_bytes, 300)


@pytest.mark.parametrize(
    "num_rows, row_group_byte_size, max_row_group_byte_size, expected",
    [
        (1000, 1000, 500, 100),
        (1000, 1000_000, 500_000, 100),
        (123456789, 123456789, 1000, 100),
        (987654321, 987654321, 1000, 900),
        (1000, 10, 1000, 1000),
        (10, 1000, 1000, 100),
    ],
)
def test_get_writer_batch_size_from_row_group_size(
    num_rows: int, row_group_byte_size: int, max_row_group_byte_size: int, expected: int
) -> None:
    writer_batch_size = get_writer_batch_size_from_row_group_size(
        num_rows=num_rows, row_group_byte_size=row_group_byte_size, max_row_group_byte_size=max_row_group_byte_size
    )
    assert writer_batch_size == expected


def test_resolve_trust_remote_code() -> None:
    assert resolve_trust_remote_code("lhoestq/demo1", allow_list=[]) is False
    assert resolve_trust_remote_code("lhoestq/demo1", allow_list=["lhoestq/d*"]) is True
    assert resolve_trust_remote_code("lhoestq/custom_mnist", allow_list=["lhoestq/d*"]) is False


@pytest.mark.parametrize(
    "config,split,shard_idx,num_shards,partial,expected_path_in_repo",
    [
        ("config", "split", 0, 1, False, "config/split/0000.parquet"),
        ("config", "split", 0, 1, True, "config/partial-split/0000.parquet"),
        ("config", "split", 9999, 10_000, False, "config/split/9999.parquet"),
        ("config", "split", 9999, 10_001, False, "config/split-part0/9999.parquet"),
        ("config", "split", 10_000, 10_001, False, "config/split-part1/0000.parquet"),
        ("config", "split", 99_999, 100_000, False, "config/split-part9/9999.parquet"),
        ("config", "split_ending_with_part1", 0, 1, False, "config/split_ending_with_part1/0000.parquet"),
        ("config", "split", 0, 100_001, False, None),
    ],
)
def test_parquet_file_path_in_repo(
    config: str, split: str, shard_idx: int, num_shards: int, partial: bool, expected_path_in_repo: Optional[str]
) -> None:
    if expected_path_in_repo:
        parquet_file = ParquetFile(
            config=config, split=split, shard_idx=shard_idx, num_shards=num_shards, partial=partial
        )
        assert parquet_file.path_in_repo == expected_path_in_repo
        assert parse_repo_filename(parquet_file.path_in_repo) == (config, split)
    else:
        with pytest.raises(ValueError):
            ParquetFile(config=config, split=split, shard_idx=shard_idx, num_shards=num_shards, partial=partial)

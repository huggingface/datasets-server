# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
from collections.abc import Callable, Mapping
from dataclasses import replace
from http import HTTPStatus
from typing import Optional

import pandas as pd
import polars as pl
import pytest
from datasets import Dataset
from huggingface_hub.hf_api import HfApi
from libcommon.dtos import Priority
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response
from libcommon.storage import StrPath

from worker.config import AppConfig
from worker.job_runners.config.parquet import ConfigParquetJobRunner
from worker.job_runners.config.parquet_and_info import ConfigParquetAndInfoJobRunner
from worker.job_runners.config.parquet_metadata import ConfigParquetMetadataJobRunner
from worker.job_runners.split.descriptive_statistics import SplitDescriptiveStatisticsJobRunner
from worker.resources import LibrariesResource
from worker.statistics_utils import (
    ColumnType,
)

from ...constants import CI_HUB_ENDPOINT, CI_USER_TOKEN
from ...fixtures.hub import HubDatasetTest
from ...test_statistics_utils import (
    count_expected_statistics_for_bool_column,
    count_expected_statistics_for_categorical_column,
    count_expected_statistics_for_list_column,
    count_expected_statistics_for_numerical_column,
    count_expected_statistics_for_string_column,
)
from ..utils import REVISION_NAME

GetJobRunner = Callable[[str, str, str, AppConfig], SplitDescriptiveStatisticsJobRunner]
GetParquetAndInfoJobRunner = Callable[[str, str, AppConfig], ConfigParquetAndInfoJobRunner]
GetParquetJobRunner = Callable[[str, str, AppConfig], ConfigParquetJobRunner]
GetParquetMetadataJobRunner = Callable[[str, str, AppConfig], ConfigParquetMetadataJobRunner]


@pytest.fixture
def get_job_runner(
    parquet_metadata_directory: StrPath,
    statistics_cache_directory: StrPath,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        split: str,
        app_config: AppConfig,
    ) -> SplitDescriptiveStatisticsJobRunner:
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

        return SplitDescriptiveStatisticsJobRunner(
            job_info={
                "type": SplitDescriptiveStatisticsJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": REVISION_NAME,
                    "config": config,
                    "split": split,
                },
                "job_id": "job_id",
                "priority": Priority.NORMAL,
                "difficulty": 100,
                "started_at": None,
            },
            app_config=app_config,
            statistics_cache_directory=statistics_cache_directory,
            parquet_metadata_directory=parquet_metadata_directory,
        )

    return _get_job_runner


@pytest.fixture
def get_parquet_and_info_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetParquetAndInfoJobRunner:
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
                "difficulty": 100,
                "started_at": None,
            },
            app_config=app_config,
            hf_datasets_cache=libraries_resource.hf_datasets_cache,
        )

    return _get_job_runner


@pytest.fixture
def get_parquet_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> GetParquetJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigParquetJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset=dataset,
            dataset_git_revision=REVISION_NAME,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        return ConfigParquetJobRunner(
            job_info={
                "type": ConfigParquetJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
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


@pytest.fixture
def get_parquet_metadata_job_runner(
    libraries_resource: LibrariesResource,
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
    parquet_metadata_directory: StrPath,
) -> GetParquetMetadataJobRunner:
    def _get_job_runner(
        dataset: str,
        config: str,
        app_config: AppConfig,
    ) -> ConfigParquetMetadataJobRunner:
        upsert_response(
            kind="dataset-config-names",
            dataset_git_revision=REVISION_NAME,
            dataset=dataset,
            content={"config_names": [{"dataset": dataset, "config": config}]},
            http_status=HTTPStatus.OK,
        )

        return ConfigParquetMetadataJobRunner(
            job_info={
                "type": ConfigParquetMetadataJobRunner.get_job_type(),
                "params": {
                    "dataset": dataset,
                    "revision": "revision",
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


@pytest.fixture
def descriptive_statistics_expected(datasets: Mapping[str, Dataset]) -> dict:  # type: ignore
    ds = datasets["descriptive_statistics"]
    df = ds.to_pandas()
    expected_statistics = {}
    for column_name in df.columns:
        _type = column_name.split("__")[0]
        if _type == "array":
            continue
        column_type = ColumnType(_type)
        column_data = df[column_name]
        if column_type is ColumnType.STRING_LABEL:
            column_stats = count_expected_statistics_for_string_column(column_data)
        elif column_type in [ColumnType.FLOAT, ColumnType.INT]:
            column_stats = count_expected_statistics_for_numerical_column(column_data, dtype=column_type)
            if (
                column_stats["histogram"]
                and sum(column_stats["histogram"]["hist"]) != df.shape[0] - column_stats["nan_count"]
            ):
                raise ValueError(column_name, column_stats)
        elif column_type is ColumnType.CLASS_LABEL:
            column_stats = count_expected_statistics_for_categorical_column(
                column_data, class_label_feature=ds.features[column_name]
            )
        elif column_type is ColumnType.BOOL:
            column_stats = count_expected_statistics_for_bool_column(column_data)
        elif column_type is ColumnType.LIST:
            column_stats = count_expected_statistics_for_list_column(column_data)
        else:
            raise ValueError
        expected_statistics[column_name] = {
            "column_name": column_name,
            "column_type": column_type,
            "column_statistics": column_stats,
        }
    return {"num_examples": df.shape[0], "statistics": expected_statistics, "partial": False}


@pytest.fixture
def descriptive_statistics_string_text_expected(datasets: Mapping[str, Dataset]) -> dict:  # type: ignore
    ds = datasets["descriptive_statistics_string_text"]
    df = ds.to_pandas()
    expected_statistics = {}
    for column_name in df.columns:
        column_stats = count_expected_statistics_for_string_column(df[column_name])
        if sum(column_stats["histogram"]["hist"]) != df.shape[0] - column_stats["nan_count"]:
            raise ValueError(column_name, column_stats)
        expected_statistics[column_name] = {
            "column_name": column_name,
            "column_type": ColumnType.STRING_TEXT,
            "column_statistics": column_stats,
        }
    return {"num_examples": df.shape[0], "statistics": expected_statistics, "partial": False}


@pytest.fixture
def descriptive_statistics_string_text_partial_expected(datasets: Mapping[str, Dataset]) -> dict:  # type: ignore
    ds = datasets["descriptive_statistics_string_text"]
    df = ds.to_pandas()[:50]  # see `fixtures.hub.hub_public_descriptive_statistics_parquet_builder`
    expected_statistics = {}
    for column_name in df.columns:
        column_stats = count_expected_statistics_for_string_column(df[column_name])
        if sum(column_stats["histogram"]["hist"]) != df.shape[0] - column_stats["nan_count"]:
            raise ValueError(column_name, column_stats)
        expected_statistics[column_name] = {
            "column_name": column_name,
            "column_type": ColumnType.STRING_TEXT,
            "column_statistics": column_stats,
        }
    return {"num_examples": df.shape[0], "statistics": expected_statistics, "partial": True}


@pytest.fixture
def audio_statistics_expected() -> dict:  # type: ignore
    column_names_to_durations = [
        ("audio", [1.0, 2.0, 3.0, 4.0]),  # datasets consists of 4 audio files of 1, 2, 3, 4 seconds lengths
        ("audio_null", [1.0, None, 3.0, None]),  # take first and third audio file for this testcase
        ("audio_all_null", [None, None, None, None]),
    ]
    dataset_statistics = {}
    for column_name, durations in column_names_to_durations:
        statistics = count_expected_statistics_for_numerical_column(
            column=pd.Series(durations), dtype=ColumnType.FLOAT
        )
        column_statistics = {
            "column_name": column_name,
            "column_type": ColumnType.AUDIO,
            "column_statistics": statistics,
        }
        dataset_statistics.update({column_name: column_statistics})
    return {
        "num_examples": 4,
        "statistics": dataset_statistics,
        "partial": False,
    }


@pytest.fixture
def image_statistics_expected() -> dict:  # type: ignore
    column_names_to_widths = [
        ("image", [640, 1440, 520, 1240]),  # datasets consists of 4 image files
        ("image_null", [640, None, 520, None]),  # take first and third image file for this testcase
        ("image_all_null", [None, None, None, None]),
    ]
    dataset_statistics = {}
    for column_name, widths in column_names_to_widths:
        statistics = count_expected_statistics_for_numerical_column(column=pd.Series(widths), dtype=ColumnType.INT)
        column_statistics = {
            "column_name": column_name,
            "column_type": ColumnType.IMAGE,
            "column_statistics": statistics,
        }
        dataset_statistics.update({column_name: column_statistics})
    return {
        "num_examples": 4,
        "statistics": dataset_statistics,
        "partial": False,
    }


@pytest.fixture
def struct_thread_panic_error_parquet_file(tmp_path_factory: pytest.TempPathFactory) -> str:
    repo_id = "__DUMMY_TRANSFORMERS_USER__/test_polars_panic_error"
    hf_api = HfApi(endpoint=CI_HUB_ENDPOINT)

    dir_name = tmp_path_factory.mktemp("data")
    hf_api.hf_hub_download(
        repo_id=repo_id,
        filename="test_polars_panic_error.parquet",
        repo_type="dataset",
        local_dir=dir_name,
        token=CI_USER_TOKEN,
    )
    return str(dir_name / "test_polars_panic_error.parquet")


def test_polars_struct_thread_panic_error(struct_thread_panic_error_parquet_file: str) -> None:
    from polars import (
        Float64,
        List,
        String,
        Struct,
    )

    df = pl.read_parquet(struct_thread_panic_error_parquet_file)  # should not raise
    assert "conversations" in df

    conversations_schema = List(Struct({"from": String, "value": String, "weight": Float64}))
    assert "conversations" in df.schema
    assert df.schema["conversations"] == conversations_schema


@pytest.mark.parametrize(
    "hub_dataset_name,expected_error_code",
    [
        ("descriptive_statistics", None),
        ("descriptive_statistics_string_text", None),
        ("descriptive_statistics_string_text_partial", None),
        ("descriptive_statistics_not_supported", "NoSupportedFeaturesError"),
        ("audio_statistics", None),
        ("image_statistics", None),
        ("gated", None),
    ],
)
def test_compute(
    app_config: AppConfig,
    get_job_runner: GetJobRunner,
    get_parquet_and_info_job_runner: GetParquetAndInfoJobRunner,
    get_parquet_job_runner: GetParquetJobRunner,
    get_parquet_metadata_job_runner: GetParquetMetadataJobRunner,
    hub_responses_descriptive_statistics: HubDatasetTest,
    hub_responses_descriptive_statistics_string_text: HubDatasetTest,
    hub_responses_descriptive_statistics_parquet_builder: HubDatasetTest,
    hub_responses_gated_descriptive_statistics: HubDatasetTest,
    hub_responses_descriptive_statistics_not_supported: HubDatasetTest,
    hub_responses_audio_statistics: HubDatasetTest,
    hub_responses_image_statistics: HubDatasetTest,
    hub_dataset_name: str,
    expected_error_code: Optional[str],
    descriptive_statistics_expected: dict,  # type: ignore
    descriptive_statistics_string_text_expected: dict,  # type: ignore
    descriptive_statistics_string_text_partial_expected: dict,  # type: ignore
    audio_statistics_expected: dict,  # type: ignore
    image_statistics_expected: dict,  # type: ignore
) -> None:
    hub_datasets = {
        "descriptive_statistics": hub_responses_descriptive_statistics,
        "descriptive_statistics_string_text": hub_responses_descriptive_statistics_string_text,
        "descriptive_statistics_string_text_partial": hub_responses_descriptive_statistics_parquet_builder,
        "descriptive_statistics_not_supported": hub_responses_descriptive_statistics_not_supported,
        "gated": hub_responses_gated_descriptive_statistics,
        "audio_statistics": hub_responses_audio_statistics,
        "image_statistics": hub_responses_image_statistics,
    }
    expected = {
        "descriptive_statistics": descriptive_statistics_expected,
        "descriptive_statistics_partial": descriptive_statistics_expected,
        "gated": descriptive_statistics_expected,
        "descriptive_statistics_string_text": descriptive_statistics_string_text_expected,
        "descriptive_statistics_string_text_partial": descriptive_statistics_string_text_partial_expected,
        "audio_statistics": audio_statistics_expected,
        "image_statistics": image_statistics_expected,
    }
    dataset = hub_datasets[hub_dataset_name]["name"]
    splits_response = hub_datasets[hub_dataset_name]["splits_response"]
    config, split = splits_response["splits"][0]["config"], splits_response["splits"][0]["split"]
    expected_response = expected.get(hub_dataset_name)

    partial = hub_dataset_name.endswith("_partial")
    app_config = (
        app_config
        if not partial
        else replace(
            app_config,
            parquet_and_info=replace(
                app_config.parquet_and_info, max_dataset_size_bytes=1, max_row_group_byte_size_for_copy=1
            ),
        )
    )

    # computing and pushing real parquet files because we need them for stats computation
    parquet_and_info_job_runner = get_parquet_and_info_job_runner(dataset, config, app_config)
    parquet_and_info_response = parquet_and_info_job_runner.compute()
    config_parquet_and_info = parquet_and_info_response.content

    assert config_parquet_and_info["partial"] is partial

    upsert_response(
        "config-parquet-and-info",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        http_status=HTTPStatus.OK,
        content=parquet_and_info_response.content,
    )

    parquet_job_runner = get_parquet_job_runner(dataset, config, app_config)
    parquet_response = parquet_job_runner.compute()
    config_parquet = parquet_response.content

    assert config_parquet["partial"] is partial

    upsert_response(
        "config-parquet",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        http_status=HTTPStatus.OK,
        content=config_parquet,
    )

    parquet_metadata_job_runner = get_parquet_metadata_job_runner(dataset, config, app_config)
    parquet_metadata_response = parquet_metadata_job_runner.compute()
    config_parquet_metadata = parquet_metadata_response.content

    assert config_parquet_metadata["partial"] is partial

    upsert_response(
        "config-parquet-metadata",
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        http_status=HTTPStatus.OK,
        content=config_parquet_metadata,
    )
    job_runner = get_job_runner(dataset, config, split, app_config)
    job_runner.pre_compute()
    if expected_error_code:
        with pytest.raises(Exception) as e:
            job_runner.compute()
        assert e.typename == expected_error_code
    else:
        response = job_runner.compute()
        assert sorted(response.content.keys()) == ["num_examples", "partial", "statistics"]
        assert response.content["num_examples"] == expected_response["num_examples"]  # type: ignore
        if hub_dataset_name == "descriptive_statistics_string_text_partial":
            assert response.content["num_examples"] != descriptive_statistics_string_text_expected["num_examples"]

        response = response.content["statistics"]
        expected = expected_response["statistics"]  # type: ignore
        assert len(response) == len(expected), set(expected) - set([res["column_name"] for res in response])  # type: ignore
        assert set([column_response["column_name"] for column_response in response]) == set(  # type: ignore
            expected
        )  # assert returned feature names are as expected

        for column_response in response:  # type: ignore
            expected_column_response = expected[column_response["column_name"]]
            assert column_response["column_name"] == expected_column_response["column_name"]
            assert column_response["column_type"] == expected_column_response["column_type"]
            column_response_stats = column_response["column_statistics"]
            expected_column_response_stats = expected_column_response["column_statistics"]
            assert column_response_stats.keys() == expected_column_response_stats.keys()

            if column_response["column_type"] in [
                ColumnType.FLOAT,
                ColumnType.INT,
                ColumnType.STRING_TEXT,
                ColumnType.LIST,
                ColumnType.AUDIO,
                ColumnType.IMAGE,
            ]:
                hist, expected_hist = (
                    column_response_stats.pop("histogram"),
                    expected_column_response_stats.pop("histogram"),
                )
                if expected_hist:
                    assert hist["hist"] == expected_hist["hist"]
                    assert pytest.approx(hist["bin_edges"]) == expected_hist["bin_edges"]
                assert column_response_stats["nan_count"] == expected_column_response_stats["nan_count"]
                assert pytest.approx(column_response_stats) == expected_column_response_stats
                if column_response["column_type"] is ColumnType.INT:
                    assert column_response_stats["min"] == expected_column_response_stats["min"]
                    assert column_response_stats["max"] == expected_column_response_stats["max"]
            elif column_response["column_type"] in [ColumnType.STRING_LABEL, ColumnType.CLASS_LABEL]:
                assert column_response_stats == expected_column_response_stats
            elif column_response["column_type"] is ColumnType.BOOL:
                assert pytest.approx(
                    column_response_stats.pop("nan_proportion")
                ) == expected_column_response_stats.pop("nan_proportion")
                assert column_response_stats == expected_column_response_stats
            else:
                raise ValueError("Incorrect data type")

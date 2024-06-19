# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.


from collections.abc import Callable

import pytest
from huggingface_hub.hf_api import RepoSibling
from libcommon.dtos import Priority

from worker.config import AppConfig
from worker.dtos import Filetype
from worker.job_runners.dataset.filetypes import DatasetFiletypesJobRunner, get_filetypes, get_filetypes_from_archives
from worker.resources import LibrariesResource

from ..utils import REVISION_NAME


@pytest.mark.parametrize(
    "siblings,expected_filetypes",
    [
        (
            [
                RepoSibling("file1.txt"),
                RepoSibling("file2.txt"),
            ],
            [
                Filetype(extension=".txt", count=2),
            ],
        ),
        (
            [
                RepoSibling("file1.txt"),
                RepoSibling("file2.tXt"),
                RepoSibling("file2.TXT"),
            ],
            [
                Filetype(extension=".txt", count=3),
            ],
        ),
        (
            [
                RepoSibling("file1.txt"),
                RepoSibling("file2.txt"),
                RepoSibling("file3.csv"),
                RepoSibling("file3.tar"),
                RepoSibling("file3.tar.gz"),
                RepoSibling("file3.tar.gz_1"),
                RepoSibling("file3.tar.gz-1"),
                RepoSibling("file3.tar.gz?dl=1-1"),
                RepoSibling("file.gz"),
                RepoSibling("file.json.gz"),
                RepoSibling(".gitignore"),
            ],
            [
                Filetype(extension=".txt", count=2),
                Filetype(extension=".csv", count=1),
                Filetype(extension=".tar", count=1),
                Filetype(extension=".gz", count=6),
                Filetype(extension=".tar", count=4, compressed_in=".gz"),
                Filetype(extension=".json", count=1, compressed_in=".gz"),
                Filetype(extension="", count=1),
            ],
        ),
    ],
)
def test_get_filetypes(siblings: list[RepoSibling], expected_filetypes: list[Filetype]) -> None:
    filetypes = get_filetypes(siblings)
    assert filetypes == expected_filetypes


@pytest.mark.real_dataset
@pytest.mark.parametrize(
    "dataset,archive_filenames,filetypes",
    [
        (
            "severo/LILA",
            ["data/Caltech_Camera_Traps.jsonl.zip"],
            [
                Filetype(extension=".jsonl", count=1, archived_in=".zip"),
            ],
        ),
        (
            "severo/winogavil",
            ["winogavil_images.zip"],
            [
                Filetype(extension=".jpg", count=2044, archived_in=".zip"),
            ],
        ),
    ],
)
def test_get_filetypes_from_archive(
    use_hub_prod_endpoint: pytest.MonkeyPatch,
    dataset: str,
    archive_filenames: list[str],
    filetypes: list[Filetype],
) -> None:
    assert (
        get_filetypes_from_archives(dataset=dataset, archive_filenames=archive_filenames, hf_token=None) == filetypes
    )


GetJobRunner = Callable[[str, AppConfig], DatasetFiletypesJobRunner]


@pytest.fixture
def get_job_runner(
    libraries_resource: LibrariesResource,
) -> GetJobRunner:
    def _get_job_runner(
        dataset: str,
        app_config: AppConfig,
    ) -> DatasetFiletypesJobRunner:
        return DatasetFiletypesJobRunner(
            job_info={
                "type": DatasetFiletypesJobRunner.get_job_type(),
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


@pytest.mark.real_dataset
@pytest.mark.parametrize(
    "dataset,filetypes",
    [
        (
            "severo/LILA",
            [
                Filetype(extension="", count=1),
                Filetype(extension=".py", count=1),
                Filetype(extension=".zip", count=18),
                Filetype(extension=".md", count=1),
                Filetype(extension=".jsonl", count=18, archived_in=".zip"),
            ],
        ),
        (
            "severo/winogavil",
            [
                Filetype(extension="", count=1),
                Filetype(extension=".md", count=1),
                Filetype(extension=".py", count=1),
                Filetype(extension=".csv", count=1),
                Filetype(extension=".zip", count=1),
                Filetype(extension=".jpg", count=2044, archived_in=".zip"),
            ],
        ),
    ],
)
def test_compute(
    app_config_prod: AppConfig,
    use_hub_prod_endpoint: pytest.MonkeyPatch,
    dataset: str,
    filetypes: list[Filetype],
    get_job_runner: GetJobRunner,
) -> None:
    job_runner = get_job_runner(dataset, app_config_prod)
    response = job_runner.compute()
    content = response.content
    assert content["filetypes"] == filetypes

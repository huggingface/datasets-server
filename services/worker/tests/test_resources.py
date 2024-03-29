# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import datasets
import pytest
from pytest import TempPathFactory

from worker.resources import LibrariesResource


@pytest.mark.parametrize(
    "define_init_hf_datasets_cache,define_numba_path",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_libraries(
    tmp_path_factory: TempPathFactory, define_init_hf_datasets_cache: bool, define_numba_path: bool
) -> None:
    hf_endpoint = "https://another.endpoint"
    init_hf_datasets_cache = (
        str(tmp_path_factory.mktemp("hf_datasets_cache")) if define_init_hf_datasets_cache else None
    )
    numba_path = str(tmp_path_factory.mktemp("numba_path")) if define_numba_path else None
    assert datasets.config.HF_ENDPOINT != hf_endpoint
    resource = LibrariesResource(
        hf_endpoint=hf_endpoint, init_hf_datasets_cache=init_hf_datasets_cache, numba_path=numba_path
    )
    assert datasets.config.HF_ENDPOINT == hf_endpoint
    assert not datasets.config.HF_UPDATE_DOWNLOAD_COUNTS
    assert (str(resource.hf_datasets_cache) == init_hf_datasets_cache) == define_init_hf_datasets_cache

    resource.release()

    assert datasets.config.HF_ENDPOINT != hf_endpoint


def test_libraries_context_manager(tmp_path_factory: TempPathFactory) -> None:
    hf_endpoint = "https://another.endpoint"
    init_hf_datasets_cache = str(tmp_path_factory.mktemp("hf_datasets_cache"))
    numba_path = str(tmp_path_factory.mktemp("numba_path"))
    assert datasets.config.HF_ENDPOINT != hf_endpoint
    with LibrariesResource(
        hf_endpoint=hf_endpoint, init_hf_datasets_cache=init_hf_datasets_cache, numba_path=numba_path
    ):
        assert datasets.config.HF_ENDPOINT == hf_endpoint
    assert datasets.config.HF_ENDPOINT != hf_endpoint

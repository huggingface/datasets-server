# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import shutil
from os import PathLike, makedirs
from pathlib import Path
from typing import Optional, Union

from appdirs import user_cache_dir  # type:ignore

from libcommon.constants import (
    DESCRIPTIVE_STATISTICS_CACHE_APPNAME,
    DUCKDB_INDEX_CACHE_APPNAME,
    HF_DATASETS_CACHE_APPNAME,
    PARQUET_METADATA_CACHE_APPNAME,
)

StrPath = Union[str, PathLike[str]]


def init_dir(directory: Optional[StrPath] = None, appname: Optional[str] = None) -> StrPath:
    """Initialize a directory.

    If directory is None, it will be set to the default cache location on the machine (using appname as a key, if
    not None).

    Args:
        directory (`StrPath`, *optional*): The directory to initialize. Defaults to None.
        appname (`str`, *optional*): The name of the application. Used if `directory`is None. Defaults to None.

    Returns:
        `StrPath`: The directory.
    """
    if directory is None:
        directory = user_cache_dir(appname=appname)
        logging.debug(f"Directory defaulting to user-specific cache: {directory}")
    makedirs(directory, exist_ok=True)
    logging.debug(f"Directory created at: {directory}")
    return directory


def init_parquet_metadata_dir(directory: Optional[StrPath] = None) -> StrPath:
    """Initialize the parquet metadata directory.

    If directory is None, it will be set to the default cache location on the machine.

    Args:
        directory (`StrPath`, *optional*): The directory to initialize. Defaults to None.

    Returns:
        `StrPath`: The directory.
    """
    return init_dir(directory, appname=PARQUET_METADATA_CACHE_APPNAME)


def init_duckdb_index_cache_dir(directory: Optional[StrPath] = None) -> StrPath:
    """Initialize the duckdb index directory.

    If directory is None, it will be set to the default duckdb index location on the machine.

    Args:
        directory (`StrPath`, *optional*): The directory to initialize. Defaults to None.

    Returns:
        `StrPath`: The directory.
    """
    return init_dir(directory, appname=DUCKDB_INDEX_CACHE_APPNAME)


def init_hf_datasets_cache_dir(directory: Optional[StrPath] = None) -> StrPath:
    """Initialize the cache directory for the datasets library.

    If directory is None, it will be set to the default cache location on the machine.

    Args:
        directory (`StrPath`, *optional*): The directory to initialize. Defaults to None.

    Returns:
        `StrPath`: The directory.
    """
    return init_dir(directory, appname=HF_DATASETS_CACHE_APPNAME)


def init_statistics_cache_dir(directory: Optional[StrPath] = None) -> StrPath:
    """Initialize the cache directory for storage of a dataset in parquet format for statistics computations.

    If directory is None, it will be set to the default cache location on the machine.

    Args:
        directory (`StrPath`, *optional*): The directory to initialize. Defaults to None.

    Returns:
        `StrPath`: The directory.
    """
    return init_dir(directory, appname=DESCRIPTIVE_STATISTICS_CACHE_APPNAME)


def exists(path: StrPath) -> bool:
    """Check if a path exists.

    Args:
        path (`StrPath`): The path to check.

    Returns:
        `bool`: True if the path exists, False otherwise.
    """
    return Path(path).exists()


def remove_dir(directory: StrPath) -> None:
    """Remove a directory.

    If the directory does not exist, don't raise.

    Args:
        directory (`StrPath`): The directory to remove.
    """
    shutil.rmtree(directory, ignore_errors=True)
    logging.debug(f"Directory removed: {directory}")

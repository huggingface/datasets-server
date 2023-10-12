# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import glob
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import Any

from libcommon.storage import StrPath


def clean_hf_datasets_cache(hf_datasets_cache: StrPath, expired_time_interval_seconds: int) -> None:
    """
    Delete temporary cache directories from job runners with datasets cache.
    """
    # sanity check
    if len(str(hf_datasets_cache)) < 10:
        raise RuntimeError(f"Sanity check on hf_datasets_cache failed: len('{hf_datasets_cache}') < 10.")
    logging.info("delete hf datasets cache")
    # path is like {hf_datasets_cache}/{deployName}/datasets
    pattern = f"{hf_datasets_cache}/*/datasets/*"
    logging.info(f"looking for all files and directories with pattern {pattern}")
    now = datetime.now().replace(tzinfo=None)
    errors = 0
    total_dirs = 0
    total_files = 0

    def rmtree_on_error(function: Any, path: str, excinfo: Any) -> None:  # noqa: U100, args needed for onerror=
        logging.error(f"failed to delete directory {path=}")

    for path in glob.glob(pattern):
        last_access_time_value = os.path.getatime(path)
        last_access_datetime = datetime.fromtimestamp(last_access_time_value).replace(tzinfo=None)
        if last_access_datetime + timedelta(seconds=expired_time_interval_seconds) <= now:
            if os.path.isfile(path):
                logging.info(f"deleting file {path=} {last_access_datetime=}")
                os.remove(path)
                total_files += 1
            elif os.path.isdir(path):
                logging.info(f"deleting directory {path=} {last_access_datetime=}")
                shutil.rmtree(path, onerror=rmtree_on_error)
                total_dirs += 1
                errors += os.path.isdir(path)
    if errors:
        logging.error(
            f"clean_hf_datasets_cache failed to remove {errors} directories at the root of the cache directory."
        )
    if total_files:
        logging.info(f"clean_hf_datasets_cache removed {total_files} files at the root of the cache directory.")

    logging.info(
        f"clean_hf_datasets_cache removed {total_dirs - errors} directories at the root of the cache directory."
    )
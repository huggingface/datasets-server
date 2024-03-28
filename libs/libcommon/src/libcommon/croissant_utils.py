# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from collections.abc import Mapping
from typing import Any


def get_record_set(dataset: str, config_name: str) -> str:
    # Identical keys are not supported in Croissant
    # The current workaround that is used in /croissant endpoint
    # is to prefix the config name with `record_set_` if necessary.
    if dataset != config_name:
        return config_name
    else:
        return f"record_set_{config_name}"


MAX_COLUMNS = 1_000
# ^ same value as the default for FIRST_ROWS_COLUMNS_MAX_NUMBER (see services/worker)


def truncate_features_from_croissant_crumbs_response(content: Mapping[str, Any]) -> None:
    """Truncate the features from a croissant-crumbs response to avoid returning a large response."""
    if isinstance(content, dict) and "recordSet" in content and isinstance(content["recordSet"], list):
        for record in content["recordSet"]:
            if (
                isinstance(record, dict)
                and "field" in record
                and isinstance(record["field"], list)
                and len(record["field"]) > MAX_COLUMNS
            ):
                num_columns = len(record["field"])
                record["field"] = record["field"][:MAX_COLUMNS]
                record[
                    "description"
                ] += f"\n- {num_columns - MAX_COLUMNS} skipped column{'s' if num_columns - MAX_COLUMNS > 1 else ''} (max number of columns reached)"

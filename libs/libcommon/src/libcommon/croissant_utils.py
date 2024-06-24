# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from collections.abc import Mapping
from typing import Any

from datasets import ClassLabel, Image, Sequence, Value


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


HF_TO_CROISSANT_VALUE_TYPE = {
    "large_string": "sc:Text",
    "string": "sc:Text",
    "int16": "sc:Integer",
    "int32": "sc:Integer",
    "int64": "sc:Integer",
    "float8": "sc:Float",
    "float16": "sc:Float",
    "float32": "sc:Float",
    "float64": "sc:Float",
    "bool": "sc:Boolean",
}


def feature_to_croissant_field(distribution_name: str, field_name: str, column: str, feature) -> dict[str, Any] | None:
    """Converts a Hugging Face Datasets feature to a Croissant field or None if impossible."""
    if isinstance(feature, Value) and feature.dtype in HF_TO_CROISSANT_VALUE_TYPE:
        return {
            "@type": "cr:Field",
            "@id": field_name,
            "name": field_name,
            "description": f"Column '{column}' from the Hugging Face parquet file.",
            "dataType": HF_TO_CROISSANT_VALUE_TYPE[feature.dtype],
            "source": {"fileSet": {"@id": distribution_name}, "extract": {"column": column}},
        }
    elif isinstance(feature, Image):
        return {
            "@type": "cr:Field",
            "@id": field_name,
            "name": field_name,
            "description": f"Image column '{column}' from the Hugging Face parquet file.",
            "dataType": "sc:ImageObject",
            "source": {
                "fileSet": {"@id": distribution_name},
                "extract": {"column": column},
                "transform": {"jsonPath": "bytes"},
            },
        }
    elif isinstance(feature, ClassLabel):
        return {
            "@type": "cr:Field",
            "@id": field_name,
            "name": field_name,
            "description": f"ClassLabel column '{column}' from the Hugging Face parquet file.\nLabels:\n"
            + ", ".join(f"{name} ({i})" for i, name in enumerate(feature.names)),
            "dataType": "sc:Integer",
            "source": {"fileSet": {"@id": distribution_name}, "extract": {"column": column}},
        }
    elif isinstance(feature, Sequence):
        sub_feature = feature.feature
        field = feature_to_croissant_field(distribution_name, field_name, column, sub_feature)
        if field:
            field["repeated"] = True
            return field
    return None

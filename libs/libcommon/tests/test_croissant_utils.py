# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.


from unittest.mock import patch

from datasets import Sequence, Value
import pytest

from libcommon.croissant_utils import feature_to_croissant_field, truncate_features_from_croissant_crumbs_response


@pytest.mark.parametrize("num_columns", [1, 3])
def test_truncate_features_from_croissant_crumbs_response(num_columns: int) -> None:
    content = {
        "recordSet": [
            {
                "field": [{"name": f"col_{i}", "type": "string"} for i in range(num_columns)],
                "description": "description",
            }
        ]
    }
    with patch("libcommon.croissant_utils.MAX_COLUMNS", 2):
        truncate_features_from_croissant_crumbs_response(content)
    if num_columns <= 2:
        assert len(content["recordSet"][0]["field"]) == num_columns
    else:
        assert len(content["recordSet"][0]["field"]) == 2
        assert "max number of columns reached" in content["recordSet"][0]["description"]


@pytest.mark.parametrize(("hf_datasets_feature", "croissant_field"), [
    (
        Sequence(Value(dtype='int32')),
        {
            "@type": "cr:Field",
            "@id": "field_name",
            "name": "field_name",
            "description": "Column 'column_name' from the Hugging Face parquet file.",
            "dataType": "sc:Integer",
            "source": {"fileSet": {"@id": "distribution_name"}, "extract": {"column": "column_name"}},
            "repeated": True,
        }
    )
])
def test_feature_to_croissant_field(hf_datasets_feature, croissant_field) -> None:
    assert feature_to_croissant_field("distribution_name", "field_name", "column_name", hf_datasets_feature) == croissant_field

# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.


from unittest.mock import patch

import pytest

from libcommon.croissant_utils import truncate_features_from_croissant_crumbs_response


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

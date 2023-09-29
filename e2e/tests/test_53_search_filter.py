from .fixtures.hub import AuthHeaders, AuthType
from .utils import get_default_config_split, poll_until_ready_and_assert


def test_filter_endpoint(
    auth_headers: AuthHeaders,
    hf_public_dataset_repo_csv_data: str,
) -> None:
    auth: AuthType = "none"
    expected_status_code: int = 200
    expected_error_code = None
    # TODO: add dataset with various splits, or various configs
    dataset = hf_public_dataset_repo_csv_data
    config, split = get_default_config_split()
    headers = auth_headers[auth]
    offset = 1
    length = 2
    where = "col_4 = 'B'"
    filter_response = poll_until_ready_and_assert(
        relative_url=(
            f"/filter?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}&where={where}"
        ),
        expected_status_code=expected_status_code,
        expected_error_code=expected_error_code,
        headers=headers,
        check_x_revision=True,
    )
    if not expected_error_code:
        content = filter_response.json()
        assert "rows" in content, filter_response
        assert "features" in content, filter_response
        assert "num_rows_total" in content, filter_response
        assert "num_rows_per_page" in content, filter_response
        rows = content["rows"]
        features = content["features"]
        num_rows_total = content["num_rows_total"]
        num_rows_per_page = content["num_rows_per_page"]
        assert isinstance(rows, list), rows
        assert isinstance(features, list), features
        assert num_rows_total == 3
        assert num_rows_per_page == 100
        assert rows[0] == {
            "row_idx": 1,
            "row": {
                "col_1": "Vader turns round and round in circles as his ship spins into space.",
                "col_2": 1,
                "col_3": 1.0,
                "col_4": "B",
            },
            "truncated_cells": [],
        }, rows[0]
        assert rows[1] == {
            "row_idx": 2,
            "row": {
                "col_1": "The wingman spots the pirateship coming at him and warns the Dark Lord",
                "col_2": 3,
                "col_3": 3.0,
                "col_4": "B",
            },
            "truncated_cells": [],
        }, rows[1]
        assert features == [
            {"feature_idx": 0, "name": "col_1", "type": {"dtype": "string", "_type": "Value"}},
            {"feature_idx": 1, "name": "col_2", "type": {"dtype": "int64", "_type": "Value"}},
            {"feature_idx": 2, "name": "col_3", "type": {"dtype": "float64", "_type": "Value"}},
            {"feature_idx": 3, "name": "col_4", "type": {"dtype": "string", "_type": "Value"}},
        ], features
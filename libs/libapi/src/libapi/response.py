import logging

import pyarrow as pa
from datasets import Features
from libcommon.storage_client import StorageClient
from libcommon.storage_options import StorageOptions
from libcommon.utils import MAX_NUM_ROWS_PER_PAGE, PaginatedResponse
from libcommon.viewer_utils.features import to_features_list

from libapi.utils import to_rows_list

ROW_IDX_COLUMN = "__hf_index_id"


def create_response(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    cached_assets_base_url: str,
    storage_client: StorageClient,
    pa_table: pa.Table,
    offset: int,
    features: Features,
    unsupported_columns: list[str],
    num_rows_total: int,
    use_row_idx_column: bool = False,
) -> PaginatedResponse:
    if set(pa_table.column_names).intersection(set(unsupported_columns)):
        raise RuntimeError(
            "The pyarrow table contains unsupported columns. They should have been ignored in the row group reader."
        )
    logging.debug(f"create response for {dataset=} {config=} {split=}")
    storage_options = StorageOptions(
        assets_base_url=cached_assets_base_url,
        overwrite=False,
        storage_client=storage_client,
    )
    return {
        "features": to_features_list(features),
        "rows": to_rows_list(
            pa_table=pa_table,
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            storage_options=storage_options,
            offset=offset,
            features=features,
            unsupported_columns=unsupported_columns,
            row_idx_column=ROW_IDX_COLUMN if use_row_idx_column else None,
        ),
        "num_rows_total": num_rows_total,
        "num_rows_per_page": MAX_NUM_ROWS_PER_PAGE,
    }

# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

ASSETS_CACHE_APPNAME = "datasets_server_assets"
CACHED_ASSETS_CACHE_APPNAME = "datasets_server_cached_assets"
CACHE_MONGOENGINE_ALIAS = "cache"
QUEUE_MONGOENGINE_ALIAS = "queue"
METRICS_MONGOENGINE_ALIAS = "metrics"
QUEUE_TTL_SECONDS = 604_800  # 7 days

PROCESSING_STEP_CONFIG_NAMES_VERSION = 1
PROCESSING_STEP_CONFIG_PARQUET_VERSION = 4
PROCESSING_STEP_CONFIG_SIZE_VERSION = 2
PROCESSING_STEP_CONFIG_INFO_VERSION = 2
PROCESSING_STEP_DATASET_INFO_VERSION = 2
PROCESSING_STEP_DATASET_IS_VALID_VERSION = 2
PROCESSING_STEP_DATASET_PARQUET_VERSION = 2
PROCESSING_STEP_DATASET_SIZE_VERSION = 2
PROCESSING_STEP_DATASET_SPLIT_NAMES_FROM_DATASET_INFO_VERSION = 2
PROCESSING_STEP_DATASET_SPLIT_NAMES_FROM_STREAMING_VERSION = 2
PROCESSING_STEP_PARQUET_AND_DATASET_INFO_VERSION = 2
PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_PARQUET_VERSION = 2
PROCESSING_STEP_CONFIG_PARQUET_AND_INFO_VERSION = 2
PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION = 3
PROCESSING_STEP_SPLIT_NAMES_FROM_DATASET_INFO_VERSION = 3
PROCESSING_STEP_SPLIT_NAMES_FROM_STREAMING_VERSION = 3
PROCESSING_STEP_DATASET_SPLIT_NAMES_VERSION = 2

PROCESSING_STEP_PARQUET_AND_DATASET_INFO_ROW_GROUP_SIZE_FOR_AUDIO_DATASETS = 100
PROCESSING_STEP_PARQUET_AND_DATASET_INFO_ROW_GROUP_SIZE_FOR_IMAGE_DATASETS = 100
PARQUET_REVISION = "refs/convert/parquet"

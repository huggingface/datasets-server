# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Mapping
from dataclasses import dataclass, field

from libapi.config import ApiConfig
from libcommon.config import (
    AssetsConfig,
    CacheConfig,
    CachedAssetsConfig,
    CloudFrontConfig,
    CommonConfig,
    LogConfig,
    QueueConfig,
    S3Config,
)
from libcommon.processing_graph import InputType


@dataclass(frozen=True)
class AppConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    cached_assets: CachedAssetsConfig = field(default_factory=CachedAssetsConfig)
    cloudfront: CloudFrontConfig = field(default_factory=CloudFrontConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    log: LogConfig = field(default_factory=LogConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    s3: S3Config = field(default_factory=S3Config)

    @classmethod
    def from_env(cls) -> "AppConfig":
        common_config = CommonConfig.from_env()
        return cls(
            common=common_config,
            assets=AssetsConfig.from_env(),
            cache=CacheConfig.from_env(),
            cached_assets=CachedAssetsConfig.from_env(),
            cloudfront=CloudFrontConfig.from_env(),
            log=LogConfig.from_env(),
            queue=QueueConfig.from_env(),
            api=ApiConfig.from_env(hf_endpoint=common_config.hf_endpoint),
            s3=S3Config.from_env(),
        )


ProcessingStepNameByInputType = Mapping[InputType, str]

ProcessingStepNameByInputTypeAndEndpoint = Mapping[str, ProcessingStepNameByInputType]


@dataclass(frozen=True)
class EndpointConfig:
    """Contains the endpoint config specification to relate with step names.
    The list of processing steps corresponds to the priority in which the response
    has to be reached. The cache from the first step in the list will be used first
    then, if it's an error or missing, the second one, etc.
    The related steps depend on the query parameters passed in the request
    (dataset, config, split)
    """

    processing_step_name_by_input_type_and_endpoint: ProcessingStepNameByInputTypeAndEndpoint = field(
        default_factory=lambda: {
            "/splits": {
                "dataset": "dataset-split-names",
                "config": "config-split-names",
            },
            "/first-rows": {"split": "split-first-rows"},
            "/parquet": {
                "dataset": "dataset-parquet",
                "config": "config-parquet",
            },
            "/info": {"dataset": "dataset-info", "config": "config-info"},
            "/size": {
                "dataset": "dataset-size",
                "config": "config-size",
            },
            "/opt-in-out-urls": {
                "dataset": "dataset-opt-in-out-urls-count",
                "config": "config-opt-in-out-urls-count",
                "split": "split-opt-in-out-urls-count",
            },
            "/is-valid": {
                "dataset": "dataset-is-valid",
                "config": "config-is-valid",
                "split": "split-is-valid",
            },
            "/statistics": {"split": "split-descriptive-statistics"},
            "/compatible-libraries": {"dataset": "dataset-compatible-libraries"},
            "/croissant-crumbs": {"dataset": "dataset-croissant-crumbs"},
        }
    )

    @classmethod
    def from_env(cls) -> "EndpointConfig":
        # TODO: allow passing the mapping between endpoint and processing steps via env vars
        return cls()

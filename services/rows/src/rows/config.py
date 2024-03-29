# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from dataclasses import dataclass, field

from libapi.config import ApiConfig
from libcommon.config import (
    AssetsConfig,
    CacheConfig,
    CachedAssetsConfig,
    CloudFrontConfig,
    CommonConfig,
    LogConfig,
    ParquetMetadataConfig,
    QueueConfig,
    RowsIndexConfig,
    S3Config,
)


@dataclass(frozen=True)
class AppConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    cached_assets: CachedAssetsConfig = field(default_factory=CachedAssetsConfig)
    cloudfront: CloudFrontConfig = field(default_factory=CloudFrontConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    log: LogConfig = field(default_factory=LogConfig)
    parquet_metadata: ParquetMetadataConfig = field(default_factory=ParquetMetadataConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    rows_index: RowsIndexConfig = field(default_factory=RowsIndexConfig)
    s3: S3Config = field(default_factory=S3Config)

    @classmethod
    def from_env(cls) -> "AppConfig":
        common_config = CommonConfig.from_env()
        return cls(
            api=ApiConfig.from_env(hf_endpoint=common_config.hf_endpoint),
            assets=AssetsConfig.from_env(),
            cache=CacheConfig.from_env(),
            cached_assets=CachedAssetsConfig.from_env(),
            cloudfront=CloudFrontConfig.from_env(),
            common=common_config,
            log=LogConfig.from_env(),
            parquet_metadata=ParquetMetadataConfig.from_env(),
            queue=QueueConfig.from_env(),
            rows_index=RowsIndexConfig.from_env(),
            s3=S3Config.from_env(),
        )

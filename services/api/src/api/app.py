# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import uvicorn
from libapi.config import UvicornConfig
from libapi.jwt_token import get_jwt_public_keys
from libapi.routes.healthcheck import healthcheck_endpoint
from libapi.routes.metrics import create_metrics_endpoint
from libapi.utils import EXPOSED_HEADERS
from libcommon.cloudfront import get_cloudfront_signer
from libcommon.log import init_logging
from libcommon.processing_graph import processing_graph
from libcommon.resources import CacheMongoResource, QueueMongoResource, Resource
from libcommon.storage_client import StorageClient
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Route
from starlette_prometheus import PrometheusMiddleware

from api.config import AppConfig, EndpointConfig
from api.routes.croissant_crumbs import create_croissant_crumbs_endpoint
from api.routes.endpoint import EndpointsDefinition, create_endpoint
from api.routes.webhook import create_webhook_endpoint


def create_app() -> Starlette:
    app_config = AppConfig.from_env()
    endpoint_config = EndpointConfig.from_env()
    return create_app_with_config(app_config=app_config, endpoint_config=endpoint_config)


def create_app_with_config(app_config: AppConfig, endpoint_config: EndpointConfig) -> Starlette:
    init_logging(level=app_config.log.level)
    # ^ set first to have logs as soon as possible

    endpoints_definition = EndpointsDefinition(processing_graph, endpoint_config)
    hf_jwt_public_keys = get_jwt_public_keys(
        algorithm_name=app_config.api.hf_jwt_algorithm,
        public_key_url=app_config.api.hf_jwt_public_key_url,
        additional_public_keys=app_config.api.hf_jwt_additional_public_keys,
        timeout_seconds=app_config.api.hf_timeout_seconds,
    )

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True,
            expose_headers=EXPOSED_HEADERS,
        ),
        Middleware(GZipMiddleware),
        Middleware(PrometheusMiddleware, filter_unhandled_paths=True),
    ]

    cached_assets_storage_client = StorageClient(
        protocol=app_config.cached_assets.storage_protocol,
        storage_root=app_config.cached_assets.storage_root,
        base_url=app_config.cached_assets.base_url,
        s3_config=app_config.s3,
        # no need to specify a url_signer
    )

    url_signer = get_cloudfront_signer(cloudfront_config=app_config.cloudfront)
    assets_storage_client = StorageClient(
        protocol=app_config.assets.storage_protocol,
        storage_root=app_config.assets.storage_root,
        base_url=app_config.assets.base_url,
        s3_config=app_config.s3,
        url_signer=url_signer,
    )
    storage_clients = [cached_assets_storage_client, assets_storage_client]

    cache_resource = CacheMongoResource(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url)
    queue_resource = QueueMongoResource(database=app_config.queue.mongo_database, host=app_config.queue.mongo_url)
    resources: list[Resource] = [cache_resource, queue_resource]
    if not cache_resource.is_available():
        raise RuntimeError("The connection to the cache database could not be established. Exiting.")
    if not queue_resource.is_available():
        raise RuntimeError("The connection to the queue database could not be established. Exiting.")

    routes = [
        Route(
            endpoint_name,
            endpoint=create_endpoint(
                endpoint_name=endpoint_name,
                step_by_input_type=step_by_input_type,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                blocked_datasets=app_config.common.blocked_datasets,
                assets_storage_client=assets_storage_client,
                hf_jwt_public_keys=hf_jwt_public_keys,
                hf_jwt_algorithm=app_config.api.hf_jwt_algorithm,
                external_auth_url=app_config.api.external_auth_url,
                hf_timeout_seconds=app_config.api.hf_timeout_seconds,
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
                storage_clients=storage_clients,
            ),
        )
        for endpoint_name, step_by_input_type in endpoints_definition.step_by_input_type_and_endpoint.items()
    ] + [
        Route(
            "/croissant-crumbs",
            endpoint=create_croissant_crumbs_endpoint(
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                blocked_datasets=app_config.common.blocked_datasets,
                hf_jwt_public_keys=hf_jwt_public_keys,
                hf_jwt_algorithm=app_config.api.hf_jwt_algorithm,
                external_auth_url=app_config.api.external_auth_url,
                hf_timeout_seconds=app_config.api.hf_timeout_seconds,
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
                storage_clients=storage_clients,
            ),
        ),
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        Route("/metrics", endpoint=create_metrics_endpoint()),
        # ^ called by Prometheus
        Route(
            "/webhook",
            endpoint=create_webhook_endpoint(
                hf_webhook_secret=app_config.api.hf_webhook_secret,
                blocked_datasets=app_config.common.blocked_datasets,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                hf_timeout_seconds=app_config.api.hf_timeout_seconds,
                storage_clients=storage_clients,
            ),
            methods=["POST"],
        ),
        # ^ called by the Hub webhooks
    ]

    return Starlette(routes=routes, middleware=middleware, on_shutdown=[resource.release for resource in resources])


def start() -> None:
    uvicorn_config = UvicornConfig.from_env()
    uvicorn.run(
        "app:create_app",
        host=uvicorn_config.hostname,
        port=uvicorn_config.port,
        factory=True,
        workers=uvicorn_config.num_workers,
    )

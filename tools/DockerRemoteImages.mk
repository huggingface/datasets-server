export SERVICE_ADMIN_DOCKER_IMAGE := $(shell jq -r '.dockerImage.admin' ${DOCKER_IMAGES})
export SERVICE_API_DOCKER_IMAGE := $(shell jq -r '.dockerImage.api' ${DOCKER_IMAGES})
export SERVICE_REVERSE_PROXY_DOCKER_IMAGE := $(shell jq -r '.dockerImage.reverseProxy' ${DOCKER_IMAGES})
export SERVICE_WORKER_DATASETS_DOCKER_IMAGE := $(shell jq -r '.dockerImage.worker.datasets' ${DOCKER_IMAGES})
export SERVICE_WORKER_SPLITS_DOCKER_IMAGE := $(shell jq -r '.dockerImage.worker.splits' ${DOCKER_IMAGES})

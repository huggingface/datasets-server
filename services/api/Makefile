# environment variables for the commands (docker compose, poetry)
export COMPOSE_PROJECT_NAME := api
export MONGO_PORT := 27031
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml
TEST_PATH ?= tests

include ../../tools/Python.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	$(POETRY) run python src/api/main.py

.PHONY: watch
watch:
	$(POETRY) run watchmedo auto-restart -d src/api -p "*.py" -R python src/api/main.py

# override the default test target to test prometheus depending on the environment
# we cannot set the env var with pytest.MonkeyPatch, it's too late
.PHONY: test
test:
	$(MAKE) down
	$(MAKE) up
	$(POETRY) run python -m pytest -vv -x ${ADDOPTS} $(TEST_PATH)
	rm -rf /tmp/api.prometheus
	mkdir /tmp/api.prometheus
	PROMETHEUS_MULTIPROC_DIR=/tmp/api.prometheus $(POETRY) run python -m pytest -vv -x -k "test_metrics" ${ADDOPTS} $(TEST_PATH)
	rm -rf /tmp/api.prometheus
	$(MAKE) down

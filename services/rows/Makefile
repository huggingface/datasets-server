# environment variables for the commands (docker compose, poetry)
export COMPOSE_PROJECT_NAME := rows
export MONGO_PORT := 27032
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml
TEST_PATH ?= tests

include ../../tools/Python.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	$(POETRY) run python src/rows/main.py

.PHONY: watch
watch:
	$(POETRY) run watchmedo auto-restart -d src/rows -p "*.py" -R python src/rows/main.py

# override the default test target to test prometheus depending on the environment
# we cannot set the env var with pytest.MonkeyPatch, it's too late
.PHONY: test
test:
	$(MAKE) down
	$(MAKE) up
	$(POETRY) run python -m pytest -vv -x ${ADDOPTS} $(TEST_PATH)
	rm -rf /tmp/rows.prometheus
	mkdir /tmp/rows.prometheus
	PROMETHEUS_MULTIPROC_DIR=/tmp/rows.prometheus $(POETRY) run python -m pytest -vv -x -k "test_metrics" ${ADDOPTS} $(TEST_PATH)
	rm -rf /tmp/rows.prometheus
	$(MAKE) down

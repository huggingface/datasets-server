# environment variables for the commands (docker compose, poetry)
export COMPOSE_PROJECT_NAME := mongodb_migration
export MONGO_PORT := 27070
export CACHE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export QUEUE_MONGO_URL := mongodb://localhost:${MONGO_PORT}
export DATABASE_MIGRATIONS_MONGO_URL := mongodb://localhost:${MONGO_PORT}
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml

include ../../tools/Python.mk
include ../../tools/PythonTest.mk
include ../../tools/Docker.mk

.PHONY: run
run:
	$(POETRY) run python src/mongodb_migration/main.py

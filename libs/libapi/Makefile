# environment variables for the commands (docker compose, poetry)
export MONGO_PORT := 27021
export COMPOSE_PROJECT_NAME := libapi
# makefile variables
DOCKER_COMPOSE := ../../tools/docker-compose-mongo.yml

include ../../tools/Python.mk
include ../../tools/PythonTest.mk
include ../../tools/Docker.mk

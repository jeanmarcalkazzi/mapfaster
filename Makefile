CONFIG_FILE = Makefile.config
include ${CONFIG_FILE}

PYTHON = python3
TEST_CMD = pytest

ifeq (${TEST_VERBOSE}, 1)
    TEST_CMD += $(empty) --verbose -vv
endif

ifeq (${TEST_COVERAGE}, 1)
	TEST_CMD += $(empty) --cov --cov-report=html
endif

# ===================== Help =====================

.PHONY: help
help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo ""
	@echo "  env		prepare environment and install required dependencies"
	@echo "  dev        run local development containers to interact with local simulation"
	@echo "  clean		remove all temp files along with docker images and docker-compose networks"
	@echo "  clean-all	runs clean + removes the virtualenv"
	@echo "  lint		run the code linters"
	@echo "  format	reformat code"
	@echo "  test		run all the tests"
	@echo ""
	@echo "Check the Makefile to know exactly what each target is doing."

# ===================== Setup =====================

.PHONY: env
env:
	which poetry | grep . && echo 'poetry installed' || curl -sSL https://install.python-poetry.org | python3 -
	poetry --version
	poetry env use python3.9
	$(eval VIRTUAL_ENVS_PATH=$(shell poetry env info --path))
	@echo $(VIRTUAL_ENVS_PATH)
	poetry install
	poetry show

.PHONY: prepare
prepare: format lint test docs
	@echo "==========================="
	@echo "Ready to push your changes!"
	@echo "==========================="

# =============================== Docker =================================

.PHONY: env-docker
env-docker:
	which poetry | grep . && echo 'poetry installed' || curl -sSL https://install.python-poetry.org | python3 -
	/root/.local/bin/poetry env use python3.9 && /root/.local/bin/poetry install

.PHONY: dev
dev:
	docker-compose -f ${DOCKERCOMPOSE_FILE} up --build --force-recreate --remove-orphans --renew-anon-volumes

.PHONY: stop
stop:
	docker-compose -f ${DOCKERCOMPOSE_FILE} stop


# ============================== Benchmark ===================================

.PHONY: benchmark-local
benchmark-local:
	WANDB_MODE=disabled poetry run python benchmark.py

.PHONY: benchmark
benchmark:
	poetry run benchmark.py

# ============================== Formatting/Linting ==============================

.PHONY: lint
lint: env clean-pyc
	poetry run bash scripts/lint.sh

.PHONY: format
format: env clean-pyc
#	Format the code.
	poetry run bash scripts/format.sh

# ============================== Document =====================================

.PHONY: docs
docs: env
	poetry run bash scripts/generate-docs.sh

# ============================== Test =========================================

.PHONY: test
test: env
	poetry run bash scripts/test.sh

# ============================== Clean =========================================

.PHONY: clean
clean: clean-pyc clean-test clean-docker

.PHONY: clean-all
clean-all: clean

clean-pyc: # Remove Python file artifacts
	find . -name '*.pyc' -exec rm -rf {} +
	find . -name '*.pyo' -exec rm -rf {} +
	find . -name '*~' -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: # Remove test and coverage artifacts
	rm -rf .tox/
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache

clean-docker:  # Remove docker image
	if docker images | grep ${PROJECT_NAME}; then \
	 	docker rmi -f ${PROJECT_NAME} || true;\
	fi;
	docker-compose -f ${DOCKERCOMPOSE_FILE} down --remove-orphans

# ==============================================================================
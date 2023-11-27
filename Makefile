SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --warn-undefined-variables
.DEFAULT_GOAL := help

PWD := $(realpath $(dir $(abspath $(firstword $(MAKEFILE_LIST)))))

PY := python
PIP := $(PY) -m pip

PROCESS_NUM := 2
MPI_OUTPUT_FILE_DIR := $(realpath $(shell mktemp -d))
MPI_OPTIONS := --allow-run-as-root -n $(PROCESS_NUM) --output-filename $(MPI_OUTPUT_FILE_DIR) -x TORCH_DISTRIBUTED_DEBUG=DETAIL

.PHONY: format
format: ## Format the Python code.	
	cp "$$($(PIP) show torch | awk '/^Location:/ { print $$2 }')/torch/__init__.py" stubs/torch/__init__.py
	trap "rm -f stubs/torch/__init__.py" EXIT; MYPYPATH="$(PWD)/stubs" $(PY) -m pysen run format lint

.PHONY: lint
lint: ## Lint the Python code.
	cp "$$($(PIP) show torch | awk '/^Location:/ { print $$2 }')/torch/__init__.py" stubs/torch/__init__.py
	trap "rm -f stubs/torch/__init__.py" EXIT; MYPYPATH="$(PWD)/stubs" $(PY) -m pysen run lint

.PHONY: test
test: ## Run all tests.
	$(PY) -m pytest -m "not mpi" tests

.PHONY: cputest
cputest: ## Run all tests except for ones requiring GPU.
	$(PY) -m pytest -m "not gpu and not mpi" tests

.PHONY: mpitest
mpitest: ## Run all tests except for ones requiring GPU.
	mpirun $(MPI_OPTIONS) $(PY) -m pytest -m mpi tests > /dev/null 2> /dev/null &&:; \
	ret=$$?; \
	for i in $$(seq 0 $$(($(PROCESS_NUM) - 1))); do echo ========= MPI process $$i =========; cat $(MPI_OUTPUT_FILE_DIR)/1/rank.$$i/stdout; cat $(MPI_OUTPUT_FILE_DIR)/1/rank.$$i/stderr; done; \
	[ $$ret = 0 ]

.PHONY: example_lint
example_lint: ## Format the Python code.
	$(PY) -m pysen --config ./example/pysen.toml run lint

.PHONY: help
help: ## Display this help message.
	@grep -E '^[%%a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

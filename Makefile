SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --warn-undefined-variables
.DEFAULT_GOAL := help
.SECONDARY:

PWD := $(realpath $(dir $(abspath $(firstword $(MAKEFILE_LIST)))))

PY := python3
PIP = $(PY) -m pip
TORCH_PACKAGE_NAME := torch
LATEST_TORCH_VERSION := $(shell $(PIP) install $(TORCH_PACKAGE_NAME)==random 2>&1 | grep -oP '(?<=from versions: )(.+)(?=.$$)' | tr ' ' '\n' | sort -V | tail -n 1)
DEFAULT_TORCH_VERSION := $(LATEST_TORCH_VERSION)
DEFAULT_DEVICE := cu117

TARGET_NAME = $(firstword $(subst -, ,$@))
SUB_TARGET = ${@:$(TARGET_NAME)-%=%}
SUB_WORDS = $(subst -, ,$(SUB_TARGET))

DEVICE = $(word 1,$(SUB_WORDS))
TORCH_VERSION = $(word 2,$(SUB_WORDS))
MINIMAL = $(findstring minimal,$(SUB_WORDS))
NIGHTLY = $(findstring nightly,$(SUB_WORDS))

INSTALL_PACKAGE = -e
INSTALL_PACKAGE += $(if $(MINIMAL),.,".[test]")
INSTALL_PACKAGE += $(if $(findstring latest,$(TORCH_VERSION)),torch,torch==$(TORCH_VERSION))
INSTALL_PACKAGE += $(if $(NIGHTLY),--pre,)
INSTALL_PACKAGE += --extra-index-url
INSTALL_PACKAGE += $(if $(NIGHTLY),https://download.pytorch.org/whl/nightly/$(DEVICE),https://download.pytorch.org/whl/$(DEVICE))

VENV = venv-$(SUB_TARGET)
VPY = $(VENV)/bin/python
VPIP = $(VPY) -m pip

venv: venv-$(DEFAULT_DEVICE)-$(DEFAULT_TORCH_VERSION) ## Create a Python virtual environment with default device and PyTorch version.

venv-%: setup.py ## Create a Python virtual environment specified by device and PyTorch version.
	$(PY) -m venv $(VENV)
	$(VPIP) install -U pip 'setuptools<59.6' wheel build
	$(VPIP) install $(INSTALL_PACKAGE)

import_check: import_check-cpu-$(DEFAULT_TORCH_VERSION)-minimal ## Check if the package can be imported with default device and PyTorch version.

import_check-%: venv-% ## Check if the package can be imported specified by device and PyTorch version
	$(VPY) -c 'import pytorch_pfn_extras'

format: format-$(DEFAULT_DEVICE)-$(DEFAULT_TORCH_VERSION) ## Format the Python code with default device and PyTorch version.

format-%: venv-% ## Format the Python code specified by device and PyTorch version.
	cp "$$($(VPIP) show torch | awk '/^Location:/ { print $$2 }')/torch/__init__.py" stubs/torch/__init__.py
	MYPYPATH="$(PWD)/stubs" $(VPY) -m pysen run format lint
	rm stubs/torch/__init__.py

lint: lint-$(DEFAULT_DEVICE)-$(DEFAULT_TORCH_VERSION) ## Lint the Python code with default device and PyTorch version.

lint-%: venv-% ## Lint the Python code specified by device and PyTorch version.
	cp "$$($(VPIP) show torch | awk '/^Location:/ { print $$2 }')/torch/__init__.py" stubs/torch/__init__.py
	MYPYPATH="$(PWD)/stubs" $(VPY) -m pysen run lint
	rm stubs/torch/__init__.py

test: test-$(DEFAULT_DEVICE)-$(DEFAULT_TORCH_VERSION) ## Run tests with default device and PyTorch version.

test-%: venv-% ## Run tests specified by device and PyTorch version.
	$(VPY) -m pytest tests

cputest: cputest-$(DEFAULT_DEVICE)-$(DEFAULT_TORCH_VERSION) ## Run no gpu tests with default device and PyTorch version.

cputest-%: venv-% ## Run no gpu tests specified by device and PyTorch version.
	$(VPY) -m pytest -m "not gpu" tests

example_lint: example_lint-$(DEFAULT_DEVICE)-$(DEFAULT_TORCH_VERSION) ## Format the Python code with default device and PyTorch version.

example_lint-%: venv-% ## Format the Python code specified by device and PyTorch version.
	$(VPY) -m pysen --config ./example/pysen.toml run lint

clean: ## Remove all Python virtual environments.
	rm -rf venv-*

help: ## Display this help message.
	@grep -E '^[%%a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo "If you want to specify the device and PyTorch version, please specify it like command-<device>-<torch_version>."
	@echo "For example, specify as test-$(DEFAULT_DEVICE)-$(DEFAULT_TORCH_VERSION)."


.PHONY: format format-% lint lint-% test test-% import_check import_check-% clean help

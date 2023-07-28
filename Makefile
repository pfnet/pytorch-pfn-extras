SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --warn-undefined-variables
.DEFAULT_GOAL := help

PWD := $(realpath $(dir $(abspath $(firstword $(MAKEFILE_LIST)))))

py := python3
pip = $(py) -m pip
TORCH_PACKAGE_NAME := torch

SPECIVIED_TORCH_VERSION = $(word 3,$(subst -, ,$@))
TORCH_VERSION = $(or $(SPECIVIED_TORCH_VERSION),$(DEFAULT_TORCH_VERSION))
DEVICE = $(word 2,$(subst -, ,$@))
LATEST_TORCH_VERSION := $(shell $(pip) install $(TORCH_PACKAGE_NAME)==random 2>&1 | grep -oP '(?<=from versions: )(.+)(?=.$$)' | tr ' ' '\n' | sort -V | tail -n 1)
DEFAULT_TORCH_VERSION := $(LATEST_TORCH_VERSION)
DEFAULT_DEVICE := cu117

venv = venv-$(DEVICE)$(if $(SPECIVIED_TORCH_VERSION),-$(SPECIVIED_TORCH_VERSION))
vpy = $(venv)/bin/python
vpip = $(vpy) -m pip


venv: venv-$(DEFAULT_DEVICE)-$(DEFAULT_TORCH_VERSION) ## Create a Python virtual environment with default device and PyTorch version.

venv-%: setup.py ## Create a Python virtual environment specified by device and PyTorch version.
	$(py) -m venv $(venv)
	$(vpip) install -U pip 'setuptools<59.6' wheel build
	$(vpip) install -e ".[test]" torch==$(TORCH_VERSION) -f https://download.pytorch.org/whl/$(DEVICE)/torch_stable.html
	touch $@

format: format-$(DEFAULT_DEVICE)-$(DEFAULT_TORCH_VERSION) ## Format the Python code with default device and PyTorch version.

format-%: venv-% ## Format the Python code specified by device and PyTorch version.
	cp "$$($(vpip) show torch | awk '/^Location:/ { print $$2 }')/torch/__init__.py" stubs/torch/__init__.py
	MYPYPATH="$(PWD)/stubs" $(vpy) -m pysen run format lint
	rm stubs/torch/__init__.py

lint: lint-$(DEFAULT_DEVICE)-$(DEFAULT_TORCH_VERSION) ## Lint the Python code with default device and PyTorch version.

lint-%: venv-% ## Lint the Python code specified by device and PyTorch version.
	cp "$$($(vpip) show torch | awk '/^Location:/ { print $$2 }')/torch/__init__.py" stubs/torch/__init__.py
	MYPYPATH="$(PWD)/stubs" $(vpy) -m pysen run lint
	rm stubs/torch/__init__.py

test: test-$(DEFAULT_DEVICE)-$(DEFAULT_TORCH_VERSION) ## Run tests with default device and PyTorch version.

test-%: venv-% ## Run tests specified by device and PyTorch version.
	$(vpy) -m pytest tests

cputest: cputest-$(DEFAULT_DEVICE)-$(DEFAULT_TORCH_VERSION) ## Run no gpu tests with default device and PyTorch version.

cputest-%: venv-% ## Run no gpu tests specified by device and PyTorch version.
	$(vpy) -m pytest -m "not gpu" tests

clean: ## Remove all Python virtual environments.
	rm -rf venv-*

help: ## Display this help message.
	@grep -E '^[%%a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo "If you want to specify the device and PyTorch version, please specify it like command-<device>-<torch_version>."
	@echo "For example, specify as test-$(DEFAULT_DEVICE)-$(DEFAULT_TORCH_VERSION)."


.PHONY: venv venv-% format format-% lint lint-% test test-% clean help

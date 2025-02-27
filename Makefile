# https://www.gnu.org/software/make/manual/make.html
ENV := dev # default, other options: test, prod

include .env.$(ENV)
export

include *.make

WORKDIR := $(shell pwd)
VERSION := $(shell cat VERSION)
PROJECT := $(shell basename $(CURDIR))

ENV_PATH := $(ENV_PATH_ROOT)/$(PROJECT)
ENV_NAME := $(shell $(PYTHON) -c 'import sys;import socket;print(f"env_{socket.gethostname()}_{sys.platform}_{sys.version_info.major}.{sys.version_info.minor}")')

ENV_JUPYTER := $(ENV_PATH_ROOT)/jupyter
JUPYTER_PORT := 3000

ifeq ($(strip $(VIRTUAL_ENV)),)
	ifeq ($(strip $(CONDA_PREFIX)),)
		PATH := $(ENV_PATH)/$(ENV_NAME)/bin:$(PATH)
		PY := $(ENV_PATH)/$(ENV_NAME)/bin/python
	else
		PATH := $(CONDA_PREFIX)/bin:$(PATH)
		PY := $(CONDA_PREFIX)/bin/python
	endif
else	
	PATH := $(VIRTUAL_ENV)/bin:$(PATH)
	PY := $(VIRTUAL_ENV)/bin/python
endif

SRC := pkg# just for this template
# SRC := $(PROJECT)# for a python package
DIST := dist
BUILD := build
API := api

# PY_FILES = $(shell find $(SRC) -type f -name '*.py')
PY_FILES := $(shell find $(SRC) -type f -name '*.py' | grep -v '^.*\/test_.*\.py$$')
PY_FILES_TEST := $(shell find $(SRC) -type f -name 'test_*.py')
PY_FILES_API := $(shell find $(API) -type f -name '*.py')

IGNORE_LIST := .gitignore .dockerignore exclude.lst

PYTHONPATH := $(SRC):$(PYTHONPATH)

ifeq ($(SSL), true)
PROTOCOL := HTTPS
else
PROTOCOL := HTTP
endif
URL := $(PROTOCOL)://$(HOST):$(PORT)

.PHONY: env test all dev clean dev pyserve gen-commands $(SRC) $(DIST) $(BUILD)

.DEFAULT_GOAL := test

.ONESHELL:

%: # https://www.gnu.org/software/make/manual/make.html#Automatic-Variables 
		@:

init:
		./scripts/init  $(filter-out $@,$(MAKECMDGOALS))

switch-env:
		sed -i 's/ENV := dev/ENV := $(filter-out $@,$(MAKECMDGOALS))/' Makefile

cert: # HTTPS server
		if [ ! -d "./certs" ]; then mkdir ./certs; fi
		if [ -f "./certs/openssl.conf" ] ; then \
		openssl req -x509 -new -config ./certs/openssl.conf -out ./certs/cert.pem -keyout ./certs/key.pem ;  else \
		openssl req -x509 -nodes -newkey rsa:4096 -out ./certs/cert.pem -keyout ./certs/key.pem -sha256 -days 365 ;fi

clean:

		@if [ -d "$(BUILD)" ]; then rm -r ./$(BUILD)/*; else echo "not exist"; fi
		@if [ -d "$(DIST)" ]; then rm -r ./$(DIST)/*; else echo "not exist"; fi

clcache: 
		rm -r ./__pycache__

env-local: $(IGNORE_LIST)
		if [ ! -d $(ENV_NAME) ] ; then \
			$(PYTHON) -m venv $(ENV_NAME) && \
			for file in $(IGNORE_LIST); do \
				if ! grep -q $(ENV_NAME) $$file; then \
					echo $(ENV_NAME) >> $$file; \
				fi \
			done; \
		fi

env:
		if [ ! -d $(ENV_PATH)/$(ENV_NAME) ] ; then \
			$(PYTHON) -m venv $(ENV_PATH)/$(ENV_NAME); fi

check:
		$(PY) -m ensurepip --default-pip
		$(PY) -m pip install --upgrade pip setuptools wheel

test:
		@echo $(PATH)
		$(PY) --version
		$(PY) -m pip --version

pi: 
		$(PY) -m pip install $(filter-out $@,$(MAKECMDGOALS))
		$(PY) -m pip freeze > requirements.txt

piu:
		$(PY) -m pip install --upgrade $(filter-out $@,$(MAKECMDGOALS))
		$(PY) -m pip freeze > requirements.txt

pireq:
		make piu black isort pylint mypy

pia: requirements.txt
		$(PY) -m pip install -r requirements.txt

pkg-build: clean
		# cp VERSION $(SRC)
		$(PY) -m pip install --upgrade build
		$(PY) -m build

pkg-install-editable:
		$(PY) -m pip install -e .

pkg-install:
		$(PY) -m pip install .

pkg-check:
		$(PY) -m pip install --upgrade twine
		twine check dist/*

pkg-publish-test: .pypirc.test
		twine upload --config-file .pypirc.test -r testpypi dist/*  --verbose 

pkg-publish: .pypirc
		twine upload --config-file .pypirc dist/* --verbose  

pkg-publish-gh: 
		twine upload  dist/* --verbose  

pkg-flit-init:
		$(PY) -m pip install --upgrade flit
		if [ -f "pyproject.toml" ] ; then mv pyproject.toml pyproject.backup ;  fi
		flit init

pkg-flit-build:
		flit build

# pkg-flit-check:
# 		flit install 

pkg-flit-publish-test:
		flit publish --repository testpypi

pkg-flit-publish:
		flit publish

pkg-poetry-init:
		if [ ! -d "./.poetry" ]; then mkdir ./poetry; fi
		wget -P ./.poetry https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py
		export POETRY_HOME=./.poetry && python ./.poetry/get-poetry.py --no-modify-path
		if [ -f "pyproject.toml" ] ; then mv pyproject.toml pyproject.backup ;  fi
		./.poetry/bin/poetry init

pkg-poetry-build:
		poetry build

pkg-poetry-publish-test:
		poetry publish --repository testpypi

pkg-poetry-publish:
		poetry publish

lint:
		ruff check $(SRC)

pylint-dev:
		pylint --rcfile .pylintrc.dev $(SRC)

pylint-prod:
		pylint --rcfile .pylintrc.prod $(SRC)

format:
		black $(SRC)

sort:
		isort $(SRC) --profile black --skip $(SRC)/__init__.py
		
type:
		mypy

type-prod:
		mypy --config-file .mypy.ini.prod

script-upgrade:
		./scripts/upgrade_dependencies.sh

set-dataset-permissions:
		chmod -R a-w dataset

clean-commands: py.make api.make
		head -n 5 py.make > temp.txt && mv temp.txt py.make
		head -n 3 api.make > temp.txt && mv temp.txt api.make

gen-commands: clean-commands
		$(foreach file,$(PY_FILES),$(shell echo "\n$(subst /,-,$(subst $(SRC)/,,$(basename $(file)))):\n\t\t$(PY) $(file)" >> py.make))
		$(foreach file,$(PY_FILES_API),$(shell echo "\n$(subst /,-,$(subst $(API)/,,$(basename $(file)))):\n\t\t$(PY) $(file)" >> api.make))

jupyter-global-start:
		if [ ! -d "./logs" ] ; then \
					mkdir -p ./logs; fi
		if [ ! -d $(ENV_JUPYTER) ] ; then \
			$(PYTHON) -m venv $(ENV_JUPYTER); fi
		$(ENV_JUPYTER)/bin/python -m pip install --upgrade jupyter
		$(ENV_JUPYTER)/bin/jupyter notebook --no-browser --port=$(JUPYTER_PORT) --notebook-dir=$(WORKDIR) > ./logs/jupyter.log 2>&1 & echo $$! > ./logs/jupyter.pid

jupyter-local-start:
		if [ ! -d "./logs" ] ; then \
			mkdir -p ./logs; fi
		if [ ! -d $(ENV_PATH) ] ; then \
			$(PYTHON) -m venv $(ENV_PATH); fi
		$(PY) -m pip install --upgrade jupyter
		$(PY) -m jupyter notebook --no-browser --port=$(JUPYTER_PORT) --notebook-dir=$(WORKDIR) > ./logs/jupyter.log 2>&1 & echo $$! > ./logs/jupyter.pid

jupyter-url:
		@echo "Fetching Jupyter URL..."
		@grep -oP "http://.*\?token=[a-z0-9]*" ./logs/jupyter.log || echo "URL not found in logs."

jupyter-stop:
		if [ -f ./logs/jupyter.pid ]; then \
			kill `cat ./logs/jupyter.pid` && rm ./logs/jupyter.pid; \
		else \
			echo "No Jupyter PID file found."; \
		fi

jupyter-voila:
		$(PY) -m pip install voila
		$(PY) -m voila --port=$(JUPYTER_PORT) --no-browser --notebook-dir=$(WORKDIR)
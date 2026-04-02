SHELL := /bin/bash
PYTHON := py -3.11 # TODO: add your python path
VENV_DIR := .venv
ACTIVATE_VENV := source $(VENV_DIR)/Scripts/activate

install: venv

venv: $(VENV_DIR)/Scripts/activate

$(VENV_DIR)/Scripts/activate: requirements.txt
	test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	$(ACTIVATE_VENV); pip install -Ur requirements.txt
	touch $(VENV_DIR)/Scripts/activate

test:
	$(VENV_DIR)/Scripts/python -m pytest tests

run:
	docker-compose -f src/compose.yml up --build

clean:
	rm -rf $(VENV_DIR)
	find . -type f -name '*.pyc' -delete || true
	find . -type d -name '__pycache__' -delete || true
	rm -rf .pytest_cache
	rm -rf data/models/*

.PHONY: all venv clean test install run

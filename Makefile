SHELL := /bin/bash
PYTHON := C:\Users\chamb\AppData\Local\Microsoft\WindowsApps\python3.exe # TODO: add your python path
VENV_DIR := .venv
ACTIVATE_VENV := source $(VENV_DIR)/Scripts/activate

install: venv

venv: $(VENV_DIR)/Scripts/activate

$(VENV_DIR)/Scripts/activate: requirements.txt
	test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	$(ACTIVATE_VENV); pip install -Ur requirements.txt
	touch $(VENV_DIR)/Scripts/activate

test:
	pytest tests

run:
	docker-compose -f src/compose.yml up 

clean:
	rm -rf $(VENV_DIR)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf .pytest_cache
	rm -rf data/models/*

.PHONY: all venv clean test install run

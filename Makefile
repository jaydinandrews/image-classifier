VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

test: $(VENV)/bin/activate
	$(PYTHON) test.py --input list.txt --svm saves/95/2022-11-29-21:46:47-RBF-SVC.svm --model saves/95/2022-11-29-21:44:45-K-Means.model

help: $(VENV)/bin/activate
	$(PYTHON) main.py --help

train: $(VENV)/bin/activate
	$(PYTHON) main.py --action train

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

clean:
	find . | grep -E "__pycache__" | xargs rm -rf
	rm -rf logs/*.log
	rm -rf output.txt

wipe: clean
	rm -rf $(VENV)

lint:
	flake8 --exit-zero .

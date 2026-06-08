# JMLC — common tasks. Targets get implemented task-by-task per the build plan.
.PHONY: setup data train eval test report clean

setup:  ## install core dependencies
	pip install -r requirements.txt

data:  ## generate the synthetic dataset (train + held-out + OOD)
	python -m ml.generate --n 1500 --out data/raw/

train:  ## train models on the synthetic train set
	python -m ml.train --data data/raw/train.jsonl --out models/

eval:  ## evaluate model vs the rule baseline (AUC, precision@k, calibration)
	python -m ml.evaluate --models models/ --data data/raw/test.jsonl

test:  ## run the test suite
	pytest -q

report:  ## execute + render the report notebook to HTML
	jupyter nbconvert --to html --execute notebooks/eda.ipynb --output-dir docs/report

clean:  ## remove generated artifacts (keeps data/sample and real calls)
	rm -rf data/raw/* data/processed/* models/* mlruns/*

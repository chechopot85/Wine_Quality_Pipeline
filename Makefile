install:
	python3 -m pip install -r requirements.txt

train:
	python3 src/train.py

test:
	python3 src/evaluate.py

ci: install train test

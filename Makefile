install:
	pip install -r requirements.txt

train:
	python train.py

validate:
	python validate.py

all: install train validate
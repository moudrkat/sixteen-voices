.PHONY: install data train knockout steer figures test lint all clean

install:
	pip install -e ".[all]"

data:
	cd data && python get_books.py
	cd data && python get_author_datasets.py

train:
	python scripts/train_all.py

knockout:
	python scripts/knockout.py

steer:
	python scripts/steer.py

sweep:
	python scripts/steering_sweep.py

figures:
	python scripts/fig_knockout_heatmap.py
	python scripts/fig_steering.py

test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/

all: install data train knockout steer

clean:
	rm -rf outputs/plots/ outputs/*.json

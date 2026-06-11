.PHONY: all synth test clean

all:
	python run_all.py --config config.yaml

synth:
	python run_all.py --config config.yaml --synth-only

no-cnn:
	python run_all.py --config config.yaml --skip-cnn

test:
	pytest tests/ -v

clean:
	rm -rf artifacts/ data/ plots/
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true

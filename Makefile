.PHONY: format test install clean

install:
	pip install -e .

format:
	black src tests
	isort src tests

style: format

test:
	pytest tests/

clean:
	rm -rf build dist *.egg-info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

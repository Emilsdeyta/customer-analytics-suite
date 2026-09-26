.PHONY: install train-churn train-nbo train-clv api dashboard test lint docker-build docker-up

install:
	pip install -e ".[dev,dashboard]"

train-churn:
	python -m cas.churn.train --config configs/churn.yaml

train-nbo:
	python -m cas.nbo.train --config configs/nbo.yaml

train-clv:
	python -m cas.clv.train --config configs/clv.yaml

api:
	uvicorn cas.api.main:app --reload --port 8000

dashboard:
	streamlit run dashboards/streamlit_app.py

test:
	pytest

lint:
	ruff check src tests

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up

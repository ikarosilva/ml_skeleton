.PHONY: help install dev-install test lint format clean docker-build docker-up docker-down mlflow

# Default target
help:
	@echo "explr - Deep Learning Training Framework"
	@echo ""
	@echo "Usage:"
	@echo "  make install        Install the package"
	@echo "  make dev-install    Install with development dependencies"
	@echo "  make test           Run tests"
	@echo "  make lint           Run linting checks"
	@echo "  make format         Format code with black"
	@echo "  make clean          Clean build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start all services (MLflow + training)"
	@echo "  make docker-down    Stop all services"
	@echo "  make docker-shell   Open shell in training container"
	@echo ""
	@echo "MLflow:"
	@echo "  make mlflow         Start MLflow server locally"
	@echo "  make mlflow-stop    Stop local MLflow server"
	@echo ""
	@echo "Examples:"
	@echo "  make run-example    Run PyTorch example"

# Installation
install:
	pip install -e .

dev-install:
	pip install -e ".[all,dev]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=explr --cov-report=html

# Linting and formatting
lint:
	ruff check explr/
	mypy explr/

format:
	black explr/ tests/ examples/
	ruff check --fix explr/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d mlflow
	@echo "MLflow UI available at http://localhost:5000"

docker-down:
	docker-compose down

docker-shell:
	docker-compose run --rm training bash

docker-jupyter:
	docker-compose --profile jupyter up -d
	@echo "Jupyter Lab available at http://localhost:8888"

# Local MLflow
mlflow:
	mlflow server \
		--host 0.0.0.0 \
		--port 5000 \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns &
	@echo "MLflow server started at http://localhost:5000"

mlflow-stop:
	pkill -f "mlflow server" || true

# Run examples
run-example:
	python examples/pytorch_example.py

run-example-tune:
	explr tune configs/example.yaml --train-fn examples.pytorch_example:train_model --n-trials 10

# GPU info
gpu-info:
	explr gpu-info

# Using existing kaggle:torch Docker image
docker-kaggle:
	docker run --shm-size 50G --runtime=nvidia --privileged -it \
		-p 8888:8888 -p 5000:5000 \
		--env TF_ENABLE_ONEDNN_OPTS=0 \
		-e NVIDIA_VISIBLE_DEVICES=all \
		-v ~/git:/git \
		-v ~/PycharmProjects:/projects \
		--rm kaggle:torch bash

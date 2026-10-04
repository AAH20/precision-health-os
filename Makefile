.PHONY: help install test lint format security run clean docker-up docker-down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package with dev dependencies
	pip install -e ".[dev]"

test: ## Run all tests
	pytest --cov=src/precision_health_os --cov-report=term

test-fast: ## Run tests in parallel
	pytest -n auto --cov=src/precision_health_os

lint: ## Run ruff linter
	ruff check src tests

format: ## Run ruff formatter
	ruff format src tests

security: ## Run bandit security scan
	bandit -r src -ll

typecheck: ## Run mypy type checker
	mypy src

run: ## Run the application
	phos serve --host 0.0.0.0 --port 8000

worker: ## Run background worker
	phos worker

docker-up: ## Start Docker Compose stack
	docker compose up -d

docker-down: ## Stop Docker Compose stack
	docker compose down -v

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

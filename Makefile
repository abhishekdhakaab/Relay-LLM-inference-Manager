.PHONY: dev up down logs test lint format loadtest eval_baseline eval_candidate eval_gate

up:
	docker compose -f infra/docker-compose.yml up -d

down:
	docker compose -f infra/docker-compose.yml down --remove-orphans

nuke:
	docker compose -f infra/docker-compose.yml down --remove-orphans -v
	
logs:
	docker compose -f infra/docker-compose.yml logs -f

dev: up
	cd relay && poetry install
	cd relay && poetry run uvicorn app.main:app --host $${RELAY_HOST:-0.0.0.0} --port $${RELAY_PORT:-8000} --reload

test:
	cd relay && poetry run pytest -q

lint:
	cd relay && poetry run ruff check .
	cd relay && poetry run mypy app

format:
	cd relay && poetry run ruff format .

loadtest:
	poetry -C relay run locust -f ../scripts/locustfile.py --host http://localhost:8000

eval_baseline:
	poetry -C relay run python ../scripts/eval_replay.py --host http://localhost:8000 --gold ../eval/gold.jsonl --out eval/baseline.json --policy-label baseline

eval_candidate:
	poetry -C relay run python ../scripts/eval_replay.py --host http://localhost:8000 --gold ../eval/gold.jsonl --out eval/candidate.json --policy-label candidate --baseline-out eval/baseline.json

eval_gate:
	poetry -C relay run python ../scripts/eval_gate.py --baseline eval/baseline.json --candidate eval/candidate.json

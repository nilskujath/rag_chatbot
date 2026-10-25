RUN := PYTHONPATH=src poetry run

PDF      ?= data/raw/core_rules.pdf
MARKDOWN ?= data/processed/core_rules.md
VECTORS  ?= data/vectors.db
K        ?= 5
SIZE     ?= 15
QUESTION ?= Which Aeldari units have toughness 4?

.DEFAULT_GOAL := help
.PHONY: help install install-dev parse index pipeline chat ask-sql gen-testset eval test lint format

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	poetry install

install-dev: ## Install runtime + evaluation + development dependencies
	poetry install --with eval,dev
	$(RUN) pre-commit install

parse: ## 1. PDF -> markdown with Docling (PDF=... MARKDOWN=...)
	$(RUN) python -m scripts.parse --pdf "$(PDF)" --markdown "$(MARKDOWN)"

index: ## 2. Chunk, embed and rebuild the vector store (MARKDOWN=... VECTORS=...)
	$(RUN) python -m scripts.index --markdown "$(MARKDOWN)" --vector-db "$(VECTORS)"

pipeline: parse index ## Run parse then index

chat: ## Talk to the agent (VECTORS=... K=...)
	$(RUN) python -m scripts.chat --vector-db "$(VECTORS)" -k $(K)

ask-sql: ## Run the text-to-SQL retriever alone (QUESTION="...")
	$(RUN) python -m scripts.ask_sql "$(QUESTION)"

gen-testset: ## Generate the RAGAS test set to review (SIZE=15); overwrites data/eval/testset.jsonl
	$(RUN) python -m scripts.generate_testset --size $(SIZE)

eval: ## Answer the frozen test set with the agent and score it with RAGAS (resumable)
	$(RUN) python -m scripts.eval_retrieval --vector-db "$(VECTORS)" -k $(K)

test: ## Run the unit tests
	$(RUN) pytest

lint: ## Check style and imports
	$(RUN) ruff check src scripts tests
	$(RUN) ruff format --check src scripts tests

format: ## Fix style and imports in place
	$(RUN) ruff check --fix src scripts tests
	$(RUN) ruff format src scripts tests

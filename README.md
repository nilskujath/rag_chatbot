# Course project: Warhammer 40,000 chatbot

## Quick start

Prerequisites: Python 3.11 or newer and [Poetry](https://python-poetry.org/) 2 (the lock file is in the Poetry 2 format).

Two files from the course material are not in git and have to be placed first:

| File | Put it at | Needed by |
|---|---|---|
| `wahadb.sqlite` (unit datasheets) | `data/tables/wahadb.sqlite` | `make chat`, `make ask-sql`, `make eval` |
| `core_rules.pdf` (Core Rules) | `data/raw/core_rules.pdf` | `make parse` only; the parsed rulebook `data/processed/core_rules.md` is committed, so this step is optional |

Then:

```bash
cp .env.example .env            # then paste your Google AI Studio key as GOOGLE_API_KEY
make install-dev                # install packages
make index                      # chunk + embed data/processed/core_rules.md -> data/vectors.db
make chat                       # talk to the agent
make test                       # run tests
make eval                       # RAGAS scores on the frozen test set in data/eval/ (slow)
```

The first `make index` and `make chat` download two small sentence-transformers
models (about 180 MB); a warning about an unset `HF_TOKEN` is expected and
harmless.

`make help` lists every command.

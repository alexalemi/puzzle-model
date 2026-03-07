.PHONY: deploy refit combine suggest

deploy: refit
	bash deploy.sh

refit: combine
	uv run python scripts/refit_all.py

combine:
	uv run python -m scraper.combine

suggest:
	uv run python scripts/suggest_player_links.py

.PHONY: deploy refit refit-team combine suggest

deploy: refit refit-team
	bash deploy.sh

refit: combine
	uv run python scripts/refit_all.py

refit-team: combine
	uv run python scripts/refit_team.py

combine:
	uv run python -m scraper.combine

suggest:
	uv run python scripts/suggest_player_links.py

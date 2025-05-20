"""
ingest.py
~~~~~~~~~
Download and cache MoneyPuck shot data **and** current rosters for all
seven Canadian NHL clubs.  Results are stored in `data/raw/`.

Usage
------
Run as a script:

    python -m src.ingest               # (from repo root)
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, List

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------#
# Configuration constants
# ---------------------------------------------------------------------#
DATA_DIR = pathlib.Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# MoneyPuck season key:   20242025  ==  2024-25
#SEASON = "20242025"
SEASON = "20232024"
MP_URL = (
	f"https://moneypuck.com/moneypuck/playerData/shots/{SEASON}_shots.csv"
)

CANADIAN_TEAMS: Dict[str, int] = {
	"MTL": 8,
	"OTT": 9,
	"TOR": 10,
	"VAN": 12,
	"WPG": 17,
	"CGY": 20,
	"EDM": 22,
}

NHL_API_ROOT = "https://api-web.nhle.com/v1/roster"


# ---------------------------------------------------------------------#
# MoneyPuck download
# ---------------------------------------------------------------------#
def download_moneypuck(season: str = SEASON) -> pathlib.Path:
	"""
    Download MoneyPuck shot-level CSV for *one* season.
    Returns the local file path.
    """
	out_file = DATA_DIR / f"shots_{season}.csv"
	if out_file.exists():
		print(f"[skip] {out_file.name} already present.")
		return out_file

	print(f"Downloading MoneyPuck shots for season {season} …")
	resp = requests.get(MP_URL, timeout=60)
	resp.raise_for_status()

	out_file.write_bytes(resp.content)

	# --- robust, cross-platform print ---
	try:
		rel_path = out_file.relative_to(pathlib.Path.cwd())
	except ValueError:
		rel_path = out_file
	print(f"Saved → {rel_path}")

	return out_file


# ---------------------------------------------------------------------#
# NHL roster download
# ---------------------------------------------------------------------#
def fetch_roster(team_code: str) -> List[dict]:
	"""Return a list of player dictionaries for the given franchise code."""
	url = f"{NHL_API_ROOT}/{team_code}/current"
	resp = requests.get(url, timeout=30)
	resp.raise_for_status()
	data = resp.json()
	return data["forwards"] + data["defensemen"] + data["goalies"]


def download_rosters(team_codes: List[str] | None = None) -> None:
	"""
    Download and cache rosters for all specified teams (defaults to the
    seven Canadian clubs).  Skips any roster already on disk.
    """
	team_codes = team_codes or list(CANADIAN_TEAMS.keys())

	for code in tqdm(team_codes, desc="Rosters"):
		out_file = DATA_DIR / f"roster_{code}.json"
		if out_file.exists():
			continue

		try:
			roster = fetch_roster(code)
		except requests.HTTPError as exc:
			print(f"{code} roster failed: {exc}")
			continue

		out_file.write_text(json.dumps(roster, indent=2))
		print(f"Saved roster → {out_file.name}")


# ---------------------------------------------------------------------#
# Entry-point
# ---------------------------------------------------------------------#
def main() -> None:
	"""Run both data downloads."""
	try:
		download_moneypuck()
	except requests.HTTPError as exc:
		print(f"[ERR] MoneyPuck download failed: {exc}")
		sys.exit(1)

	download_rosters()


if __name__ == "__main__":
	main()

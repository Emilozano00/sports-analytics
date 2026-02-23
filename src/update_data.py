"""
Actualización automática de datos del Clausura 2026.
Descarga resultados recientes, reconstruye CSVs, y hace commit+push a GitHub.

Uso:
    python src/update_data.py              # descarga + rebuild + commit + push
    python src/update_data.py --no-push    # descarga + rebuild + commit (sin push)
    python src/update_data.py --dry-run    # descarga + rebuild (sin git)
"""

import sys
import os
import subprocess
import argparse
from datetime import datetime

# ── Ensure project root is importable ─────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_ROOT)

from src.api_client import get
from src.extract_season import (
    LEAGUE_ID,
    CACHE_DIR,
    RAW_DIR,
    ensure_dirs,
    get_fixture_list,
    download_fixture,
    fixture_is_cached,
    build_csvs,
    official_year,
    DELAY_BETWEEN_CALLS,
)

import time

# ── Config ────────────────────────────────────────────────────────────
API_SEASON = 2025          # API season 2025 -> Clausura 2026
TOURNAMENT = "Clausura"
YEAR = official_year(API_SEASON)  # 2026


def refresh_fixture_list():
    """Force-refresh the fixture list from API to pick up newly played matches."""
    cache_file = os.path.join(CACHE_DIR, f"{TOURNAMENT.lower()}_{YEAR}_fixtures_list.json")
    full_cache = os.path.join(CACHE_DIR, f"full_season_{API_SEASON}_fixtures.json")

    # Remove stale caches so get_fixture_list re-downloads
    for f in [cache_file, full_cache]:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Eliminado cache: {os.path.basename(f)}")

    return get_fixture_list(TOURNAMENT, API_SEASON)


def download_new_results(fixtures_list):
    """Download fixture data for finished matches that aren't cached yet."""
    # Only download fixtures with status FT (finished) or AET/PEN
    finished_statuses = {"FT", "AET", "PEN"}
    finished = [f for f in fixtures_list if f["status"] in finished_statuses]
    pending = [f for f in finished if not fixture_is_cached(f["fixture_id"])]

    total_finished = len(finished)
    cached = total_finished - len(pending)
    print(f"\n{TOURNAMENT} {YEAR}: {total_finished} partidos jugados | {cached} en cache | {len(pending)} nuevos")

    if not pending:
        print("  Todo al dia, no hay nuevos resultados.")
        return 0

    downloaded = 0
    for i, fx in enumerate(pending):
        fid = fx["fixture_id"]
        print(f"  [{i + 1}/{len(pending)}] {fx['date'][:10]} {fx['home']} vs {fx['away']} (ID:{fid})...", end=" ", flush=True)
        try:
            result = download_fixture(fid)
            if result:
                has_stats = len(result.get("statistics", [])) > 0
                has_players = len(result.get("players", [])) > 0
                print(f"OK (stats:{'Y' if has_stats else 'N'} players:{'Y' if has_players else 'N'})")
                downloaded += 1
            else:
                print("EMPTY (partido sin datos aun)")
        except RuntimeError as e:
            print(f"\n  RATE LIMIT: {e}")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        if i < len(pending) - 1:
            time.sleep(DELAY_BETWEEN_CALLS)

    return downloaded


def rebuild_csvs():
    """Rebuild CSVs from cached fixture data."""
    print("\nReconstruyendo CSVs...")
    build_csvs([TOURNAMENT], API_SEASON)

    # Also rebuild combined file if Apertura data exists
    apertura_list = os.path.join(CACHE_DIR, f"apertura_{YEAR}_fixtures_list.json")
    if os.path.exists(apertura_list):
        print("  Tambien reconstruyendo combinado (Apertura + Clausura)...")
        build_csvs(["Apertura", TOURNAMENT], API_SEASON)


def git_commit_and_push(push=True):
    """Stage CSV changes, commit, and optionally push."""
    os.chdir(PROJECT_ROOT)

    # Check for changes in CSV files
    result = subprocess.run(
        ["git", "status", "--porcelain", "data/raw/"],
        capture_output=True, text=True,
    )
    changed_files = [line.strip().split()[-1] for line in result.stdout.strip().split("\n") if line.strip()]
    csv_changes = [f for f in changed_files if f.endswith(".csv")]

    if not csv_changes:
        print("\nNo hay cambios en los CSVs — nada que commitear.")
        return False

    print(f"\nArchivos modificados:")
    for f in csv_changes:
        print(f"  {f}")

    # Stage only CSV files (not cache directories)
    for f in csv_changes:
        subprocess.run(["git", "add", f], check=True)

    # Commit
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = f"Update {TOURNAMENT} {YEAR} data ({timestamp})"
    subprocess.run(["git", "commit", "-m", msg], check=True)
    print(f"\nCommit: {msg}")

    # Push
    if push:
        print("Pushing to GitHub...")
        subprocess.run(["git", "push"], check=True)
        print("Push exitoso.")
    else:
        print("(--no-push: commit creado, sin push)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Actualizar datos Clausura 2026")
    parser.add_argument("--no-push", action="store_true",
                        help="Commit sin push a GitHub")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo descarga y rebuild, sin git")
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"  Actualizacion de datos — {TOURNAMENT} {YEAR}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    ensure_dirs()

    # Step 1: Refresh fixture list from API
    print("\n[1/3] Refrescando lista de fixtures...")
    fixtures = refresh_fixture_list()
    total = len(fixtures)
    finished = sum(1 for f in fixtures if f["status"] in {"FT", "AET", "PEN"})
    upcoming = total - finished
    print(f"  {TOURNAMENT} {YEAR}: {total} partidos total, {finished} jugados, {upcoming} por jugar")

    # Step 2: Download new results
    print("\n[2/3] Descargando resultados nuevos...")
    downloaded = download_new_results(fixtures)

    # Step 3: Rebuild CSVs (always, even if no new downloads — fixture list may have updated)
    print("\n[3/3] Reconstruyendo CSVs...")
    rebuild_csvs()

    # Step 4: Git commit + push
    if not args.dry_run:
        git_commit_and_push(push=not args.no_push)
    else:
        print("\n(--dry-run: sin operaciones git)")

    print(f"\nListo. {downloaded} nuevos partidos descargados.")


if __name__ == "__main__":
    main()

"""Download race and qualifying data with FastF1 and save it as CSV in data/.

Usage:
    .venv/bin/python src/collect_data.py            # 2026 season, standings + US GP history
    .venv/bin/python src/collect_data.py --weekend  # 2026 US GP weekend sessions
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.ergast import Ergast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402  (enables the FastF1 cache on import)

SEASON = 2026
USGP_NAME = "United States Grand Prix"
USGP_HISTORY_YEARS = [2023, 2024, 2025]
RESULT_SESSIONS = {"Q": "qualifying", "R": "race"}
WEEKEND_SESSIONS = {"FP1": "fp1", "FP2": "fp2", "FP3": "fp3", "Q": "qualifying"}


class Summary:
    """Collects what was saved, skipped and failed, then prints it."""

    def __init__(self):
        self.saved, self.skipped, self.failed = [], [], []

    def print(self):
        print("\n=== Summary ===")
        print(f"Saved ({len(self.saved)}):")
        for item in self.saved:
            print(f"  + {item}")
        print(f"Skipped ({len(self.skipped)}):")
        for item in self.skipped:
            print(f"  - {item}")
        print(f"Failed ({len(self.failed)}):")
        for item in self.failed:
            print(f"  ! {item}")


def _load_session(year, event, identifier, summary, laps=False):
    """Load one session, or return None if it doesn't exist, hasn't run, or fails."""
    label = f"{year} {event} {identifier}"
    try:
        session = fastf1.get_session(year, event, identifier)
    except ValueError as exc:  # session not part of this event's format
        summary.skipped.append(f"{label}: not in schedule ({exc})")
        return None

    # session.date is naive UTC
    if pd.notna(session.date) and session.date > datetime.now(timezone.utc).replace(tzinfo=None):
        summary.skipped.append(f"{label}: not run yet ({session.date:%Y-%m-%d %H:%M} UTC)")
        return None

    try:
        session.load(laps=laps, telemetry=False, weather=False, messages=False)
    except Exception as exc:  # network errors, missing data, rate limits
        summary.failed.append(f"{label}: {type(exc).__name__}: {exc}")
        return None

    if session.results is None or session.results.empty:
        summary.failed.append(f"{label}: loaded but no results available")
        return None
    return session


def _tag(df, session, session_name):
    """Prefix a results/laps frame with event identifiers."""
    df = df.copy()
    df.insert(0, "Session", session_name)
    df.insert(0, "EventName", session.event["EventName"])
    df.insert(0, "Round", session.event["RoundNumber"])
    df.insert(0, "Year", session.event.year)
    return df


def _save(frames, filename, summary):
    if not frames:
        return
    path = config.DATA_DIR / filename
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    summary.saved.append(f"{path.relative_to(config.PROJECT_ROOT)} ({len(frames)} sessions)")


def completed_rounds(year):
    """Rounds whose race start time is in the past."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    now = pd.Timestamp.now(tz="UTC")
    race_start = pd.to_datetime(schedule["Session5DateUtc"]).dt.tz_localize("UTC")
    return schedule[race_start < now]


def collect_season_results(year=SEASON, summary=None):
    """Race and qualifying results for every completed round of a season."""
    summary = summary or Summary()
    rounds = completed_rounds(year)
    print(f"{year}: {len(rounds)} completed rounds")
    frames = {name: [] for name in RESULT_SESSIONS.values()}
    for _, event in rounds.iterrows():
        for identifier, name in RESULT_SESSIONS.items():
            session = _load_session(year, event["RoundNumber"], identifier, summary)
            if session is not None:
                frames[name].append(_tag(session.results, session, name))
    for name, session_frames in frames.items():
        _save(session_frames, f"{year}_{name}_results.csv", summary)
    return summary


def collect_usgp_history(years=USGP_HISTORY_YEARS, summary=None):
    """Race and qualifying results for past US Grands Prix."""
    summary = summary or Summary()
    frames = {name: [] for name in RESULT_SESSIONS.values()}
    for year in years:
        for identifier, name in RESULT_SESSIONS.items():
            session = _load_session(year, USGP_NAME, identifier, summary)
            if session is not None:
                frames[name].append(_tag(session.results, session, name))
    for name, session_frames in frames.items():
        _save(session_frames, f"usgp_history_{name}_results.csv", summary)
    return summary


def collect_driver_standings(year=SEASON, summary=None):
    """Official driver standings (including sprint points) after each completed round."""
    summary = summary or Summary()
    ergast = Ergast()
    frames = []
    for round_number in completed_rounds(year)["RoundNumber"]:
        label = f"{year} round {round_number} driver standings"
        try:
            standings = ergast.get_driver_standings(year, round=int(round_number)).content[0]
        except Exception as exc:  # network errors, rate limits, round not published yet
            summary.failed.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        standings = standings[["position", "driverCode", "points", "wins"]].copy()
        standings.insert(0, "Round", round_number)
        standings.insert(0, "Year", year)
        frames.append(standings)
    if frames:
        path = config.DATA_DIR / f"{year}_driver_standings.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        summary.saved.append(f"{path.relative_to(config.PROJECT_ROOT)} ({len(frames)} rounds)")
    return summary


def collect_usgp_2026_weekend(summary=None):
    """FP1, FP2, FP3 and qualifying for the 2026 US GP: laps and results per session.

    Sessions that aren't in the weekend format (e.g. FP2/FP3 on a sprint
    weekend) or haven't run yet are skipped. Safe to re-run as the weekend goes on.
    """
    summary = summary or Summary()
    for identifier, name in WEEKEND_SESSIONS.items():
        session = _load_session(SEASON, USGP_NAME, identifier, summary, laps=True)
        if session is None:
            continue
        _save([_tag(session.laps, session, name)], f"usgp_2026_{name}_laps.csv", summary)
        _save([_tag(session.results, session, name)], f"usgp_2026_{name}_results.csv", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weekend", action="store_true",
                        help="fetch the 2026 US GP weekend sessions instead")
    args = parser.parse_args()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary = Summary()
    if args.weekend:
        collect_usgp_2026_weekend(summary)
    else:
        collect_season_results(SEASON, summary)
        collect_driver_standings(SEASON, summary)
        collect_usgp_history(USGP_HISTORY_YEARS, summary)
    summary.print()


if __name__ == "__main__":
    main()

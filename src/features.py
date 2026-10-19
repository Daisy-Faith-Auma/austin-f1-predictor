"""Build one row per driver per 2026 race, using only information known before the race starts.

Usage:
    .venv/bin/python src/features.py

Reads the CSVs written by src/collect_data.py and writes data/features.csv.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

SEASON = 2026
FORM_WINDOW = 5                 # races used for driver and team rolling form
FILL_AVG_FINISH = 11.5          # mid-grid of 22 cars; used when there is no earlier race
NO_QUALI_TIME_PENALTY_S = 1.0   # added to the slowest gap in that session
FILL_USGP_BEST = 21             # one worse than last place in the 20-car 2023-2025 fields

ID_COLUMNS = ["Year", "Round", "EventName", "Abbreviation", "TeamName"]
FEATURE_COLUMNS = [
    "grid_position",
    "quali_gap_s",
    "no_quali_time",
    "avg_finish_prev5",
    "points_before",
    "team_avg_finish_prev5",
    "dnf_rate_before",
    "usgp_best_finish",
    "no_usgp_history",
    "prev_races",
]
TARGET = "won"


def load_inputs():
    data = config.DATA_DIR
    race = pd.read_csv(data / f"{SEASON}_race_results.csv")
    quali = pd.read_csv(data / f"{SEASON}_qualifying_results.csv")
    standings = pd.read_csv(data / f"{SEASON}_driver_standings.csv")
    usgp = pd.read_csv(data / "usgp_history_race_results.csv")
    return race, quali, standings, usgp


def is_dnf(classified_position):
    """Not classified (R = retired, D = disqualified, W = withdrew/DNS, N = not classified)."""
    return ~classified_position.astype(str).str.isdigit()


def base_table(race):
    """One row per race entry. Built from race results so nobody is dropped by a quali join."""
    df = race[ID_COLUMNS + ["GridPosition", "Position", "ClassifiedPosition"]].copy()
    df["grid_position"] = df["GridPosition"].astype(int)
    df["finish"] = df["Position"]
    df["dnf"] = is_dnf(df["ClassifiedPosition"]).astype(int)
    df[TARGET] = (df["Position"] == 1).astype(int)
    return df.sort_values(["Round", "grid_position"]).reset_index(drop=True)


def add_quali_gap(df, quali):
    """Best Q1/Q2/Q3 time minus pole time. No time set -> slowest gap that session + penalty."""
    q = quali[["Round", "Abbreviation", "Q1", "Q2", "Q3"]].copy()
    times = q[["Q1", "Q2", "Q3"]].apply(pd.to_timedelta)
    q["best_s"] = times.min(axis=1).dt.total_seconds()
    q["quali_gap_s"] = q["best_s"] - q.groupby("Round")["best_s"].transform("min")

    df = df.merge(q[["Round", "Abbreviation", "quali_gap_s"]], on=["Round", "Abbreviation"], how="left")
    df["no_quali_time"] = df["quali_gap_s"].isna().astype(int)
    slowest = df.groupby("Round")["quali_gap_s"].transform("max")
    df["quali_gap_s"] = df["quali_gap_s"].fillna(slowest + NO_QUALI_TIME_PENALTY_S).round(3)
    return df


def add_driver_form(df):
    """Rolling stats over each driver's earlier races only (shift(1) excludes the current race)."""
    df = df.sort_values(["Abbreviation", "Round"])
    by_driver = df.groupby("Abbreviation")
    prev_finish = by_driver["finish"].shift(1)
    prev_dnf = by_driver["dnf"].shift(1)

    df["prev_races"] = by_driver.cumcount()
    df["avg_finish_prev5"] = (
        prev_finish.groupby(df["Abbreviation"])
        .rolling(FORM_WINDOW, min_periods=1).mean()
        .reset_index(level=0, drop=True)
        .fillna(FILL_AVG_FINISH)
    )
    df["dnf_rate_before"] = (
        prev_dnf.groupby(df["Abbreviation"]).cumsum() / df["prev_races"]
    ).fillna(0.0)
    return df.sort_index()


def add_team_form(df):
    """Mean finish of both cars per team per round, averaged over the team's previous 5 rounds."""
    team_round = df.groupby(["TeamName", "Round"])["finish"].mean().reset_index()
    team_round = team_round.sort_values(["TeamName", "Round"])
    team_round["team_avg_finish_prev5"] = (
        team_round.groupby("TeamName")["finish"]
        .transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
        .fillna(FILL_AVG_FINISH)
    )
    return df.merge(team_round[["TeamName", "Round", "team_avg_finish_prev5"]],
                    on=["TeamName", "Round"], how="left")


def add_points_before(df, standings):
    """Official championship points (incl. sprints) after the previous round; 0 before round 1."""
    prev = standings[["Round", "driverCode", "points"]].rename(
        columns={"driverCode": "Abbreviation", "points": "points_before"})
    prev["Round"] = prev["Round"] + 1
    df = df.merge(prev, on=["Round", "Abbreviation"], how="left")
    df["points_before"] = df["points_before"].fillna(0.0)
    return df


def add_usgp_history(df, usgp):
    """Best US GP finish 2023-2025 (all before 2026, so no leakage). No history -> fill + flag."""
    best = usgp.groupby("Abbreviation")["Position"].min().rename("usgp_best_finish")
    df = df.merge(best, left_on="Abbreviation", right_index=True, how="left")
    df["no_usgp_history"] = df["usgp_best_finish"].isna().astype(int)
    df["usgp_best_finish"] = df["usgp_best_finish"].fillna(FILL_USGP_BEST).astype(int)
    return df


def build_features():
    race, quali, standings, usgp = load_inputs()
    df = base_table(race)
    df = add_quali_gap(df, quali)
    df = add_driver_form(df)
    df = add_team_form(df)
    df = add_points_before(df, standings)
    df = add_usgp_history(df, usgp)
    df = df.sort_values(["Round", "grid_position"]).reset_index(drop=True)
    return df[ID_COLUMNS + FEATURE_COLUMNS + [TARGET]]


def main():
    features = build_features()
    path = config.DATA_DIR / "features.csv"
    features.to_csv(path, index=False)
    print(f"Saved {path.relative_to(config.PROJECT_ROOT)}: {len(features)} rows, "
          f"{features['Round'].nunique()} races, {features[TARGET].sum()} winners")

    print("\nSample rows:")
    print(features.sample(5, random_state=1).to_string(index=False))

    print("\nDrivers with no qualifying time (gap = slowest that session + "
          f"{NO_QUALI_TIME_PENALTY_S:.0f}s):")
    cols = ["Round", "EventName", "Abbreviation", "grid_position", "quali_gap_s", "no_quali_time"]
    print(features[features["no_quali_time"] == 1][cols].to_string(index=False))


if __name__ == "__main__":
    main()

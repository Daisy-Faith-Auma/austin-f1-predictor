"""Evaluate simple winner-picking baselines on 2026 rounds 6-17.

Usage:
    .venv/bin/python src/baseline.py

Reads data/features.csv (from src/features.py) and writes outputs/baselines.csv.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

EVAL_ROUNDS = range(6, 18)

# name -> (stage, sort columns, ascending). The first row after sorting is the pick;
# later sort columns only break ties. Pre-qualifying baselines must not use grid/quali data.
BASELINES = {
    "pole_sitter": ("post-qualifying", ["quali_gap_s", "no_quali_time"], [True, True]),
    "best_5_race_avg_finish": ("pre-qualifying", ["avg_finish_prev5", "points_before"], [True, False]),
    "championship_leader": ("pre-qualifying", ["points_before", "avg_finish_prev5"], [False, True]),
}


def pick(race_rows, sort_cols, ascending):
    """Return (picked driver, whether the primary metric was tied for first)."""
    ranked = race_rows.sort_values(sort_cols, ascending=ascending)
    primary = sort_cols[0]
    tied = (race_rows[primary] == ranked[primary].iloc[0]).sum() > 1
    return ranked["Abbreviation"].iloc[0], tied


def evaluate(features):
    features = features[features["Round"].isin(EVAL_ROUNDS)]
    picks = []
    for round_number, rows in features.groupby("Round"):
        winner = rows.loc[rows["won"] == 1, "Abbreviation"].item()
        for name, (stage, sort_cols, ascending) in BASELINES.items():
            driver, tied = pick(rows, sort_cols, ascending)
            picks.append({"Round": round_number, "EventName": rows["EventName"].iloc[0],
                          "baseline": name, "pick": driver, "winner": winner,
                          "correct": driver == winner, "tie_broken": tied})
    picks = pd.DataFrame(picks)

    summary = (picks.groupby("baseline", sort=False)
               .agg(correct=("correct", "sum"), races=("correct", "size"),
                    ties_broken=("tie_broken", "sum"))
               .reset_index())
    summary.insert(1, "stage", summary["baseline"].map(lambda b: BASELINES[b][0]))
    summary.insert(2, "rounds", f"{min(EVAL_ROUNDS)}-{max(EVAL_ROUNDS)}")
    summary["accuracy_pct"] = (100 * summary["correct"] / summary["races"]).round(1)
    return summary, picks


def main():
    features = pd.read_csv(config.DATA_DIR / "features.csv")
    summary, picks = evaluate(features)

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUTS_DIR / "baselines.csv"
    summary.to_csv(path, index=False)

    print("Picks by round (* = correct):")
    table = picks.assign(pick=picks["pick"] + picks["correct"].map({True: "*", False: ""}))
    print(table.pivot(index=["Round", "winner"], columns="baseline", values="pick")
          [list(BASELINES)].to_string())
    print(f"\nSaved {path.relative_to(config.PROJECT_ROOT)}:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

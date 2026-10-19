"""Predict 2026 US GP win probabilities.

Usage:
    .venv/bin/python src/predict.py --stage pre    # now: season form only
    .venv/bin/python src/predict.py --stage post   # after qualifying (Oct 24, 21:00 UTC)

Post-qualifying probabilities are temperature-scaled with outputs/calibration_post_quali.json
(from src/calibrate.py); pass --no-calibration to skip that.

Writes outputs/prediction_<pre|post>_quali.csv and a top-10 bar chart .png.
"""

import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.calibrate import apply_temperature, load_temperature  # noqa: E402
from src.collect_data import Summary, collect_usgp_2026_weekend  # noqa: E402
from src.features import SEASON, build_features, load_inputs  # noqa: E402
from src.model import predict_race_probs  # noqa: E402

USGP_EVENT = "United States Grand Prix"
TOP_N_CHART = 10

# Chart tokens (reference palette, light surface)
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SERIES = "#2a78d6"


def load_model(stage):
    with open(config.OUTPUTS_DIR / f"model_{stage}_quali.pkl", "rb") as f:
        return pickle.load(f)


def latest_lineup(race):
    """Drivers and teams from the most recent completed round."""
    last = race[race["Round"] == race["Round"].max()]
    return last[["Abbreviation", "TeamName"]].reset_index(drop=True)


def fetch_usgp_qualifying():
    """Run the weekend collection and return the US GP qualifying results, or exit if not run yet."""
    summary = collect_usgp_2026_weekend(Summary())
    summary.print()
    path = config.DATA_DIR / "usgp_2026_qualifying_results.csv"
    if not path.exists():
        sys.exit("\nUS GP qualifying results aren't available yet. Run --stage post after qualifying.")
    return pd.read_csv(path)


def upcoming_rows(lineup, round_number, grid=None):
    """Race rows for the upcoming round, with no result yet."""
    rows = lineup.copy()
    rows["Year"] = SEASON
    rows["Round"] = round_number
    rows["EventName"] = USGP_EVENT
    rows["GridPosition"] = rows["Abbreviation"].map(grid) if grid is not None else pd.NA
    rows["Position"] = float("nan")
    rows["ClassifiedPosition"] = pd.NA
    # all-NA columns are refilled as NaN by concat; dropping them avoids a pandas dtype warning
    return rows.dropna(axis=1, how="all")


def post_quali_grid(quali_usgp, lineup):
    """Entry list and provisional grid from qualifying order. Penalties aren't applied.

    Drivers in the latest line-up with no qualifying row go to the back of the grid
    (and get the no_quali_time fill in features.py).
    """
    q = quali_usgp.sort_values("Position", na_position="last")
    entries = q[["Abbreviation", "TeamName"]]
    missing = lineup[~lineup["Abbreviation"].isin(entries["Abbreviation"])]
    if not missing.empty:
        print(f"Warning: no US GP qualifying row for {', '.join(missing['Abbreviation'])}; "
              "placing them at the back of the grid. Remove them if they aren't racing.")
    entries = pd.concat([entries, missing], ignore_index=True)
    grid = pd.Series(range(1, len(entries) + 1), index=entries["Abbreviation"])
    return entries.reset_index(drop=True), grid


def build_prediction(stage, calibrated=True):
    race, quali, standings = load_inputs()
    round_number = int(race["Round"].max()) + 1
    lineup = latest_lineup(race)

    if stage == "pre":
        new_race = upcoming_rows(lineup, round_number)
        new_quali = quali.iloc[0:0]
    else:
        quali_usgp = fetch_usgp_qualifying()
        entries, grid = post_quali_grid(quali_usgp, lineup)
        new_race = upcoming_rows(entries, round_number, grid)
        new_quali = quali_usgp.assign(Round=round_number)

    inputs = (pd.concat([race, new_race], ignore_index=True),
              pd.concat([quali, new_quali], ignore_index=True),
              standings)
    features = build_features(inputs)
    rows = features[features["Round"] == round_number].copy()

    bundle = load_model(stage)
    rows["win_prob"] = predict_race_probs(bundle["model"], rows, bundle["features"])
    bundle["temperature"] = None
    if stage == "post" and calibrated:
        bundle["temperature"] = load_temperature(bundle["model_name"])
        rows["win_prob_uncalibrated"] = rows["win_prob"]
        rows["win_prob"] = apply_temperature(rows["win_prob"], bundle["temperature"])
    rows = rows.sort_values("win_prob", ascending=False).reset_index(drop=True)
    rows.insert(0, "rank", range(1, len(rows) + 1))
    rows["stage"] = f"{stage}_qualifying"
    rows["model"] = bundle["model_name"]
    rows["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    rows["temperature"] = bundle["temperature"]
    extra = ["win_prob_uncalibrated"] if "win_prob_uncalibrated" in rows else []
    cols = ["rank", "Abbreviation", "TeamName", "win_prob", *extra, *bundle["features"],
            "stage", "model", "temperature", "generated_utc"]
    out = rows[cols].rename(columns={"Abbreviation": "driver", "TeamName": "team"})
    return out, bundle, round_number


def plot_top(prediction, stage, bundle, path):
    top = prediction.head(TOP_N_CHART).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    labels = [f"{d}  {t}" for d, t in zip(top["driver"], top["team"])]
    ax.barh(labels, top["win_prob"], height=0.55, color=SERIES)
    xmax = top["win_prob"].max()
    for y, p in enumerate(top["win_prob"]):
        ax.text(p + xmax * 0.015, y, f"{p:.1%}", va="center", fontsize=9, color=TEXT_PRIMARY)

    ax.set_xlim(0, xmax * 1.15)
    ax.xaxis.set_visible(False)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(TEXT_SECONDARY)
    ax.tick_params(axis="y", length=0, labelsize=9, labelcolor=TEXT_PRIMARY)

    stage_label = "pre-qualifying" if stage == "pre" else "post-qualifying"
    fig.text(0.02, 0.965, f"2026 US Grand Prix: win probability ({stage_label}), top {TOP_N_CHART}",
             ha="left", va="top", fontsize=12, color=TEXT_PRIMARY, fontweight="bold")
    subtitle = (f"{bundle['model_name'].replace('_', ' ')} trained on rounds 1-"
                f"{max(bundle['trained_on_rounds'])}")
    if bundle["temperature"]:
        subtitle += f", temperature-calibrated (T={bundle['temperature']:.2f})"
    subtitle += f"; all {len(prediction)} drivers sum to 100%"
    fig.text(0.02, 0.915, subtitle, ha="left", va="top", fontsize=9, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["pre", "post"], required=True)
    parser.add_argument("--no-calibration", action="store_true",
                        help="post stage: skip temperature scaling")
    args = parser.parse_args()

    prediction, bundle, round_number = build_prediction(args.stage, not args.no_calibration)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.OUTPUTS_DIR / f"prediction_{args.stage}_quali.csv"
    png_path = config.OUTPUTS_DIR / f"prediction_{args.stage}_quali.png"
    prediction.to_csv(csv_path, index=False)
    plot_top(prediction, args.stage, bundle, png_path)

    calibration = f", T={bundle['temperature']:.2f}" if bundle["temperature"] else ""
    print(f"\nRound {round_number} US GP, {args.stage}-qualifying ({bundle['model_name']}{calibration}), "
          f"{len(prediction)} drivers, probabilities sum to {prediction['win_prob'].sum():.3f}")
    print(prediction.head(5)[["rank", "driver", "team", "win_prob"]]
          .assign(win_prob=lambda d: d["win_prob"].map("{:.1%}".format)).to_string(index=False))
    print(f"\nSaved {csv_path.relative_to(config.PROJECT_ROOT)} and "
          f"{png_path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

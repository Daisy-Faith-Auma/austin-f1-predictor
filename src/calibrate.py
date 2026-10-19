"""Temperature-scale the post-qualifying model's win probabilities.

Usage:
    .venv/bin/python src/calibrate.py

Fits one temperature T on the walk-forward backtest predictions in
outputs/backtest/predictions.csv, where each race was predicted by a model trained only on
earlier rounds. Within a race, calibrated p_i = p_i^(1/T) / sum_j p_j^(1/T); T > 1 softens
overconfident predictions. Writes outputs/calibration_post_quali.json (used by
src/predict.py --stage post --calibrated) and outputs/backtest/calibration_post_quali.csv.

Re-run after src/model.py, since the backtest predictions and saved model change together.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

VERSION = "post_quali"
CALIBRATION_PATH = config.OUTPUTS_DIR / f"calibration_{VERSION}.json"
LOG_T_BOUNDS = (np.log(0.2), np.log(10.0))
EPS = 1e-12


def apply_temperature(probs, temperature):
    """Rescale one race's probabilities (summing to 1) by temperature; result sums to 1."""
    logits = np.log(np.clip(np.asarray(probs, dtype=float), EPS, 1.0)) / temperature
    logits -= logits.max()
    scaled = np.exp(logits)
    return scaled / scaled.sum()


def calibrate(predictions, temperature):
    out = predictions.copy()
    out["win_prob"] = out.groupby("Round")["win_prob"].transform(
        lambda p: apply_temperature(p, temperature))
    return out


def winner_log_loss(predictions):
    winners = predictions[predictions["won"] == 1]
    return float(-np.log(np.clip(winners["win_prob"], EPS, 1.0)).mean())


def fit_temperature(predictions):
    result = minimize_scalar(lambda log_t: winner_log_loss(calibrate(predictions, np.exp(log_t))),
                             bounds=LOG_T_BOUNDS, method="bounded")
    return float(np.exp(result.x))


def leave_one_race_out(predictions):
    """Calibrate each race with a temperature fitted on the other races only."""
    parts, temps = [], {}
    for round_number in sorted(predictions["Round"].unique()):
        others = predictions[predictions["Round"] != round_number]
        temps[int(round_number)] = fit_temperature(others)
        parts.append(calibrate(predictions[predictions["Round"] == round_number],
                               temps[int(round_number)]))
    return pd.concat(parts), temps


def summarize(label, predictions):
    winners = predictions[predictions["won"] == 1]
    poles = predictions[predictions["pole"]]
    return {
        "predictions": label,
        "log_loss": round(winner_log_loss(predictions), 3),
        "avg_winner_prob": round(winners["win_prob"].mean(), 3),
        "avg_pole_sitter_prob": round(poles["win_prob"].mean(), 3),
        "max_prob": round(predictions["win_prob"].max(), 3),
        "pole_sitter_win_rate": round(poles["won"].mean(), 3),
    }


def load_backtest():
    """Backtest predictions of the saved post-qualifying model, with a pole-sitter flag."""
    with open(config.OUTPUTS_DIR / f"model_{VERSION}.pkl", "rb") as f:
        model_name = pickle.load(f)["model_name"]
    preds = pd.read_csv(config.OUTPUTS_DIR / "backtest" / "predictions.csv")
    preds = preds[(preds["version"] == VERSION) & (preds["model"] == model_name)]
    features = pd.read_csv(config.DATA_DIR / "features.csv")
    pole = features[(features["quali_gap_s"] == 0) & (features["no_quali_time"] == 0)]
    preds = preds.merge(pole[["Round", "Abbreviation"]].assign(pole=True),
                        on=["Round", "Abbreviation"], how="left")
    preds["pole"] = preds["pole"].notna()
    return preds, model_name


def load_temperature(model_name):
    """Saved temperature for the post-qualifying model; errors if it was fitted to a different model."""
    if not CALIBRATION_PATH.exists():
        sys.exit(f"{CALIBRATION_PATH.relative_to(config.PROJECT_ROOT)} not found. "
                 "Run src/calibrate.py first, or drop --calibrated.")
    with open(CALIBRATION_PATH) as f:
        calibration = json.load(f)
    if calibration["model_name"] != model_name:
        sys.exit(f"Calibration was fitted for {calibration['model_name']}, but the saved model is "
                 f"{model_name}. Re-run src/calibrate.py.")
    return calibration["temperature"]


def main():
    preds, model_name = load_backtest()
    rounds = sorted(int(r) for r in preds["Round"].unique())

    temperature = fit_temperature(preds)
    loro, loro_temps = leave_one_race_out(preds)
    report = pd.DataFrame([
        summarize("uncalibrated", preds),
        summarize(f"calibrated, T={temperature:.2f} fitted on all {len(rounds)} races (in-sample)",
                  calibrate(preds, temperature)),
        summarize("calibrated, leave-one-race-out T (out-of-sample)", loro),
    ])
    report.to_csv(config.OUTPUTS_DIR / "backtest" / f"calibration_{VERSION}.csv", index=False)

    with open(CALIBRATION_PATH, "w") as f:
        json.dump({
            "version": VERSION,
            "model_name": model_name,
            "method": "temperature scaling: p_i^(1/T), renormalized per race",
            "temperature": round(temperature, 4),
            "fitted_on_rounds": rounds,
            "leave_one_race_out_temperatures": {k: round(v, 3) for k, v in loro_temps.items()},
            "backtest": report.to_dict(orient="records"),
        }, f, indent=2)

    print(f"Post-qualifying {model_name}, walk-forward backtest rounds {rounds[0]}-{rounds[-1]}")
    print(f"Fitted temperature T = {temperature:.3f} (leave-one-race-out range "
          f"{min(loro_temps.values()):.2f}-{max(loro_temps.values()):.2f})\n")
    print(report.to_string(index=False))
    print(f"\nSaved {CALIBRATION_PATH.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

"""Train and walk-forward backtest pre- and post-qualifying win-probability models.

Usage:
    .venv/bin/python src/model.py

Reads data/features.csv. Writes outputs/backtest/ (per-round predictions and a model
comparison) and the best model of each version, refit on all rounds, to
outputs/model_pre_quali.pkl and outputs/model_post_quali.pkl.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.baseline import EVAL_ROUNDS, evaluate as evaluate_baselines  # noqa: E402

TARGET = "won"
RANDOM_STATE = 0

SEASON_FEATURES = [
    "avg_finish_prev5",
    "points_share_before",
    "team_avg_finish_prev5",
    "dnf_rate_before",
]
# Feature choices are backtest-driven (walk-forward log loss on the winner, rounds 6-17):
# - form_ewm improved pre-qualifying logistic regression (2.092 -> 2.061) but worsened
#   post-qualifying gradient boosting (1.573 -> 1.588), so only the pre model uses it.
# - prev_races made no difference pre-qualifying. Post-qualifying it still beats the
#   0.02 threshold after calibration (leave-one-race-out 1.593 with vs 1.622 without),
#   so it stays despite COTA's value (17) sitting at the edge of the training range.
PRE_QUALI_FEATURES = SEASON_FEATURES + ["form_ewm"]
POST_QUALI_FEATURES = SEASON_FEATURES + [
    "prev_races", "grid_position", "quali_gap_s", "no_quali_time"]

VERSIONS = {
    "pre_quali": {"stage": "pre-qualifying", "features": PRE_QUALI_FEATURES},
    "post_quali": {"stage": "post-qualifying", "features": POST_QUALI_FEATURES},
}


def make_models():
    return {
        # L2 is LogisticRegression's default penalty; C is the inverse regularization strength.
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE),
    }


def predict_race_probs(model, race_rows, features):
    """P(win) for each driver in one race, normalized to sum to 1."""
    raw = model.predict_proba(race_rows[features])[:, 1]
    return raw / raw.sum()


def walk_forward(df, features, model_name):
    """For each eval round, fit on all earlier rounds and predict that round."""
    predictions = []
    for round_number in EVAL_ROUNDS:
        train = df[df["Round"] < round_number]
        test = df[df["Round"] == round_number].copy()
        model = make_models()[model_name]
        model.fit(train[features], train[TARGET])
        test["win_prob"] = predict_race_probs(model, test, features)
        test["rank"] = test["win_prob"].rank(ascending=False, method="first").astype(int)
        predictions.append(test[["Round", "EventName", "Abbreviation", "TeamName",
                                 "win_prob", "rank", TARGET]])
    return pd.concat(predictions, ignore_index=True)


def score(predictions):
    """Winner ranked 1st / top 3, mean probability on the winner, and log loss on the winner."""
    winners = predictions[predictions[TARGET] == 1]
    races = len(winners)
    top1 = int((winners["rank"] == 1).sum())
    top3 = int((winners["rank"] <= 3).sum())
    return {
        "races": races,
        "top1": top1,
        "top1_pct": round(100 * top1 / races, 1),
        "top3": top3,
        "top3_pct": round(100 * top3 / races, 1),
        "avg_winner_prob": round(winners["win_prob"].mean(), 3),
        "log_loss": round(-np.log(winners["win_prob"]).mean(), 3),
    }


def logistic_coefficients(model, features):
    coefs = model.named_steps["logisticregression"].coef_[0]
    return pd.Series(coefs, index=features, name="coef (per 1 std dev)").round(3)


def main():
    df = pd.read_csv(config.DATA_DIR / "features.csv")
    backtest_dir = config.OUTPUTS_DIR / "backtest"
    backtest_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary, _ = evaluate_baselines(df)
    rows, all_predictions = [], []
    for version, spec in VERSIONS.items():
        for model_name in make_models():
            preds = walk_forward(df, spec["features"], model_name)
            all_predictions.append(preds.assign(version=version, model=model_name))
            rows.append({"version": version, "stage": spec["stage"], "model": model_name,
                         **score(preds)})
    results = pd.DataFrame(rows)
    pd.concat(all_predictions).to_csv(backtest_dir / "predictions.csv", index=False)
    results.to_csv(backtest_dir / "model_comparison.csv", index=False)

    rounds = f"{min(EVAL_ROUNDS)}-{max(EVAL_ROUNDS)}"
    for version, spec in VERSIONS.items():
        print(f"\n=== {spec['stage']} (walk-forward, rounds {rounds}) ===")
        models = results[results["version"] == version].copy()
        bases = baseline_summary[baseline_summary["stage"] == spec["stage"]].rename(
            columns={"baseline": "model", "correct": "top1", "accuracy_pct": "top1_pct"})
        table = pd.concat([models, bases[["model", "races", "top1", "top1_pct"]]], ignore_index=True)
        cols = ["model", "top1", "top1_pct", "top3", "top3_pct", "avg_winner_prob", "log_loss"]
        print(table[cols].to_string(index=False, na_rep="-"))

    print("\nBaselines make a single pick, so top 3 and probabilities don't apply to them.")
    print("Best model per version = lowest log loss on the winner (a proper scoring rule).")

    print("\n=== Logistic regression coefficients (trained on rounds 1-17, standardized features) ===")
    print("Negative = higher feature value lowers win odds.")
    final_rounds = sorted(df["Round"].unique())
    for version, spec in VERSIONS.items():
        lr = make_models()["logistic_regression"].fit(df[spec["features"]], df[TARGET])
        print(f"\n{spec['stage']}:")
        print(logistic_coefficients(lr, spec["features"]).to_string())

    print()
    for version, spec in VERSIONS.items():
        best = results[results["version"] == version].sort_values("log_loss").iloc[0]
        model = make_models()[best["model"]].fit(df[spec["features"]], df[TARGET])
        path = config.OUTPUTS_DIR / f"model_{version}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "model": model,
                "model_name": best["model"],
                "version": version,
                "stage": spec["stage"],
                "features": spec["features"],
                "trained_on_rounds": final_rounds,
                "backtest": best.drop(["version", "stage", "model"]).to_dict(),
                "sklearn_version": sklearn.__version__,
            }, f)
        print(f"Saved {path.relative_to(config.PROJECT_ROOT)}: {best['model']} "
              f"(backtest log loss {best['log_loss']}), refit on rounds 1-{max(final_rounds)}")


if __name__ == "__main__":
    main()

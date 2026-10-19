# Austin F1 Predictor

## Goal

Predict the winner of the **2026 F1 United States Grand Prix** at the Circuit of the Americas (COTA), race day **October 25, 2026**, using Python and the [FastF1](https://docs.fastf1.dev/) library.

The output is a **win probability for every driver** on the grid, not a single pick. Probabilities must sum to 1.

## Project layout

```
src/       All Python code (data loading, features, model, simulation, CLI entry points)
data/      Raw and processed data (git-ignored)
cache/     FastF1 cache (git-ignored)
outputs/   Generated predictions, plots, and backtest reports
notebooks/ Exploratory Jupyter notebooks (kernel: "Python 3.12 (austin-f1-predictor .venv)")
```

- Put code in `src/` only. No scripts at the repo root.
- `data/`, `cache/` and `.venv/` are git-ignored; never commit them.
- Anything in `outputs/` must be reproducible by re-running code in `src/`.

## Environment

- Standard `venv` in `.venv/` (not conda), built from python.org Python 3.12. Dependencies pinned in `requirements.txt`.
- conda's base environment is usually active on this machine. Always call `.venv/bin/python` / `.venv/bin/python -m pip` explicitly so nothing installs into conda base.
- Core libraries: `fastf1`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `jupyter`.
- `requirements-core.txt` lists just these six direct dependencies; `requirements.txt` is the full locked `pip freeze` for exact reproduction. Update both when adding a package.

## FastF1 usage

- Always enable the cache before loading anything. Importing `src/config.py` does this (cache in `cache/`).
- Get the calendar with `fastf1.get_event_schedule(2026)`. Look up the COTA event and its session format (conventional vs. sprint weekend) from the schedule instead of hard-coding it.
- Load only what is needed. `session.load(laps=True, telemetry=False, weather=True, messages=False)` is usually enough; telemetry is slow and rarely needed for this model.
- FastF1 and its Jolpica/Ergast backend are rate-limited. Load sessions once, save processed results to `data/` (Parquet or CSV), and read from there on later runs.
- Sessions for the US GP weekend (practice, sprint, qualifying) may not exist yet. Code must handle missing sessions and produce a pre-weekend prediction from season data alone.

## Modeling principles

### 2026 is a regulation reset

The 2026 season brought new chassis and power-unit regulations, so the competitive order changed substantially. Weight data accordingly:

- **2026 season form is the main signal.** Car/team performance should come almost entirely from 2026 sessions.
- Within 2026, weight recent races more heavily (e.g. exponential decay by race recency), since teams develop through the season.
- **Pre-2026 seasons get a much lower weight.** Use them only for things that plausibly carry over across regulations: driver-level skill and racecraft, and driver track history at COTA. Never use pre-2026 data to estimate car pace.
- Keep all weights (season weight, recency decay rate, etc.) as named config values in one place, not scattered magic numbers.

### Useful signals (in rough priority order)

1. 2026 race results and race pace (clean-air long-run lap times, normalized per event).
2. 2026 qualifying pace (gap to pole, as a percentage).
3. US GP weekend sessions when available: qualifying/grid position first, then sprint and long runs from practice.
4. Reliability (2026 DNF rate per team/driver).
5. Low-weight priors: driver COTA history and pre-2026 driver form.

### Producing probabilities

- Turn driver strength estimates into a full finishing-order distribution, e.g. Monte Carlo race simulation or a Plackett–Luce/softmax model, and report P(win) for each driver.
- Include DNF probability so the favourite is never treated as a certainty.
- Use a fixed random seed for simulations so results are reproducible.

### Validation

- Backtest walk-forward on the 2026 races already run: to predict race N, use only data available before race N. No leakage from future sessions.
- Report log loss / Brier score for the winner, and compare against simple baselines (uniform, pole-sitter wins, championship-leader wins).
- Check calibration: across backtests, drivers given ~30% should win about 30% of the time.

## Outputs

- Main output: `outputs/usgp_2026_win_probabilities.csv` with columns `driver`, `team`, `win_prob`, sorted descending. Optional extra columns: `podium_prob`, `dnf_prob`.
- Note in the file name or a metadata field which sessions the prediction used (e.g. `pre_weekend`, `post_qualifying`), so pre-weekend and post-qualifying predictions can sit side by side.
- Backtest results go in `outputs/backtest/`.

## Conventions

- Identify drivers by their three-letter FastF1 abbreviation (`VER`, `HAM`, ...) and teams by FastF1's `TeamName`.
- Driver line-ups changed for 2026, so map history by driver, not by car number or seat.
- Keep functions small and testable. Loading data, building features, fitting the model, and simulating should be separate modules in `src/`.

## Workflow

- Repo: https://github.com/Daisy-Faith-Auma/austin-f1-predictor (remote `origin`, branch `main`).
- After each completed step, commit with a descriptive message and push to `origin main`.
- Never commit anything in `data/`, `cache/` or `.venv/`. Check `git status` before committing and stage files by name rather than `git add -A`.

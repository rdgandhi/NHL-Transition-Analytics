# Transition Efficiency & Win‑Probability Dashboard (MoneyPuck×NHL Stats)

Analyze how every **Canadian NHL club** turns shots into momentum — and explore those insights in a Streamlit app.

---

## 1) Quick‑Start

```bash
# clone & set up
$ git clone <repo>
$ cd NHL-Transition-Analytics
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt

# run the whole pipeline (data→ model→ app)
$ make all
```

```bash
# or step‑by‑step
make data     # downloads MoneyPuck shots **and** roster JSONs (player hover tool‑tips)
make model    # cross‑validates 5 regressors, saves the best (best_model.pkl)
make app      # launches Streamlit on http://localhost:8501
```

> **Prereqs:** Python 3.9+, pip, and on macOS Homebrew’s `libomp` if you plan to swap back to XGBoost (current default uses pure‑Python HistGradientBoosting).

---

## 2) Project Layout

```
.
├─ app/                # Streamlit front‑end
├─ data/
│   └─ raw/            # MoneyPuck shots + NHL roster JSONs
├─ models/             # best_model.pkl + metrics
├─ notebooks/          # (optional) EDA & sanity checks
├─ src/
│   ├─ ingest.py       # downloads & caches data
│   ├─ roster_utils.py # loads & flattens roster JSONs
│   ├─ features.py     # feature engineering pipeline
│   └─ models.py       # model selection + training
├─ requirements.txt
└─ Makefile
```

---

## 3) Data Sources (100% Free)

| Source                 | What we pull                                                              | Link                                                                                           |
| ---------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **MoneyPuck shots**    | `shots_<SEASON>.zip` event file (arena‑adjusted coords, xGoal, 300+ vars) | [https://peter-tanner.com/moneypuck/downloads/](https://peter-tanner.com/moneypuck/downloads/) |
| **NHL Stats API (v1)** | Current roster for each Canadian franchise                                | [https://api-web.nhle.com/](https://api-web.nhle.com/)                                         |

Season key is set in `src/ingest.py` (default **2023‑2024** so every Canadian team appears). Change `SEASON = "20242025"` if you want 2024‑25 preseason/test data.

---

## 4) Model Pipeline

1. **Pre‑processing** (`ColumnTransformer`)

   * One‑hot: `entry_type`, `manpower_situation`, `offensive_zone`, `pos`, `hand`
   * Numeric: shot distance, adjusted angle, period, game‑clock seconds
2. **Model selection** `src/models.py`

   * Evaluates **HGBR, GBR, RandomForest, ExtraTrees, ElasticNet** (5‑fold CV)
   * Prints mean±stdR², retrains the best on 80 % / tests on 20 %
3. **Serialisation** `best_model.pkl` stores the *full* pipeline (prep+ model) so the dashboard can `.predict()` with one call.

Typical CV leaderboard on 2023‑24 data:

```
HGBR   : 0.73 ± 0.01
GBR    : 0.71 ± 0.01
RF     : 0.70 ± 0.01
ET     : 0.70 ± 0.01
ElasticNet : 0.41 ± 0.01
🏆 Best = HGBR • hold‑out R² ≈ 0.74
```

---

## 5) Streamlit Dashboard

* **Shot Map** – rink‑normalised coordinates for the selected team, coloured by density.
* **Predicted ΔxG Histogram** – model’s expected goal differential in the 25s after each shot.
* **Hover tool‑tips** – show player name, position, handedness, shot distance (enabled via roster join).

> Swap seasons from the sidebar (future todo) or via `SEASON` env var.

---

## 6) Extending the Project

* **Add live win‑prob:** call MoneyPuck live JSON and update Streamlit every 10s.
* **Player leaderboards:** aggregate `xGD_shift` by `shooterPlayerId` to rank transition drivers.
* **Schedule API:** incorporate rest‑vs‑travel features (back‑to‑back detection) for model v2.

PRs welcome— open an issue or ping @your‑handle.

---

## 7) License

*Code* is MIT. *MoneyPuck data* is free for non‑commercial use with attribution—see [https://moneypuck.com](https://moneypuck.com).

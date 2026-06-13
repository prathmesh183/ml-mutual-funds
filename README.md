# RetireWise+ ML Portfolio Advisor 🏦

> An end-to-end machine learning system for personalized Indian mutual fund portfolio recommendations — built for [RetireWise+](https://retirewiseplus.com), a Pune-based AMFI-registered advisory firm.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![MFAPI](https://img.shields.io/badge/Data-MFAPI.in-green)](https://mfapi.in)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 🧠 Models Overview

This project combines **3 machine learning models** into a single pipeline:

| Model | Type | Task | Performance |
|-------|------|------|-------------|
| **Risk Profiler** | `RandomForestClassifier` | Classifies investor as Conservative / Moderate / Aggressive | **85.8% accuracy** |
| **SIP Return Predictor** | `GradientBoostingRegressor` | Predicts future portfolio value across 3 market scenarios | **R² = 0.9957** |
| **Portfolio Recommender** | Weighted Scoring Model | Selects 5–6 optimal mutual funds from a 17-fund database | Diversification-aware |

---

## 🗂️ Project Structure

```
retirewise-ml/
├── train_models.py          # Data generation + model training
├── app.py                   # Flask REST API
├── templates/
│   └── index.html           # Full website UI (matches RetireWise+ design)
├── models/
│   ├── risk_profiler.pkl    # Trained RandomForest
│   ├── sip_predictor.pkl    # Trained GradientBoosting
│   ├── sip_scaler.pkl       # StandardScaler for SIP features
│   ├── fund_database.pkl    # 17-fund database with metadata
│   └── model_meta.json      # Model metrics summary
├── data/
│   ├── risk_dataset.csv     # 3,000 synthetic investor profiles
│   ├── sip_dataset.csv      # 3,000 SIP simulation records
│   └── fund_database.csv    # Mutual fund reference data
└── requirements.txt
```

---

## ⚙️ How It Works

### 1. Risk Profiler (RandomForestClassifier)
**Features used:**
- Age, income, expenses, savings ratio
- Investment horizon, existing investments, dependents
- Behavioral inputs: crash response, experience level, focus

**Key insight:** The model can detect *true* risk tolerance even when stated risk contradicts behavior — e.g., an "aggressive" investor who would panic-sell in a crash gets classified as moderate.

### 2. SIP Return Predictor (GradientBoostingRegressor)
**Features used:**
- SIP amount, investment horizon, annual step-up %
- Risk category (from Model 1), simulated market condition

**Output:** Predicts future portfolio value under 3 market scenarios (bear/normal/bull) using log-transformed target to handle skewed distributions.

### 3. Portfolio Recommender (Weighted Scoring)
**Scoring formula:**
```
score = (CAGR_5y / 30) × 40      # Returns
      + (1 - expense/2) × 20     # Low cost bonus
      + (rating / 5) × 20        # Fund quality
      + ELSS_bonus (if applicable)
      + goal_specific_bonus
```
**Constraints:** Max 1 fund per AMC, max 6 funds total, no category overlap.

---

## 📡 Live Data — MFAPI.in

Real-time NAV data is fetched from [mfapi.in](https://mfapi.in) for each recommended fund:

```python
GET https://api.mfapi.in/mf/{scheme_code}
```

Mapped scheme codes for 17 major Indian mutual funds are included in `app.py`.

---

## 🚀 Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/retirewise-ml.git
cd retirewise-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all 3 models (generates synthetic data + saves .pkl files)
python train_models.py

# 4. Start Flask server
python app.py

# 5. Open browser
open http://localhost:5000
```

---

## 🔌 API Reference

### `POST /api/predict`
**Request body:**
```json
{
  "income": 80000,
  "expenses": 45000,
  "age": 32,
  "dependents": 2,
  "existing": 200000,
  "sip": 15000,
  "horizon": 15,
  "stepup": 10,
  "elss": "yes",
  "goals": ["retirement", "wealth"],
  "crash_behavior": "wait",
  "experience": "some",
  "focus": "balanced"
}
```
**Response:** Risk profile + 3-scenario projection + fund portfolio with live NAV

### `GET /api/nav/<scheme_code>`
Returns live NAV for any MFAPI scheme code

### `GET /api/models`
Returns model metadata (accuracy, R², feature list)

---

## 📊 Dataset

Synthetic datasets were generated using domain-specific rules:

- **Risk dataset:** 3,000 investor profiles with rule-based + noisy risk labels
- **SIP dataset:** 3,000 SIP simulations with realistic compound growth calculations including step-up
- **Fund database:** 17 hand-curated real Indian mutual funds with historical CAGR, expense ratio, AMC, and category metadata

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | scikit-learn (RandomForest, GradientBoosting) |
| Data | pandas, numpy |
| Backend | Flask |
| Frontend | Vanilla JS + CSS (RetireWise+ design system) |
| Live Data | MFAPI.in REST API |
| Serialization | joblib |

---

## 📈 Model Metrics

```
Risk Profiler (RandomForestClassifier):
  Accuracy:  85.8%
  F1 (avg):  0.86
  Classes:   Conservative / Moderate / Aggressive

SIP Return Predictor (GradientBoostingRegressor):
  R²:        0.9957
  Features:  SIP amount, horizon, step-up, risk level, market condition
  Transform: log1p on target (skewed distribution)

Portfolio Recommender (Weighted Scoring):
  Fund DB:   17 Indian mutual funds
  Output:    5–6 diversified funds, 1 per AMC max
  Live NAV:  via MFAPI.in
```

---

## 🙏 Acknowledgements

- [MFAPI.in](https://mfapi.in) — Free open-source Indian mutual fund data API
- [AMFI India](https://amfiindia.com) — Fund metadata reference
- [RetireWise+](https://retirewiseplus.com) — Domain expertise & use case (ARN-330249)

---

## ⚠️ Disclaimer

This project is for **educational and portfolio-building purposes only**. It does not constitute financial advice. The synthetic training data does not represent real investor outcomes. All investment decisions should be made after consulting a qualified AMFI-registered advisor.

---

*Built with ❤️ for RetireWise+ | Pune, Maharashtra*

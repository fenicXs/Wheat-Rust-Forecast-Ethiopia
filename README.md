# Wheat Rust Forecast Ethiopia

Improving wheat rust disease prediction in Ethiopia using machine learning and explainable AI, extending [Meyer et al. (2021)](https://doi.org/10.1371/journal.pone.0245697).

## Results

Binary classification (disease presence/absence) evaluated with **Leave-One-Year-Out Cross-Validation** (2010--2019), AUC-ROC:

| Rust Type | Meyer et al. (Logistic Regression) | Our Best Model | Improvement |
|-----------|-----------------------------------|----------------|-------------|
| **Stem**   | 0.772 | **0.833** (XGBoost) | +7.9% |
| **Stripe** | 0.598 | **0.719** (Gradient Boosting) | +20.2% |
| **Leaf**   | 0.638 | **0.798** (Gradient Boosting) | +25.1% |

### Key Findings
- **Per-rust model selection** outperforms a single model: XGBoost for stem rust (climate-driven), Gradient Boosting for stripe (race-driven) and leaf (spatial-driven) rust
- **Spatial cross-validation** (leave-one-block-out, K=8) confirms generalization with only 5--7% AUC degradation for stem/leaf rust and no degradation for stripe rust
- **SHAP analysis** identifies latitude-altitude interactions, pathogen race pressure, and crop growth stage as the dominant predictive features
- ERA5 climate features help stem rust (+1.1% AUC) but not stripe/leaf rust
- MODIS NDVI provides minimal direct signal; engineered features (anomaly, change) rank low

## Project Structure

```
Wheat-Rust-Forecast-Ethiopia/
├── src/
│   ├── extract_era5_modis.py      # Feature extraction from ERA5 + MODIS
│   ├── lag_features.py            # Spatial-temporal lag feature computation
│   ├── pipeline_v6_train.py       # Training pipeline (XGB/GB + TabNet)
│   └── pipeline_v7_analysis.py    # Analysis pipeline (Spatial CV, SHAP, calibration)
├── data/
│   ├── StemRust_with_climate_v2.csv
│   ├── StripeRust_with_climate_v2.csv
│   └── LeafRust_with_climate_v2.csv
├── results/
│   ├── stem_rust/                 # SHAP plots, calibration, spatial CV
│   ├── stripe_rust/
│   ├── leaf_rust/
│   ├── v6_results_summary.csv     # Per-year AUC for all models
│   └── v7_summary.csv            # Final summary with spatial CV
├── report/
│   ├── main.tex                   # Full research paper (CVPR format)
│   ├── figs/                      # All figures used in the paper
│   └── cvpr.sty                   # CVPR LaTeX style file
├── requirements.txt
└── README.md
```

## Data

Each CSV contains ~11,000--12,000 field survey records (2007--2019) from the Ethiopian Institute of Agricultural Research (EIAR), with the following columns:

| Column | Description |
|--------|-------------|
| `CountryID`, `Year`, `Month`, `Day` | Survey metadata |
| `Latitude`, `Longitude`, `Altitude` | Geographic location |
| `Area` | Field area (hectares) |
| `HostCultivar` | Wheat cultivar name |
| `GrowthStage` | Zadoks growth stage (1--9) |
| `*Severity`, `*Incidence` | Raw disease measurements |
| `Binary disease presence` | **Target variable** (0/1) |
| `era5_t2m`, `era5_d2m`, `era5_tp`, `era5_ssrd` | ERA5 climate: temperature (K), dewpoint (K), precipitation (m/day), solar radiation (J/m²) |
| `era5_*_lag1m`, `era5_*_lag2m` | 1-month and 2-month lagged climate |
| `era5_rh`, `era5_rh_lag1m` | Relative humidity (%) |
| `ndvi`, `ndvi_lag1m` | MODIS NDVI vegetation index |

### External Data Sources
- **Field surveys**: EIAR wheat rust surveillance network (2007--2019)
- **ERA5 reanalysis**: ECMWF 0.25° monthly climate data via Copernicus Climate Data Store
- **MODIS NDVI**: MOD13A2 1km 16-day vegetation index from NASA LPDAAC

## Reproducibility

### 1. Environment Setup

```bash
conda create -n wheat-rust python=3.11
conda activate wheat-rust
pip install -r requirements.txt
```

### 2. Run Training Pipeline

Trains per-rust best-of-breed models with LOYO-CV:

```bash
python src/pipeline_v6_train.py
```

Outputs to `results/` with per-year AUC scores and model comparisons.

### 3. Run Analysis Pipeline

Generates spatial CV, SHAP explanations, calibration curves, and threshold optimization:

```bash
python src/pipeline_v7_analysis.py
```

Outputs SHAP beeswarm plots, dependence plots, spatial vs temporal CV comparisons, calibration reliability diagrams, and threshold curves to `results/{stem,stripe,leaf}_rust/`.

### 4. Compile Report

Requires a LaTeX distribution (e.g., MiKTeX or TeX Live):

```bash
cd report
pdflatex main.tex
pdflatex main.tex  # run twice to resolve references
```

## Feature Engineering

The pipeline engineers 30+ features from the raw survey data:

| Category | Features | Description |
|----------|----------|-------------|
| **Geographic** | Latitude, Longitude, Altitude, interactions | Location and terrain |
| **Temporal** | DayOfYear, BiweekIntoMainSeason, GrowthStage | Phenological timing |
| **Cultivar** | cult_high/med/low_risk | Susceptibility encoding from historical data |
| **Race regime** | race_pressure, race_*_age/peak/active | Pathogen race dynamics |
| **Climate (ERA5)** | temp_suit, VPD, leaf_wet, infection_risk | Disease-relevant bioclimatic variables |
| **Vegetation** | ndvi_anomaly, ndvi_change, ndvi_x_alt | MODIS-derived stress indicators |
| **Spatial lags** | lag_25/50/100km_1/2m_prev/count | Nearby disease history (LOYO-safe) |

## Methods

- **Models**: XGBoost (stem rust), Gradient Boosting (stripe/leaf rust), with global Optuna hyperparameter tuning
- **Evaluation**: Leave-One-Year-Out CV (2010--2019), Meher cropping season only
- **Metric**: AUC-ROC (primary), F1-score with optimized thresholds
- **Explainability**: SHAP TreeExplainer with beeswarm, bar, and dependence plots
- **Validation**: Spatial leave-one-block-out CV (K=8 KMeans clusters) to test geographic generalization

## Authors

- Pradeep Krishnamoorthy
- Mithesh Chandrasekar
- Ruchitha Gowda
- Rohan John Varghese

**Supervisor**: Vinutha N

## References

Meyer, M., Bacha, N., Tesfaye, T. et al. (2021). Wheat rust epidemics damage Ethiopian wheat production: A decade of field surveillance reveals national-scale trends in past outbreaks. *PLOS ONE*, 16(2), e0245697.

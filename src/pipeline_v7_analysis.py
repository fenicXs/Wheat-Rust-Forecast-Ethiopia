"""
Wheat Rust ML Pipeline v7 — Spatial CV, Calibration, SHAP
==========================================================
Three analyses to complete the project:

1. SPATIAL CROSS-VALIDATION
   - Cluster survey locations into ~8 spatial blocks
   - Leave-One-Block-Out CV to test geographic generalization
   - Compare with LOYO-CV to assess spatial vs temporal overfitting

2. CALIBRATION & THRESHOLD OPTIMIZATION
   - Platt scaling (sigmoid) calibration via nested CV
   - Reliability diagrams + Brier scores
   - Optimal threshold via F1 maximization on LOYO-CV

3. SHAP INTERPRETABILITY
   - TreeExplainer for XGB (Stem) and GB (Stripe/Leaf)
   - Summary beeswarm plots per rust type
   - Top-10 feature importance bar charts
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import (roc_auc_score, brier_score_loss, f1_score,
                             precision_score, recall_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
import shap

# Lag features
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lag_features import compute_lag_features_loyo_safe, LAG_CONFIGS

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

CLIMATE_FILES = {
    "StemRust":   "StemRust_with_climate_v2.csv",
    "StripeRust": "StripeRust_with_climate_v2.csv",
    "LeafRust":   "LeafRust_with_climate_v2.csv",
}
CV_YEARS    = list(range(2010, 2020))
TARGET      = "Binary disease presence"
PAPER_FEATURES = ["Latitude", "Longitude", "Altitude", "DaysIntoMainSeason"]

TEMP_OPTIMA = {
    "StemRust":   {"opt": 20.0, "sigma": 6.0},
    "StripeRust": {"opt": 12.0, "sigma": 4.0},
    "LeafRust":   {"opt": 18.0, "sigma": 5.0},
}

STEM_XGB_PARAMS = {
    "n_estimators": 342, "max_depth": 3, "learning_rate": 0.03,
    "subsample": 0.955, "colsample_bytree": 0.985,
    "reg_alpha": 4.9, "reg_lambda": 3.5e-5,
    "min_child_weight": 17, "gamma": 2.4,
}

N_SPATIAL_BLOCKS = 8


# ─────────────────────────────────────────────
# FEATURE ENGINEERING (reused from v6)
# ─────────────────────────────────────────────

CULTIVAR_CANONICAL = {
    "digelu": "Digalu", "digalu": "Digalu", "kubsa": "Kubsa", "kakaba": "Kakaba",
    "ogolcho": "Ogolcho", "dandaa": "Dandaa", "danda'a": "Dandaa", "galema": "Galema",
    "local": "Local", "locals": "Local", "improved": "Improved", "unknown": "Unknown",
}

def canonicalise_cultivar(name):
    if pd.isna(name): return "Unknown"
    return CULTIVAR_CANONICAL.get(str(name).strip().lower(), str(name).strip())

def cultivar_risk_features(df, rust_type):
    cult, year = df["HostCultivar_clean"], df["Year"]
    if rust_type == "StemRust":
        df["cult_high_risk"] = ((cult == "Digalu") & (year >= 2013)).astype(int)
        df["cult_med_risk"]  = (cult.isin(["Ogolcho", "Dandaa"]) & (year >= 2019)).astype(int)
        df["cult_local"]     = (cult == "Local").astype(int)
        df["cult_low_risk"]  = 0
    elif rust_type == "StripeRust":
        df["cult_high_risk"] = (cult.isin(["Kubsa", "Galema"]) & (year >= 2010)).astype(int)
        df["cult_med_risk"]  = ((cult == "Digalu") & (year >= 2016)).astype(int)
        df["cult_low_risk"]  = (cult.isin(["Ogolcho", "Dandaa"]) & (year >= 2016)).astype(int)
        df["cult_local"]     = 0
    elif rust_type == "LeafRust":
        df["cult_high_risk"] = (cult == "Local").astype(int)
        df["cult_med_risk"]  = (cult.isin(["Kubsa", "Kakaba"])).astype(int)
        df["cult_low_risk"]  = (cult == "Improved").astype(int)
        df["cult_local"]     = 0
    return df

def encode_race_regimes(df, rust_type):
    year = df["Year"]
    df = df.copy()
    if rust_type == "StemRust":
        df["race_TKTTF_active"]    = (year >= 2013).astype(int)
        df["race_TKTTF_age"]       = np.clip(year - 2013, 0, None) * (year >= 2013)
        df["race_TKTTF_peak"]      = year.isin([2013, 2014, 2015]).astype(int)
        df["race_new_2019_active"] = (year >= 2019).astype(int)
        pmap = {y: (0 if y < 2013 else 2 if y <= 2015 else 1 if y <= 2018 else 3)
                for y in range(2007, 2020)}
        df["race_pressure"]  = year.map(pmap).fillna(0)
        df["race_x_cult"]    = df["race_TKTTF_active"] * df.get("cult_high_risk", 0)
    elif rust_type == "StripeRust":
        df["race_PstS6_active"]   = (year >= 2010).astype(int)
        df["race_PstS6_age"]      = np.clip(year - 2010, 0, None)
        df["race_PstS6_peak"]     = year.isin([2010, 2011]).astype(int)
        df["race_PstS11_active"]  = (year >= 2016).astype(int)
        df["race_PstS11_age"]     = np.clip(year - 2016, 0, None) * (year >= 2016)
        df["race_PstS11_peak"]    = year.isin([2016, 2017, 2018]).astype(int)
        df["race_dual_active"]    = (year >= 2016).astype(int)
        pmap = {y: (0 if y < 2010 else 2 if y <= 2011 else 1 if y <= 2015 else 3 if y <= 2018 else 2)
                for y in range(2007, 2020)}
        df["race_pressure"]       = year.map(pmap).fillna(0)
        df["race_x_cult_PstS6"]   = df["race_PstS6_active"] * df.get("cult_high_risk", 0)
        df["race_x_cult_PstS11"]  = df["race_PstS11_active"] * df.get("cult_med_risk", 0)
        df["race_x_cult"]         = df["race_x_cult_PstS6"] + df["race_x_cult_PstS11"]
    elif rust_type == "LeafRust":
        df["race_pressure"]   = (year >= 2014).astype(int)
        df["race_endemic"]    = 1
        df["race_year_trend"] = np.clip(year - 2010, 0, None)
        df["race_x_cult"]     = df["race_pressure"] * df.get("cult_high_risk", 0)
    return df

def get_race_feature_names(rust_type):
    if rust_type == "StemRust":
        return ["race_TKTTF_active", "race_TKTTF_age", "race_TKTTF_peak",
                "race_new_2019_active", "race_pressure", "race_x_cult"]
    elif rust_type == "StripeRust":
        return ["race_PstS6_active", "race_PstS6_age", "race_PstS6_peak",
                "race_PstS11_active", "race_PstS11_age", "race_PstS11_peak",
                "race_dual_active", "race_pressure",
                "race_x_cult_PstS6", "race_x_cult_PstS11", "race_x_cult"]
    elif rust_type == "LeafRust":
        return ["race_pressure", "race_endemic", "race_x_cult", "race_year_trend"]

def thermal_suitability(temp_c, opt, sigma):
    return np.exp(-0.5 * ((temp_c - opt) / sigma) ** 2)

def vpd_kpa(t2m_K, rh_pct):
    T_C = t2m_K - 273.15
    es = 0.6108 * np.exp(17.27 * T_C / (T_C + 237.3))
    ea = es * np.clip(rh_pct, 0, 105) / 100.0
    return np.clip(es - ea, 0, None)

def engineer_ndvi_smart(df):
    ndvi_cols = []
    if "ndvi" not in df.columns:
        return df, ndvi_cols
    df["_lat_bin"] = (df["Latitude"] * 2).round() / 2
    df["_lon_bin"] = (df["Longitude"] * 2).round() / 2
    clim_mean = df.groupby(["_lat_bin", "_lon_bin", "Month"])["ndvi"].transform("mean")
    df["ndvi_anomaly"] = df["ndvi"] - clim_mean
    ndvi_cols.append("ndvi_anomaly")
    if "ndvi_lag1m" in df.columns:
        df["ndvi_change"] = df["ndvi"] - df["ndvi_lag1m"]
        ndvi_cols.append("ndvi_change")
    df["ndvi_x_alt"] = df["ndvi"] * df["Altitude"] / 1000
    ndvi_cols.append("ndvi_x_alt")
    if "GrowthStage" in df.columns:
        df["ndvi_x_growth"] = df["ndvi"] * df["GrowthStage"]
        ndvi_cols.append("ndvi_x_growth")
    ndvi_cols.extend(["ndvi", "ndvi_lag1m"])
    df.drop(columns=["_lat_bin", "_lon_bin"], inplace=True)
    for col in ndvi_cols:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med if not pd.isna(med) else 0)
    return df, [c for c in ndvi_cols if c in df.columns]


def engineer_all_features(df, rust_type):
    df = df.copy()
    df["Altitude"] = df["Altitude"].replace([-999.99, -1000.0], np.nan)
    df["Altitude_missing"] = df["Altitude"].isna().astype(int)
    df["Altitude"] = df["Altitude"].fillna(df.loc[df["Altitude"] > 0, "Altitude"].median())
    df["GrowthStage"] = df["GrowthStage"].replace(-9, np.nan)
    df["GrowthStage_missing"] = df["GrowthStage"].isna().astype(int)
    df["GrowthStage"] = df["GrowthStage"].fillna(df["GrowthStage"].median())

    season_start = pd.to_datetime(df[["Year"]].assign(month=8, day=1))
    survey_date = pd.to_datetime(df[["Year", "Month", "Day"]].rename(
        columns={"Month": "month", "Day": "day"}), errors="coerce")
    if survey_date.isna().any():
        survey_date = survey_date.fillna(season_start)

    df["DaysIntoMainSeason"]   = (survey_date - season_start).dt.days
    df["BiweekIntoMainSeason"] = (df["DaysIntoMainSeason"] / 14) + 1
    df["DayOfYear"]            = survey_date.dt.dayofyear
    df["IsMeherSeason"]        = df["Month"].between(8, 12).astype(int)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["AltitudeBand"] = pd.cut(
        df["Altitude"], bins=[0, 1800, 2200, 2500, 2800, 5000],
        labels=[1, 2, 3, 4, 5]).astype(float).fillna(3)
    df["Lat_x_Alt"]  = df["Latitude"] * df["Altitude"] / 1000
    df["Lon_x_Alt"]  = df["Longitude"] * df["Altitude"] / 1000
    df["Lat_x_Lon"]  = df["Latitude"] * df["Longitude"]
    df["Lat_sq"]     = df["Latitude"] ** 2
    df["Lon_sq"]     = df["Longitude"] ** 2
    df["Alt_x_Time"] = df["Altitude"] * df["DaysIntoMainSeason"] / 1000

    df["HostCultivar_clean"] = df["HostCultivar"].apply(canonicalise_cultivar)
    df = cultivar_risk_features(df, rust_type)
    df["Area"] = df["Area"].replace([-999.99, -1000.0], np.nan)
    df["Area_log"]     = np.log1p(df["Area"].clip(lower=0).fillna(0))
    df["Area_missing"] = df["Area"].isna().astype(int)
    df = encode_race_regimes(df, rust_type)

    opt = TEMP_OPTIMA[rust_type]["opt"]
    sig = TEMP_OPTIMA[rust_type]["sigma"]
    climate_cols = []
    for lag in ["", "_lag1m", "_lag2m"]:
        t2m = f"era5_t2m{lag}"; tp = f"era5_tp{lag}"
        rh = f"era5_rh{lag}" if lag != "_lag2m" else None
        if t2m not in df.columns: continue
        t_C = df[t2m] - 273.15
        df[f"temp_C{lag}"] = t_C;                climate_cols.append(f"temp_C{lag}")
        df[f"temp_suit{lag}"] = thermal_suitability(t_C, opt, sig); climate_cols.append(f"temp_suit{lag}")
        df[f"precip_mm{lag}"] = df[tp] * 1000;   climate_cols.append(f"precip_mm{lag}")
        if rh and rh in df.columns:
            df[f"vpd{lag}"] = vpd_kpa(df[t2m], df[rh]);   climate_cols.append(f"vpd{lag}")
            df[f"leaf_wet{lag}"] = (df[rh] / 100.0) * np.log1p(df[f"precip_mm{lag}"])
            climate_cols.append(f"leaf_wet{lag}")
            df[f"inf_risk{lag}"] = df[f"temp_suit{lag}"] * df[f"leaf_wet{lag}"]
            climate_cols.append(f"inf_risk{lag}")
    if "temp_C" in df.columns and "temp_C_lag1m" in df.columns:
        df["temp_trend"] = df["temp_C"] - df["temp_C_lag1m"]; climate_cols.append("temp_trend")
    if all(f"precip_mm{l}" in df.columns for l in ["", "_lag1m", "_lag2m"]):
        df["precip_3m_cum"] = df["precip_mm"] + df["precip_mm_lag1m"] + df["precip_mm_lag2m"]
        climate_cols.append("precip_3m_cum")
    for col in climate_cols:
        med = df[col].median()
        df[col] = df[col].fillna(med if not pd.isna(med) else 0)

    df, ndvi_cols = engineer_ndvi_smart(df)
    return df, climate_cols, ndvi_cols


V1_FEATURES = [
    "Latitude", "Longitude", "Altitude", "DaysIntoMainSeason",
    "BiweekIntoMainSeason", "DayOfYear", "Month_sin", "Month_cos", "IsMeherSeason",
    "AltitudeBand", "Lat_x_Alt", "Lon_x_Alt", "Lat_x_Lon", "Lat_sq", "Lon_sq",
    "Alt_x_Time",
    "cult_high_risk", "cult_med_risk", "cult_low_risk", "cult_local",
    "GrowthStage", "Area_log", "Altitude_missing", "GrowthStage_missing", "Area_missing",
]


def select_features_mi_corr(df, candidate_features, target_col, corr_threshold=0.85):
    avail = [f for f in candidate_features if f in df.columns]
    X = df[avail].fillna(0).values
    y = df[target_col].values
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({"feature": avail, "mi": mi_scores}).sort_values("mi", ascending=False)
    selected = mi_df["feature"].tolist()
    if len(selected) > 1:
        corr_matrix = df[selected].fillna(0).corr().abs()
        mi_rank = {f: i for i, f in enumerate(selected)}
        to_drop = set()
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                if corr_matrix.iloc[i, j] > corr_threshold:
                    drop = selected[j] if mi_rank[selected[i]] < mi_rank[selected[j]] else selected[i]
                    to_drop.add(drop)
        selected = [f for f in selected if f not in to_drop]
    return selected, mi_df


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

def make_gb():
    return GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        min_samples_leaf=15, subsample=0.8, max_features="sqrt", random_state=42,
    )

def make_stem_xgb():
    return XGBClassifier(
        **STEM_XGB_PARAMS, random_state=42, eval_metric="logloss", verbosity=0,
    )

def get_best_model_fn(rust_name):
    """Return the best model factory for each rust type (from v6 results)."""
    if rust_name == "StemRust":
        return make_stem_xgb
    else:
        return make_gb


# ─────────────────────────────────────────────
# 1. SPATIAL CROSS-VALIDATION
# ─────────────────────────────────────────────

def create_spatial_blocks(df, n_blocks=N_SPATIAL_BLOCKS):
    """Cluster survey locations into spatial blocks using KMeans on lat/lon."""
    coords = df[["Latitude", "Longitude"]].values
    km = KMeans(n_clusters=n_blocks, random_state=42, n_init=10)
    df["spatial_block"] = km.fit_predict(coords)
    return df, km

def run_spatial_cv(df_cv, feats, model_fn, n_blocks=N_SPATIAL_BLOCKS):
    """Leave-One-Block-Out spatial cross-validation."""
    avail = [f for f in feats if f in df_cv.columns]
    block_aucs = {}

    for block_id in range(n_blocks):
        te = df_cv["spatial_block"] == block_id
        tr = ~te
        y_tr = df_cv.loc[tr, TARGET].values
        y_te = df_cv.loc[te, TARGET].values

        if len(np.unique(y_te)) < 2 or te.sum() < 10:
            continue

        X_tr = df_cv.loc[tr, avail].fillna(0).values
        X_te = df_cv.loc[te, avail].fillna(0).values

        model = model_fn()
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        block_aucs[block_id] = roc_auc_score(y_te, y_prob)

    return block_aucs

def plot_spatial_blocks(df_cv, rust_name, out_dir):
    """Plot survey locations colored by spatial block."""
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(df_cv["Longitude"], df_cv["Latitude"],
                        c=df_cv["spatial_block"], cmap="tab10", s=3, alpha=0.4)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title(f"{rust_name} — Spatial Blocks (K={N_SPATIAL_BLOCKS})", fontsize=13, fontweight="bold")
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(N_SPATIAL_BLOCKS))
    cbar.set_label("Block ID")

    # Annotate block sizes
    for bid in range(N_SPATIAL_BLOCKS):
        mask = df_cv["spatial_block"] == bid
        cx = df_cv.loc[mask, "Longitude"].mean()
        cy = df_cv.loc[mask, "Latitude"].mean()
        ax.annotate(f"B{bid}\n(n={mask.sum()})", (cx, cy),
                   fontsize=8, fontweight="bold", ha="center",
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{rust_name}_spatial_blocks.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_spatial_vs_temporal(spatial_aucs, temporal_aucs, rust_name, out_dir):
    """Compare spatial CV vs temporal LOYO-CV."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Spatial
    ax = axes[0]
    blocks = sorted(spatial_aucs.keys())
    vals = [spatial_aucs[b] for b in blocks]
    bars = ax.bar(range(len(blocks)), vals, color="#4393C3", edgecolor="white", width=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    mean_s = np.mean(vals)
    ax.axhline(mean_s, color="red", ls="--", lw=1.5, label=f"Mean: {mean_s:.3f}")
    ax.axhline(0.5, color="lightgray", ls=":", lw=1)
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels([f"Block {b}" for b in blocks], rotation=45, fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_title("Spatial CV (Leave-One-Block-Out)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Temporal
    ax = axes[1]
    years = sorted(temporal_aucs.keys())
    vals_t = [temporal_aucs[y] for y in years]
    bars = ax.bar(range(len(years)), vals_t, color="#D6604D", edgecolor="white", width=0.6)
    for bar, val in zip(bars, vals_t):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    mean_t = np.mean(vals_t)
    ax.axhline(mean_t, color="red", ls="--", lw=1.5, label=f"Mean: {mean_t:.3f}")
    ax.axhline(0.5, color="lightgray", ls=":", lw=1)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_title("Temporal CV (Leave-One-Year-Out)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"{rust_name} — Spatial vs Temporal Generalization", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{rust_name}_spatial_vs_temporal.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# 2. CALIBRATION & THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────

def run_loyo_calibrated(df_cv, feats, model_fn):
    """LOYO-CV collecting raw and calibrated probabilities + true labels."""
    avail = [f for f in feats if f in df_cv.columns]
    all_y_true, all_y_raw, all_y_cal = [], [], []
    yearly_raw_aucs = {}
    yearly_cal_aucs = {}

    for test_year in CV_YEARS:
        tr = df_cv["Year"] != test_year
        te = df_cv["Year"] == test_year
        y_tr = df_cv.loc[tr, TARGET].values
        y_te = df_cv.loc[te, TARGET].values

        if len(np.unique(y_te)) < 2:
            continue

        X_tr = df_cv.loc[tr, avail].fillna(0).values
        X_te = df_cv.loc[te, avail].fillna(0).values

        # Raw model
        model = model_fn()
        model.fit(X_tr, y_tr)
        y_raw = model.predict_proba(X_te)[:, 1]

        # Calibrated model (Platt scaling with temporal inner CV)
        # Use 2nd-to-last year as calibration set
        train_years = sorted(df_cv.loc[tr, "Year"].unique())
        cal_year = train_years[-1]
        inner_tr = tr & (df_cv["Year"] != cal_year)
        inner_cal = tr & (df_cv["Year"] == cal_year)

        X_itr = df_cv.loc[inner_tr, avail].fillna(0).values
        y_itr = df_cv.loc[inner_tr, TARGET].values
        X_ical = df_cv.loc[inner_cal, avail].fillna(0).values
        y_ical = df_cv.loc[inner_cal, TARGET].values

        base_model = model_fn()
        base_model.fit(X_itr, y_itr)
        # Get calibration data probabilities
        cal_probs = base_model.predict_proba(X_ical)[:, 1]

        # Fit Platt scaling (logistic regression on probabilities)
        from sklearn.linear_model import LogisticRegression as LR
        platt = LR(C=1e6, max_iter=1000, solver="lbfgs")
        platt.fit(cal_probs.reshape(-1, 1), y_ical)
        y_cal = platt.predict_proba(y_raw.reshape(-1, 1))[:, 1]

        all_y_true.extend(y_te)
        all_y_raw.extend(y_raw)
        all_y_cal.extend(y_cal)

        yearly_raw_aucs[test_year] = roc_auc_score(y_te, y_raw)
        yearly_cal_aucs[test_year] = roc_auc_score(y_te, y_cal)

    return (np.array(all_y_true), np.array(all_y_raw), np.array(all_y_cal),
            yearly_raw_aucs, yearly_cal_aucs)


def find_optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 score."""
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def plot_calibration(y_true, y_raw, y_cal, rust_name, out_dir):
    """Reliability diagram comparing raw vs calibrated probabilities."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reliability diagram
    ax = axes[0]
    for label, probs, color in [("Raw model", y_raw, "#D6604D"), ("Calibrated (Platt)", y_cal, "#4393C3")]:
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", color=color, lw=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives", fontsize=11)
    ax.set_title("Reliability Diagram", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Histogram of predicted probabilities
    ax = axes[1]
    ax.hist(y_raw, bins=30, alpha=0.5, color="#D6604D", label="Raw", density=True)
    ax.hist(y_cal, bins=30, alpha=0.5, color="#4393C3", label="Calibrated", density=True)
    ax.set_xlabel("Predicted probability", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Probability Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    brier_raw = brier_score_loss(y_true, y_raw)
    brier_cal = brier_score_loss(y_true, y_cal)
    plt.suptitle(f"{rust_name} — Calibration  |  Brier: raw={brier_raw:.4f}, cal={brier_cal:.4f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{rust_name}_calibration.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return brier_raw, brier_cal


def plot_threshold_analysis(y_true, y_prob, rust_name, out_dir):
    """Plot F1, precision, recall vs threshold."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))

    opt_idx = np.argmax(f1s)
    opt_t = thresholds[opt_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, f1s, color="#D6604D", lw=2.5, label="F1 Score")
    ax.plot(thresholds, precs, color="#4393C3", lw=1.5, ls="--", label="Precision")
    ax.plot(thresholds, recs, color="#1A7C41", lw=1.5, ls="--", label="Recall")
    ax.axvline(opt_t, color="gray", ls=":", lw=1.5,
               label=f"Optimal: t={opt_t:.2f}, F1={f1s[opt_idx]:.3f}")
    ax.axvline(0.5, color="black", ls=":", lw=1, alpha=0.5, label="Default t=0.50")
    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"{rust_name} — Threshold Optimization", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{rust_name}_threshold.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return opt_t, f1s[opt_idx]


# ─────────────────────────────────────────────
# 3. SHAP INTERPRETABILITY
# ─────────────────────────────────────────────

def compute_shap_values(df_cv, feats, model_fn, rust_name, out_dir):
    """Train on all Meher data, compute SHAP values, generate plots."""
    avail = [f for f in feats if f in df_cv.columns]
    X = df_cv[avail].fillna(0).values
    y = df_cv[TARGET].values

    model = model_fn()
    model.fit(X, y)

    # Workaround: shap 0.49 can't parse xgboost 3.x base_score '[0.xxx]' format
    # Patch Python's float() temporarily isn't feasible, so patch at source
    import builtins
    _orig_float = builtins.float
    def _safe_float(x):
        if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
            return _orig_float(x.strip("[]"))
        return _orig_float(x)
    builtins.float = _safe_float
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    finally:
        builtins.float = _orig_float

    # For GradientBoosting, shap_values is 1D array per sample
    # For XGBoost binary, it might be 2D — take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Summary beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=avail, show=False, max_display=20)
    plt.title(f"{rust_name} — SHAP Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{rust_name}_shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    # Bar plot (mean |SHAP|)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=avail, plot_type="bar",
                     show=False, max_display=20)
    plt.title(f"{rust_name} — Mean |SHAP| Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{rust_name}_shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    # Save feature importance table
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": avail,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(os.path.join(out_dir, f"{rust_name}_shap_importance.csv"), index=False)

    # Top-3 dependence plots
    top3_idx = np.argsort(mean_abs_shap)[::-1][:3]
    for rank, idx in enumerate(top3_idx):
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(idx, shap_values, X, feature_names=avail, show=False, ax=ax)
        plt.title(f"{rust_name} — SHAP Dependence: {avail[idx]}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{rust_name}_shap_dep_{rank+1}_{avail[idx]}.png"),
                   dpi=150, bbox_inches="tight")
        plt.close("all")

    return shap_df


# ─────────────────────────────────────────────
# LOYO-CV (for temporal AUC reference)
# ─────────────────────────────────────────────

def run_loyo_cv(df_cv, feats, model_fn):
    avail = [f for f in feats if f in df_cv.columns]
    yearly_aucs = {}
    for test_year in CV_YEARS:
        tr = df_cv["Year"] != test_year
        te = df_cv["Year"] == test_year
        y_tr = df_cv.loc[tr, TARGET].values
        y_te = df_cv.loc[te, TARGET].values
        if len(np.unique(y_te)) < 2:
            continue
        X_tr = df_cv.loc[tr, avail].fillna(0).values
        X_te = df_cv.loc[te, avail].fillna(0).values
        model = model_fn()
        model.fit(X_tr, y_tr)
        y_sc = model.predict_proba(X_te)[:, 1]
        yearly_aucs[test_year] = roc_auc_score(y_te, y_sc)
    return yearly_aucs


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

class TeeOutput:
    def __init__(self, log_path):
        self.terminal = sys.__stdout__
        self.log = open(log_path, "w", buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    log_path = os.path.join(OUT_DIR, "v7_run.log")
    sys.stdout = TeeOutput(log_path)
    sys.stderr = sys.stdout

    print("=" * 65)
    print("  Wheat Rust ML Pipeline v7")
    print("  Spatial CV + Calibration + SHAP Interpretability")
    print("=" * 65)
    print(f"  Data: {DATA_DIR}")
    print(f"  Output: {OUT_DIR}")

    summary_rows = []

    for rust_name, climate_fname in CLIMATE_FILES.items():
        print(f"\n{'='*65}")
        print(f"  {rust_name}")
        print(f"{'='*65}")

        # ── Data prep (same as v6) ──
        df = pd.read_csv(os.path.join(DATA_DIR, climate_fname))
        print(f"  Loaded {len(df)} surveys")

        df, climate_cols, ndvi_cols = engineer_all_features(df, rust_name)
        race_names = [f for f in get_race_feature_names(rust_name) if f in df.columns]

        print(f"  Computing lag features...")
        meher_idx = df.index[df["IsMeherSeason"] == 1]
        df_m, lag_names = compute_lag_features_loyo_safe(
            df.loc[meher_idx].copy(), target_col=TARGET,
            configs=LAG_CONFIGS, cv_years=CV_YEARS, verbose=True)
        for col in lag_names:
            df.loc[meher_idx, col] = df_m[col].values
            df[col] = df[col].fillna(df[TARGET].mean() if "_prev" in col else 0)

        v1_feats = [f for f in V1_FEATURES if f in df.columns]
        lag_feats = [f for f in lag_names if f in df.columns]
        v3_feats = v1_feats + lag_feats + race_names
        full_candidate = v3_feats + climate_cols + ndvi_cols
        full_candidate = list(dict.fromkeys(full_candidate))

        meher_mask = (df["Year"].isin(CV_YEARS)) & (df["IsMeherSeason"] == 1)
        selected_feats, mi_df = select_features_mi_corr(
            df[meher_mask], full_candidate, TARGET, corr_threshold=0.85)
        print(f"  Feature selection: {len(full_candidate)} → {len(selected_feats)}")

        # Per-rust best model & features
        model_fn = get_best_model_fn(rust_name)
        if rust_name == "StemRust":
            best_feats = selected_feats  # XGB + full features
        else:
            best_feats = v3_feats  # GB + v3 features

        df_cv = df[meher_mask].copy().reset_index(drop=True)
        rust_dir = os.path.join(OUT_DIR, rust_name)
        os.makedirs(rust_dir, exist_ok=True)

        # ──────────────────────────────
        # 1. SPATIAL CV
        # ──────────────────────────────
        print(f"\n  --- 1. SPATIAL CROSS-VALIDATION ---")
        df_cv, km = create_spatial_blocks(df_cv, N_SPATIAL_BLOCKS)

        # Block stats
        for bid in range(N_SPATIAL_BLOCKS):
            mask = df_cv["spatial_block"] == bid
            n = mask.sum()
            prev = df_cv.loc[mask, TARGET].mean()
            lat_m = df_cv.loc[mask, "Latitude"].mean()
            lon_m = df_cv.loc[mask, "Longitude"].mean()
            print(f"    Block {bid}: n={n:5d}, prevalence={prev:.3f}, "
                  f"center=({lat_m:.2f}, {lon_m:.2f})")

        plot_spatial_blocks(df_cv, rust_name, rust_dir)

        spatial_aucs = run_spatial_cv(df_cv, best_feats, model_fn)
        temporal_aucs = run_loyo_cv(df_cv, best_feats, model_fn)

        mean_spatial = np.mean(list(spatial_aucs.values()))
        mean_temporal = np.mean(list(temporal_aucs.values()))
        print(f"\n    Spatial CV mean AUC:  {mean_spatial:.4f}")
        print(f"    Temporal CV mean AUC: {mean_temporal:.4f}")
        print(f"    Gap (temporal - spatial): {mean_temporal - mean_spatial:+.4f}")

        for bid, auc in sorted(spatial_aucs.items()):
            print(f"      Block {bid}: AUC = {auc:.3f}")

        plot_spatial_vs_temporal(spatial_aucs, temporal_aucs, rust_name, rust_dir)

        # ──────────────────────────────
        # 2. CALIBRATION
        # ──────────────────────────────
        print(f"\n  --- 2. CALIBRATION & THRESHOLD ---")
        y_true, y_raw, y_cal, raw_aucs, cal_aucs = run_loyo_calibrated(
            df_cv, best_feats, model_fn)

        brier_raw, brier_cal = plot_calibration(y_true, y_raw, y_cal, rust_name, rust_dir)
        auc_raw = roc_auc_score(y_true, y_raw)
        auc_cal = roc_auc_score(y_true, y_cal)

        print(f"    Brier score (raw):        {brier_raw:.4f}")
        print(f"    Brier score (calibrated): {brier_cal:.4f}")
        print(f"    AUC (raw):        {auc_raw:.4f}")
        print(f"    AUC (calibrated): {auc_cal:.4f}")

        # Threshold optimization (on calibrated probabilities)
        opt_thresh, opt_f1 = find_optimal_threshold(y_true, y_cal)
        def_f1 = f1_score(y_true, (y_cal >= 0.5).astype(int), zero_division=0)
        print(f"    F1 at default t=0.50: {def_f1:.3f}")
        print(f"    Optimal threshold:    t={opt_thresh:.2f}, F1={opt_f1:.3f}")

        # Also compute precision/recall at optimal threshold
        y_pred_opt = (y_cal >= opt_thresh).astype(int)
        prec = precision_score(y_true, y_pred_opt, zero_division=0)
        rec = recall_score(y_true, y_pred_opt, zero_division=0)
        print(f"    At optimal threshold: precision={prec:.3f}, recall={rec:.3f}")

        plot_threshold_analysis(y_true, y_cal, rust_name, rust_dir)

        # ──────────────────────────────
        # 3. SHAP INTERPRETABILITY
        # ──────────────────────────────
        print(f"\n  --- 3. SHAP INTERPRETABILITY ---")
        shap_df = compute_shap_values(df_cv, best_feats, model_fn, rust_name, rust_dir)

        print(f"    Top-10 features by mean |SHAP|:")
        for _, row in shap_df.head(10).iterrows():
            print(f"      {row['feature']:<28s}: {row['mean_abs_shap']:.4f}")

        # Summary row
        summary_rows.append({
            "Rust": rust_name,
            "Model": "XGB" if rust_name == "StemRust" else "GB",
            "LOYO_AUC": round(mean_temporal, 4),
            "Spatial_AUC": round(mean_spatial, 4),
            "Gap": round(mean_temporal - mean_spatial, 4),
            "Brier_raw": round(brier_raw, 4),
            "Brier_cal": round(brier_cal, 4),
            "Optimal_threshold": round(opt_thresh, 2),
            "F1_default": round(def_f1, 3),
            "F1_optimal": round(opt_f1, 3),
            "Top_feature": shap_df.iloc[0]["feature"],
        })

    # Final summary
    print(f"\n\n{'='*65}")
    print("  FINAL SUMMARY — v7")
    print(f"{'='*65}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "v7_summary.csv"), index=False)

    for _, row in summary_df.iterrows():
        print(f"\n  {row['Rust']} ({row['Model']}):")
        print(f"    LOYO AUC:        {row['LOYO_AUC']:.4f}")
        print(f"    Spatial AUC:     {row['Spatial_AUC']:.4f}  (gap: {row['Gap']:+.4f})")
        print(f"    Brier (raw/cal): {row['Brier_raw']:.4f} / {row['Brier_cal']:.4f}")
        print(f"    Threshold:       {row['Optimal_threshold']:.2f} (F1: {row['F1_default']:.3f} → {row['F1_optimal']:.3f})")
        print(f"    Top feature:     {row['Top_feature']}")

    print(f"\n  Outputs: {OUT_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()

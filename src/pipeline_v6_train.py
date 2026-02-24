"""
Wheat Rust ML Pipeline v6 — Smarter NDVI + Per-Rust Selection + TabNet
=======================================================================
Three improvements over v5.1:

1. SMARTER NDVI FEATURES
   Raw NDVI ranked 43rd/58 in MI — useless. We now compute:
   - ndvi_anomaly:  observed - location/month mean (below-normal = stress)
   - ndvi_change:   current - lagged (rapid decline = possible rust damage)
   - ndvi_x_alt:    NDVI × Altitude interaction
   - ndvi_x_growth: NDVI × GrowthStage interaction

2. PER-RUST BEST-OF-BREED MODEL
   - StemRust:   XGBoost + full features (climate-driven)
   - StripeRust: GradientBoosting + v3 features (race-driven)
   - LeafRust:   GradientBoosting + v3 features (spatial-driven)

3. TABNET DEEP LEARNING
   - Attention-based tabular neural network
   - Built-in feature selection via sequential attention
   - May capture non-linear interactions missed by tree models

Current best to beat:
  StemRust=0.834, StripeRust=0.719, LeafRust=0.798
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import optuna
import torch

# Lag features
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lag_features import compute_lag_features_loyo_safe, LAG_CONFIGS

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
PAPER_REFS  = {"StemRust": 0.78, "StripeRust": 0.60, "LeafRust": 0.66}
PAPER_FEATURES = ["Latitude", "Longitude", "Altitude", "DaysIntoMainSeason"]

TEMP_OPTIMA = {
    "StemRust":   {"opt": 20.0, "sigma": 6.0},
    "StripeRust": {"opt": 12.0, "sigma": 4.0},
    "LeafRust":   {"opt": 18.0, "sigma": 5.0},
}

# v5.1 best XGB params (reuse for StemRust — already globally tuned)
STEM_XGB_PARAMS = {
    "n_estimators": 342, "max_depth": 3, "learning_rate": 0.03,
    "subsample": 0.955, "colsample_bytree": 0.985,
    "reg_alpha": 4.9, "reg_lambda": 3.5e-5,
    "min_child_weight": 17, "gamma": 2.4,
}

N_OPTUNA_TRIALS = 80  # for global tuning if needed

COLOUR_MAP = {
    "Paper (baseline)":           ("#4393C3", "--",  "x"),
    "v3: GB (prev best)":         ("#7B2D8B", ":",   "^"),
    "v5.1: XGB tuned":            ("#E8820C", "-.",  "s"),
    "v6: Best-of-breed":          ("#D6604D", "-",   "o"),
    "v6: TabNet":                 ("#1A7C41", "-",   "D"),
}


# ─────────────────────────────────────────────
# 1. FEATURE ENGINEERING
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
    """
    Compute smarter NDVI features that capture disease-relevant signals
    rather than raw greenness values.
    """
    ndvi_cols = []

    if "ndvi" not in df.columns:
        return df, ndvi_cols

    # 1. NDVI anomaly: observed - location/month climatological mean
    #    Bin lat/lon to ~0.5 degree cells for stable means
    df["_lat_bin"] = (df["Latitude"] * 2).round() / 2
    df["_lon_bin"] = (df["Longitude"] * 2).round() / 2
    clim_mean = df.groupby(["_lat_bin", "_lon_bin", "Month"])["ndvi"].transform("mean")
    df["ndvi_anomaly"] = df["ndvi"] - clim_mean
    ndvi_cols.append("ndvi_anomaly")

    # 2. NDVI rate of change (stress signal)
    if "ndvi_lag1m" in df.columns:
        df["ndvi_change"] = df["ndvi"] - df["ndvi_lag1m"]
        ndvi_cols.append("ndvi_change")

    # 3. Interaction features
    df["ndvi_x_alt"] = df["ndvi"] * df["Altitude"] / 1000
    ndvi_cols.append("ndvi_x_alt")

    if "GrowthStage" in df.columns:
        df["ndvi_x_growth"] = df["ndvi"] * df["GrowthStage"]
        ndvi_cols.append("ndvi_x_growth")

    # 4. Also keep raw ndvi for comparison
    ndvi_cols.extend(["ndvi", "ndvi_lag1m"])

    # Clean up temp columns
    df.drop(columns=["_lat_bin", "_lon_bin"], inplace=True)

    # Fill NaNs
    for col in ndvi_cols:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med if not pd.isna(med) else 0)

    return df, [c for c in ndvi_cols if c in df.columns]


def engineer_all_features(df, rust_type):
    df = df.copy()

    # Base features
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

    # ERA5 climate features
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

    # Smart NDVI features
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


# ─────────────────────────────────────────────
# 2. FEATURE SELECTION
# ─────────────────────────────────────────────

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
# 3. MODELS
# ─────────────────────────────────────────────

def make_paper_lr():
    return SKPipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(fit_intercept=False, max_iter=500, C=1e6, solver="lbfgs"))
    ])

def make_gb():
    return GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        min_samples_leaf=15, subsample=0.8, max_features="sqrt", random_state=42,
    )

def make_stem_xgb():
    return XGBClassifier(
        **STEM_XGB_PARAMS, random_state=42, eval_metric="logloss", verbosity=0,
    )


def run_tabnet_loyo(df_cv, feats, rust_name):
    """Run TabNet with LOYO-CV."""
    avail = [f for f in feats if f in df_cv.columns]
    yearly_aucs = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    TabNet device: {device}")

    for test_year in CV_YEARS:
        tr = df_cv["Year"] != test_year
        te = df_cv["Year"] == test_year
        y_tr = df_cv.loc[tr, TARGET].values
        y_te = df_cv.loc[te, TARGET].values

        if len(np.unique(y_te)) < 2:
            continue

        X_tr = df_cv.loc[tr, avail].fillna(0).values.astype(np.float32)
        X_te = df_cv.loc[te, avail].fillna(0).values.astype(np.float32)

        # Use latest training year as validation for early stopping
        train_years = sorted(df_cv.loc[tr, "Year"].unique())
        val_year = train_years[-1]
        inner_tr = tr & (df_cv["Year"] != val_year)
        inner_val = tr & (df_cv["Year"] == val_year)

        X_itr = df_cv.loc[inner_tr, avail].fillna(0).values.astype(np.float32)
        y_itr = df_cv.loc[inner_tr, TARGET].values
        X_ival = df_cv.loc[inner_val, avail].fillna(0).values.astype(np.float32)
        y_ival = df_cv.loc[inner_val, TARGET].values

        model = TabNetClassifier(
            n_d=32, n_a=32, n_steps=5,
            gamma=1.5, lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params=dict(step_size=10, gamma=0.9),
            mask_type="entmax",
            device_name=device,
            verbose=0,
        )

        model.fit(
            X_itr, y_itr,
            eval_set=[(X_ival, y_ival)],
            eval_metric=["auc"],
            max_epochs=100,
            patience=15,
            batch_size=256,
        )

        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
        yearly_aucs[test_year] = auc

    return yearly_aucs


def run_loyo_cv(df, feats, model_fn):
    """Run LOYO-CV with a given model factory and feature set."""
    import sklearn.base
    mask = (df["Year"].isin(CV_YEARS)) & (df["IsMeherSeason"] == 1)
    df_cv = df[mask].copy().reset_index(drop=True)
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
# 4. PLOTS
# ─────────────────────────────────────────────

def plot_yearly_auc(yearly_aucs_dict, rust_name, out_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhline(0.5, color="lightgray", ls=":", lw=1.2)
    ax.axhline(PAPER_REFS[rust_name], color="black", ls="--", lw=1.5,
               label=f"Paper: {PAPER_REFS[rust_name]:.2f}", alpha=0.6)
    for lbl, aucs in yearly_aucs_dict.items():
        vals = [aucs.get(y, np.nan) for y in CV_YEARS]
        col, ls, mk = COLOUR_MAP.get(lbl, ("gray", "-", "o"))
        ax.plot(CV_YEARS, vals, marker=mk, color=col, ls=ls, lw=2, markersize=8,
                label=f"{lbl}  ({np.nanmean(vals):.3f})", zorder=3)
    ax.set_xticks(CV_YEARS)
    ax.set_xticklabels([str(y) for y in CV_YEARS], rotation=45, ha="right")
    ax.set_ylim(0.35, 1.0)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title(f"{rust_name} — v6 Results", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary(all_results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    label_order = list(COLOUR_MAP.keys())
    for ax, rust_name in zip(axes, CLIMATE_FILES.keys()):
        lbls = [l for l in label_order if l in all_results[rust_name]]
        vals = [np.nanmean(list(all_results[rust_name][l].values())) for l in lbls]
        cols = [COLOUR_MAP[l][0] for l in lbls]
        bars = ax.bar(range(len(lbls)), vals, color=cols, edgecolor="white", width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.006,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ref = PAPER_REFS[rust_name]
        ax.axhline(ref, color="black", ls="--", lw=1.8, label=f"Paper: {ref:.2f}")
        ax.axhline(0.5, color="lightgray", ls=":", lw=1)
        short = []
        for l in lbls:
            if "Paper" in l: short.append("Paper")
            elif "v3" in l: short.append("v3 GB")
            elif "v5.1" in l: short.append("v5.1\nXGB")
            elif "Best" in l: short.append("v6\nBest")
            elif "Tab" in l: short.append("v6\nTabNet")
            else: short.append(l[:8])
        ax.set_xticks(range(len(lbls)))
        ax.set_xticklabels(short, fontsize=9)
        ax.set_ylim(0.45, 1.0)
        ax.set_title(rust_name, fontsize=13, fontweight="bold")
        ax.set_ylabel("Mean AUC-ROC", fontsize=10)
        ax.legend(fontsize=8.5)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("v6: Smarter NDVI + Per-Rust Selection + TabNet",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# 5. MAIN
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
    log_path = os.path.join(OUT_DIR, "v6_run.log")
    sys.stdout = TeeOutput(log_path)
    sys.stderr = sys.stdout

    all_results = {}
    summary_rows = []

    print("=" * 65)
    print("  Wheat Rust ML Pipeline v6")
    print("  Smarter NDVI + Per-Rust Selection + TabNet")
    print("=" * 65)
    print(f"  Data: {DATA_DIR}")
    print(f"  Output: {OUT_DIR}")
    print(f"  CUDA: {torch.cuda.is_available()}")

    for rust_name, climate_fname in CLIMATE_FILES.items():
        print(f"\n{'='*65}")
        print(f"  {rust_name}")
        print(f"{'='*65}")

        df = pd.read_csv(os.path.join(DATA_DIR, climate_fname))
        print(f"  Loaded {len(df)} surveys")

        # Feature engineering
        df, climate_cols, ndvi_cols = engineer_all_features(df, rust_name)
        race_names = [f for f in get_race_feature_names(rust_name) if f in df.columns]

        # Check NDVI smart features
        print(f"  NDVI features: {ndvi_cols}")
        for nc in ndvi_cols:
            if nc in df.columns:
                print(f"    {nc}: mean={df[nc].mean():.4f}, std={df[nc].std():.4f}, "
                      f"coverage={df[nc].notna().mean():.1%}")

        # Lag features
        print(f"  Computing lag features...")
        meher_idx = df.index[df["IsMeherSeason"] == 1]
        df_m, lag_names = compute_lag_features_loyo_safe(
            df.loc[meher_idx].copy(), target_col=TARGET,
            configs=LAG_CONFIGS, cv_years=CV_YEARS, verbose=True)
        for col in lag_names:
            df.loc[meher_idx, col] = df_m[col].values
            df[col] = df[col].fillna(df[TARGET].mean() if "_prev" in col else 0)

        # Feature sets
        v1_feats = [f for f in V1_FEATURES if f in df.columns]
        lag_feats = [f for f in lag_names if f in df.columns]
        v3_feats = v1_feats + lag_feats + race_names
        full_candidate = v3_feats + climate_cols + ndvi_cols
        full_candidate = list(dict.fromkeys(full_candidate))

        # Feature selection on Meher CV data
        meher_mask = (df["Year"].isin(CV_YEARS)) & (df["IsMeherSeason"] == 1)
        print(f"\n  Feature selection...")
        selected_feats, mi_df = select_features_mi_corr(
            df[meher_mask], full_candidate, TARGET, corr_threshold=0.85)
        print(f"    {len(full_candidate)} candidates → {len(selected_feats)} selected")

        # Show where NDVI smart features rank
        print(f"\n  NDVI MI ranking:")
        for nc in ndvi_cols:
            row = mi_df[mi_df["feature"] == nc]
            if not row.empty:
                rank = mi_df.index.get_loc(row.index[0]) + 1
                mi_val = row["mi"].values[0]
                selected = "SELECTED" if nc in selected_feats else "dropped"
                print(f"    {nc}: MI={mi_val:.4f}, rank={rank}/{len(mi_df)}, {selected}")

        mi_df.to_csv(os.path.join(OUT_DIR, f"{rust_name}_v6_mi_scores.csv"), index=False)

        # ── Per-rust best-of-breed model selection ──
        print(f"\n  Running LOYO-CV...")
        df_cv = df[meher_mask].copy().reset_index(drop=True)

        # Baseline: Paper LR
        aucs_paper = run_loyo_cv(df, PAPER_FEATURES, make_paper_lr)
        print(f"    Paper baseline: {np.nanmean(list(aucs_paper.values())):.4f}")

        # v3: GB with v3 features (no ERA5)
        aucs_v3 = run_loyo_cv(df, v3_feats, make_gb)
        print(f"    v3 GB (no ERA5): {np.nanmean(list(aucs_v3.values())):.4f}")

        # v5.1 equivalent: XGB with selected features
        aucs_v51 = run_loyo_cv(df, selected_feats, make_stem_xgb)
        print(f"    v5.1 XGB tuned: {np.nanmean(list(aucs_v51.values())):.4f}")

        # v6 Best-of-breed: pick the right model+features per rust type
        if rust_name == "StemRust":
            # XGB with full selected features (ERA5 helps stem rust)
            aucs_best = run_loyo_cv(df, selected_feats, make_stem_xgb)
        else:
            # GB with v3 features (no ERA5 — race/spatial features dominate)
            aucs_best = run_loyo_cv(df, v3_feats, make_gb)
        print(f"    v6 best-of-breed: {np.nanmean(list(aucs_best.values())):.4f}")

        # v6 TabNet
        print(f"    Running TabNet...")
        aucs_tabnet = run_tabnet_loyo(df_cv, selected_feats, rust_name)
        print(f"    v6 TabNet: {np.nanmean(list(aucs_tabnet.values())):.4f}")

        # Collect results
        yearly_aucs = {
            "Paper (baseline)": aucs_paper,
            "v3: GB (prev best)": aucs_v3,
            "v5.1: XGB tuned": aucs_v51,
            "v6: Best-of-breed": aucs_best,
            "v6: TabNet": aucs_tabnet,
        }

        # Print table
        print(f"\n  {'Year':<8}", end="")
        for lbl in yearly_aucs:
            print(f"  {lbl[:22]:<22}", end="")
        print()
        for yr in CV_YEARS:
            print(f"  {yr:<8}", end="")
            for lbl in yearly_aucs:
                print(f"  {yearly_aucs[lbl].get(yr, np.nan):<22.3f}", end="")
            print()
        print(f"  {'MEAN':<8}", end="")
        for lbl in yearly_aucs:
            mean = np.nanmean(list(yearly_aucs[lbl].values()))
            print(f"  {mean:<22.4f}", end="")
        print()

        # Plots
        rust_dir = os.path.join(OUT_DIR, rust_name)
        os.makedirs(rust_dir, exist_ok=True)
        plot_yearly_auc(yearly_aucs, rust_name,
                        os.path.join(rust_dir, f"{rust_name}_v6_yearly_auc.png"))

        all_results[rust_name] = yearly_aucs
        for lbl in yearly_aucs:
            mean = np.nanmean(list(yearly_aucs[lbl].values()))
            summary_rows.append({
                "Rust": rust_name, "Model": lbl, "Mean_AUC": round(mean, 4),
                **{f"AUC_{y}": round(yearly_aucs[lbl].get(y, np.nan), 4) for y in CV_YEARS}
            })

    # Summary
    plot_summary(all_results, os.path.join(OUT_DIR, "v6_summary_auc.png"))
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUT_DIR, "v6_results_summary.csv"), index=False)

    # Final table
    print("\n\n" + "=" * 65)
    print("  FINAL RESULTS — v6")
    print("=" * 65)
    for rust_name in CLIMATE_FILES:
        ref = PAPER_REFS[rust_name]
        print(f"\n  {rust_name}  (paper = {ref:.3f}):")
        best_lbl, best_auc = None, -1
        for lbl in all_results[rust_name]:
            mean = np.nanmean(list(all_results[rust_name][lbl].values()))
            delta = mean - ref
            sign = "+" if delta >= 0 else ""
            if mean > best_auc: best_auc, best_lbl = mean, lbl
            print(f"    {lbl:<30}: AUC = {mean:.4f}  ({sign}{delta:.4f})")
        print(f"    >>> BEST: {best_lbl} = {best_auc:.4f}")

    print(f"\n  Outputs: {OUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()

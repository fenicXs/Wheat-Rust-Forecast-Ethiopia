"""
Spatial-temporal lag features for wheat rust prediction.
========================================================
Reconstructed from the v2/v3 pipeline logic.

Computes disease prevalence in nearby surveys from previous time periods,
with LOYO-safe computation to prevent train-test leakage.

Each LAG_CONFIG entry: (radius_km, time_lag_months, aggregation, label)
  - radius_km: spatial radius to search for nearby surveys
  - time_lag_months: how many months back to look
  - aggregation: 'mean' (disease prevalence) — count is always added
  - label: feature name prefix

For each config, two features are produced:
  - {label}_prev  : mean disease prevalence of nearby surveys
  - {label}_count : number of nearby surveys found (confidence signal)
"""

import numpy as np
import pandas as pd

# Spatial-temporal lag configurations
LAG_CONFIGS = [
    (25,  1, "mean", "lag_25km_1m"),    # 25km, 1 month back
    (50,  1, "mean", "lag_50km_1m"),    # 50km, 1 month back
    (100, 1, "mean", "lag_100km_1m"),   # 100km, 1 month back
    (50,  2, "mean", "lag_50km_2m"),    # 50km, 2 months back
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in kilometres."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_lag_features_loyo_safe(df, target_col, configs, cv_years, verbose=True):
    """
    Compute spatial-temporal lag features, safe for LOYO cross-validation.

    For each survey point, find nearby surveys (within radius_km) from
    time_lag_months ago and compute the mean disease prevalence.

    LOYO safety: When computing features for year Y (a CV test year),
    only use data from years != Y. This prevents temporal leakage.
    For non-CV years or training data, all available data is used.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Latitude, Longitude, Year, Month, and target_col.
    target_col : str
        Binary disease presence column name.
    configs : list of tuples
        Each tuple: (radius_km, time_lag_months, aggregation, label)
    cv_years : list of int
        Years used in cross-validation (features for these years are
        computed excluding same-year data).
    verbose : bool

    Returns
    -------
    df : pd.DataFrame with new columns added
    lag_names : list of str, new column names
    """
    df = df.copy().reset_index(drop=True)
    lats = df["Latitude"].values
    lons = df["Longitude"].values
    years = df["Year"].values
    months = df["Month"].values
    target = df[target_col].values

    lag_names = []

    for radius_km, time_lag, agg, label in configs:
        prev_col = f"{label}_prev"
        count_col = f"{label}_count"
        lag_names.extend([prev_col, count_col])

        prev_vals = np.full(len(df), np.nan)
        count_vals = np.zeros(len(df), dtype=int)

        for i in range(len(df)):
            tgt_month = months[i] - time_lag
            tgt_year = years[i]
            if tgt_month <= 0:
                tgt_month += 12
                tgt_year -= 1

            # Find surveys in the target month/year
            month_mask = (months == tgt_month)

            # LOYO safety: if this survey's year is a CV year,
            # exclude all data from the same year
            if years[i] in cv_years:
                year_mask = (years == tgt_year) & (years != years[i])
            else:
                year_mask = (years == tgt_year)

            candidates = np.where(month_mask & year_mask)[0]

            if len(candidates) == 0:
                # Fall back: try any year for the target month (excluding own year for CV)
                if years[i] in cv_years:
                    fallback_mask = month_mask & (years != years[i])
                else:
                    fallback_mask = month_mask & (years < years[i])
                candidates = np.where(fallback_mask)[0]

            if len(candidates) == 0:
                continue

            # Compute distances to all candidates
            dists = haversine_km(lats[i], lons[i], lats[candidates], lons[candidates])
            nearby = candidates[dists <= radius_km]

            if len(nearby) > 0:
                prev_vals[i] = np.mean(target[nearby])
                count_vals[i] = len(nearby)

        df[prev_col] = prev_vals
        df[count_col] = count_vals

        if verbose:
            coverage = np.sum(~np.isnan(prev_vals)) / len(df)
            print(f"    {label}: coverage={coverage:.1%}, "
                  f"mean_count={np.mean(count_vals[count_vals > 0]):.1f}")

    return df, lag_names

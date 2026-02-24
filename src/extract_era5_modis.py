"""
CORRECTED ERA5 + MODIS NDVI Extraction Script
==============================================
Run this on YOUR LOCAL MACHINE.

FIX: The original script used simple lat/lon bounding boxes to convert
geographic coordinates to MODIS pixel positions. This was WRONG.

MODIS uses the Sinusoidal projection (also called "SIN" projection):
  - The globe is mapped onto a cylinder, then unrolled
  - Each tile is NOT a simple lat/lon rectangle
  - Pixel (row, col) within a tile must be computed via the sinusoidal
    projection formula, NOT by dividing lat/lon ranges linearly

Without this fix, every lat/lon → pixel lookup falls outside valid bounds
or hits the wrong pixel, returning NaN (fill value) for all surveys.

MODIS sinusoidal projection parameters (from LP DAAC documentation):
  Sphere radius R = 6,371,007.181 m
  Total tile columns: 36 (h00 to h35)
  Total tile rows:    18 (v00 to v17)
  Pixels per tile: 1200 × 1200 (for MOD13A3 1km product)
  Tile width  = 2πR / 36
  Tile height = πR  / 18

Conversion formula:
  lat_rad = lat × π/180
  lon_rad = lon × π/180
  x = R × cos(lat_rad) × lon_rad          # horizontal distance
  y = R × lat_rad                          # vertical distance
  global_h = (x + πR)  / (2πR/36)         # float tile column
  global_v = (πR/2 - y) / (πR/18)         # float tile row
  tile_h = int(global_h)                   # which h-tile (0-35)
  tile_v = int(global_v)                   # which v-tile (0-17)
  pixel_col = int((global_h - tile_h) × 1200)
  pixel_row = int((global_v - tile_v) × 1200)

Ethiopia's tiles in the MODIS grid:
  h21v07: central/northern Ethiopia
  h21v08: southern Ethiopia
  h22v07: eastern Ethiopia  
  h22v08: southeastern Ethiopia

Install requirements:
    pip install netCDF4 numpy pandas pyhdf

Usage:
    python extract_era5_modis_corrected.py

Edit the CONFIG section paths before running.
"""

import os
import math
import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — edit these paths to match your local data locations
# ─────────────────────────────────────────────

SURVEY_DIR = r"path/to/wheat_rust_outbreaks_ethiopia/surveyData_cleaned"
ERA5_FILE  = r"path/to/era5_ethiopia_monthly_2007_2019.nc"
MODIS_DIR  = r"path/to/MODIS_NDVI"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

SURVEY_FILES = {
    "StemRust":   "CleanSurveyDataWithAdditionals_ETWheatStemRust.csv",
    "StripeRust": "CleanSurveyDataWithAdditionals_ETWheatYellowRust.csv",
    "LeafRust":   "CleanSurveyDataWithAdditionals_ETWheatLeafRust.csv",
}

# ─────────────────────────────────────────────
# 1. MODIS SINUSOIDAL PROJECTION
# ─────────────────────────────────────────────

MODIS_R            = 6_371_007.181      # sphere radius (m)
MODIS_TILES_H      = 36                 # total tile columns
MODIS_TILES_V      = 18                 # total tile rows
MODIS_PIXELS       = 1200              # pixels per tile side (MOD13A3)
MODIS_TILE_WIDTH   = 2 * math.pi * MODIS_R / MODIS_TILES_H   # metres
MODIS_TILE_HEIGHT  = math.pi * MODIS_R / MODIS_TILES_V        # metres


def latlon_to_modis_tile_pixel(lat_deg, lon_deg):
    """
    Convert geographic latitude/longitude (WGS84 degrees) to
    MODIS tile identifier and pixel (row, col) within that tile.

    Returns:
        tile_str  : e.g. 'h21v07'
        pixel_row : row within tile (0–1199, top=0)
        pixel_col : col within tile (0–1199, left=0)
        or (None, None, None) if point is outside the MODIS grid
    """
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    # Sinusoidal projection: x and y in metres
    x = MODIS_R * math.cos(lat_rad) * lon_rad
    y = MODIS_R * lat_rad

    # Global fractional tile position
    # x ranges from -πR to +πR  → horizontal
    # y ranges from -πR/2 to +πR/2 → vertical (N positive)
    global_h = (x + math.pi * MODIS_R) / MODIS_TILE_WIDTH
    global_v = (math.pi * MODIS_R / 2 - y) / MODIS_TILE_HEIGHT

    # Integer tile
    tile_h = int(global_h)
    tile_v = int(global_v)

    # Validate bounds
    if not (0 <= tile_h < MODIS_TILES_H and 0 <= tile_v < MODIS_TILES_V):
        return None, None, None

    # Pixel within tile (0-indexed, row from top)
    frac_h = global_h - tile_h   # 0.0–1.0 within tile horizontally
    frac_v = global_v - tile_v   # 0.0–1.0 within tile vertically

    pixel_col = int(frac_h * MODIS_PIXELS)
    pixel_row = int(frac_v * MODIS_PIXELS)

    # Clamp to valid range
    pixel_col = max(0, min(pixel_col, MODIS_PIXELS - 1))
    pixel_row = max(0, min(pixel_row, MODIS_PIXELS - 1))

    tile_str = f"h{tile_h:02d}v{tile_v:02d}"
    return tile_str, pixel_row, pixel_col


def verify_projection():
    """Quick sanity check — print tile/pixel for known Ethiopian cities."""
    test_points = [
        ("Addis Ababa",  9.03,  38.74),
        ("Mekelle",     13.50,  39.47),
        ("Hawassa",      7.06,  38.47),
        ("Dire Dawa",    9.59,  41.87),
        ("Jimma",        7.67,  36.83),
    ]
    print("  Sinusoidal projection verification:")
    for name, lat, lon in test_points:
        tile, row, col = latlon_to_modis_tile_pixel(lat, lon)
        print(f"    {name:15s} ({lat:.2f}°N, {lon:.2f}°E) → tile={tile}, row={row}, col={col}")


# ─────────────────────────────────────────────
# 2. MODIS FILE INDEX
# ─────────────────────────────────────────────

# MOD13A3 uses composite periods starting on these DOYs each year
DOY_TO_MONTH = {
    1: 1, 32: 2, 60: 3, 91: 4, 121: 5, 152: 6,
    182: 7, 213: 8, 244: 9, 274: 10, 305: 11, 335: 12
}
MONTH_TO_DOY = {v: k for k, v in DOY_TO_MONTH.items()}


def build_modis_index(modis_dir):
    """
    Scan MODIS directory and index all HDF files.
    Returns dict: {(tile_str, year, month): filepath}
    """
    pattern = re.compile(r'MOD13A3\.A(\d{4})(\d{3})\.(h\d{2}v\d{2})\.')
    index   = {}
    n_found = 0

    for fname in os.listdir(modis_dir):
        m = pattern.match(fname)
        if m:
            year = int(m.group(1))
            doy  = int(m.group(2))
            tile = m.group(3)
            month = DOY_TO_MONTH.get(doy)
            if month is not None:
                index[(tile, year, month)] = os.path.join(modis_dir, fname)
                n_found += 1

    print(f"  Indexed {n_found} MODIS files across "
          f"{len(set(t for t,_,_ in index))} tiles, "
          f"years {min(y for _,y,_ in index)}–{max(y for _,y,_ in index)}")
    return index


def read_modis_ndvi(hdf_path, pixel_row, pixel_col):
    """
    Read NDVI value from a MOD13A3 HDF file at the given pixel position.
    Returns float NDVI in [-0.2, 1.0] range, or np.nan if missing/invalid.
    """
    try:
        from pyhdf.SD import SD, SDC
        hdf      = SD(hdf_path, SDC.READ)
        datasets = hdf.datasets()

        # Find the NDVI dataset (name varies slightly between collection versions)
        ndvi_name = None
        for name in datasets:
            if 'NDVI' in name.upper():
                ndvi_name = name
                break

        if ndvi_name is None:
            hdf.end()
            return np.nan

        sds    = hdf.select(ndvi_name)
        attrs  = sds.attributes()

        scale_factor = attrs.get('scale_factor', 10000.0)
        # MODIS NDVI is typically scaled by 1/10000; some files store 10000, others 0.0001
        if scale_factor and scale_factor > 1:
            scale_factor = 1.0 / scale_factor
        fill_value   = attrs.get('_FillValue', -3000)
        valid_range  = attrs.get('valid_range', [-2000, 10000])

        # Read just the needed pixel instead of the full 1200x1200 grid
        raw = int(sds.get(start=(pixel_row, pixel_col), count=(1, 1))[0][0])
        hdf.end()

        if raw == fill_value:
            return np.nan
        if not (valid_range[0] <= raw <= valid_range[1]):
            return np.nan

        return float(raw) * scale_factor  # NDVI ∈ [-0.2, 1.0]

    except Exception as e:
        return np.nan


def extract_ndvi_for_survey(lat, lon, year, month, modis_index):
    """
    Extract NDVI at a survey location for the current and preceding month.
    Uses correct sinusoidal projection for pixel lookup.
    """
    result = {}

    # Pre-compute tile and pixel for this location (same for all time steps)
    tile, pixel_row, pixel_col = latlon_to_modis_tile_pixel(lat, lon)

    for lag in [0, 1]:
        lag_str     = f"_lag{lag}m" if lag > 0 else ""
        tgt_month   = month - lag
        tgt_year    = year
        if tgt_month <= 0:
            tgt_month += 12
            tgt_year  -= 1

        key = (tile, tgt_year, tgt_month) if tile else None

        if key and key in modis_index:
            ndvi_val = read_modis_ndvi(modis_index[key], pixel_row, pixel_col)
        else:
            ndvi_val = np.nan

        result[f"ndvi{lag_str}"] = ndvi_val

    return result


# ─────────────────────────────────────────────
# 3. ERA5 EXTRACTION (same as before, working)
# ─────────────────────────────────────────────

def load_era5(era5_path):
    import netCDF4 as nc
    ds = nc.Dataset(era5_path)
    lats = np.array(ds.variables['latitude'][:])
    lons = np.array(ds.variables['longitude'][:])

    # Support both 'time' and 'valid_time' dimension names
    time_key = 'valid_time' if 'valid_time' in ds.variables else 'time'
    time_var  = ds.variables[time_key]
    time_vals = nc.num2date(time_var[:], time_var.units,
                            only_use_cftime_datetimes=False)

    skip = {'latitude','longitude','time','valid_time','expver','number'}
    available_vars = [v for v in ds.variables if v not in skip]
    print(f"  ERA5 variables: {available_vars}")
    print(f"  ERA5 time: {time_vals[0]} → {time_vals[-1]}  (n={len(time_vals)})")
    print(f"  ERA5 grid: lat {lats.min():.1f}–{lats.max():.1f}, "
          f"lon {lons.min():.1f}–{lons.max():.1f}")

    data = {}
    for var in available_vars:
        arr = ds.variables[var][:]
        if hasattr(arr, 'filled'):
            arr = arr.filled(np.nan)
        data[var] = np.array(arr)
        print(f"    {var}: shape={arr.shape}, "
              f"range=[{float(np.nanmin(arr)):.4g}, {float(np.nanmax(arr)):.4g}]")

    ds.close()
    return lats, lons, time_vals, data, available_vars


def extract_era5_for_survey(lat, lon, year, month,
                             lats, lons, time_vals, data, var_names):
    lat_idx = int(np.argmin(np.abs(lats - lat)))
    lon_idx = int(np.argmin(np.abs(lons - lon)))
    result  = {}

    for lag in [0, 1, 2]:
        tgt_month = month - lag
        tgt_year  = year
        if tgt_month <= 0:
            tgt_month += 12
            tgt_year  -= 1

        time_idx = next(
            (i for i, t in enumerate(time_vals)
             if getattr(t, 'year', None) == tgt_year
             and getattr(t, 'month', None) == tgt_month),
            None
        )
        lag_str = f"_lag{lag}m" if lag > 0 else ""

        for var in var_names:
            arr = data[var]
            if time_idx is not None and arr.ndim == 3:
                val = float(arr[time_idx, lat_idx, lon_idx])
            else:
                val = np.nan
            result[f"era5_{var}{lag_str}"] = val

    # Derived: relative humidity via Magnus formula
    for lag in [0, 1]:
        lag_str = f"_lag{lag}m" if lag > 0 else ""
        t2m = result.get(f"era5_t2m{lag_str}", np.nan)
        d2m = result.get(f"era5_d2m{lag_str}", np.nan)
        if not (np.isnan(t2m) or np.isnan(d2m)):
            def sat_vp(T_K):
                T_C = T_K - 273.15
                return 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))
            rh = 100.0 * sat_vp(d2m) / sat_vp(t2m)
            result[f"era5_rh{lag_str}"] = float(np.clip(rh, 0, 105))
        else:
            result[f"era5_rh{lag_str}"] = np.nan

    return result


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  MODIS Projection Sanity Check")
    print("=" * 60)
    verify_projection()

    print("\n" + "=" * 60)
    print("  Loading ERA5...")
    print("=" * 60)
    lats_e5, lons_e5, times_e5, era5_data, era5_vars = load_era5(ERA5_FILE)

    print("\n" + "=" * 60)
    print("  Indexing MODIS files...")
    print("=" * 60)
    modis_index = build_modis_index(MODIS_DIR)

    # Quick diagnostic: test 3 locations to verify NDVI reads work
    print("\n  NDVI read test (3 Ethiopian locations):")
    test_pts = [("Addis Ababa", 9.03, 38.74), ("Mekelle", 13.5, 39.47), ("Jimma", 7.67, 36.83)]
    for name, lat, lon in test_pts:
        tile, row, col = latlon_to_modis_tile_pixel(lat, lon)
        key = (tile, 2015, 9)   # September 2015 as test month
        if key in modis_index:
            val = read_modis_ndvi(modis_index[key], row, col)
            print(f"    {name}: tile={tile}, row={row}, col={col}, NDVI={val:.4f}" if not np.isnan(val)
                  else f"    {name}: tile={tile}, row={row}, col={col}, NDVI=NaN (check file)")
        else:
            print(f"    {name}: tile={tile} — key {key} not in index")

    # Main extraction loop
    for rust_name, survey_fname in SURVEY_FILES.items():
        print(f"\n{'='*60}")
        print(f"  Processing {rust_name}...")
        print(f"{'='*60}")

        survey_path = os.path.join(SURVEY_DIR, survey_fname)
        df = pd.read_csv(survey_path)
        n  = len(df)
        print(f"  {n} surveys to process")

        results = []
        n_ndvi_valid = 0

        for i, row in df.iterrows():
            if i % 1000 == 0:
                print(f"  [{i+1}/{n}] NDVI valid so far: {n_ndvi_valid}")

            lat   = float(row['Latitude'])
            lon   = float(row['Longitude'])
            year  = int(row['Year'])
            month = int(row['Month'])

            era5_res = extract_era5_for_survey(
                lat, lon, year, month,
                lats_e5, lons_e5, times_e5, era5_data, era5_vars
            )
            ndvi_res = extract_ndvi_for_survey(lat, lon, year, month, modis_index)

            if not np.isnan(ndvi_res.get("ndvi", np.nan)):
                n_ndvi_valid += 1

            results.append({**era5_res, **ndvi_res})

        results_df = pd.DataFrame(results)
        df_out     = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        out_path = os.path.join(OUTPUT_DIR, f"{rust_name}_with_climate_v2.csv")
        df_out.to_csv(out_path, index=False)

        print(f"\n  Saved: {out_path}")
        print(f"  Shape: {df_out.shape}")

        # Coverage report
        climate_cols = [c for c in results_df.columns]
        era5_cols    = [c for c in climate_cols if c.startswith("era5_")]
        ndvi_cols    = [c for c in climate_cols if c.startswith("ndvi")]
        print(f"\n  Coverage:")
        print(f"    ERA5 (mean): {df_out[era5_cols].notna().mean().mean():.1%}")
        for col in ndvi_cols:
            cov = df_out[col].notna().mean()
            rng = (f"{df_out[col].min():.3f}–{df_out[col].max():.3f}"
                   if cov > 0 else "all NaN — check MODIS dir path")
            print(f"    {col}: {cov:.1%} coverage, range {rng}")

    print(f"\n\nDone. Upload the 3 files from:\n  {OUTPUT_DIR}")
    for rust_name in SURVEY_FILES:
        print(f"    {rust_name}_with_climate_v2.csv")


if __name__ == "__main__":
    main()

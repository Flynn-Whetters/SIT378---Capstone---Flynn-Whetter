# ===========================================
# AFL Heatmap Pipeline (Reusable & Console-ready)
# ===========================================
# Description:
#   A pipeline to generate AFL field heatmaps from tracking/event CSV datasets.
#   It supports two common CSV schemas (Bounding Box style and Tracking style)
#   and exports heatmaps that show player activity and density.
#
# Features:
#   - Handles multiple datasets in one run
#   - Supports both bounding-box and tracking schemas
#   - Creates overall, per-player, and zone heatmaps
#   - Console interface for team use
#
# Usage (from console):
#   python afl_heatmap.py tracking.csv:kick mark.csv tackles.csv
#
#   Options:
#     path:label   -> path to CSV file with optional label name
#     --out-dir    -> set custom output folder (default: outputs)
#     --sigma      -> set blur strength (default: 2.0)
#
# Example:
#   python afl_heatmap.py tracking_csv.csv:tracking (Use this)
#
# Outputs:
#   - outputs/<label>/overall/overall.png
#   - outputs/<label>/per_id/id_<ID>.png 
#   - outputs/<label>/zones/back_50.png, midfield.png, forward_50.png
#
# Requirements:
#   pip install numpy pandas matplotlib scipy
# ===========================================

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

# -------------------------
# Field + Global Config
# -------------------------
FIELD_LENGTH_M = 159.5     # AFL ground length in metres
FIELD_WIDTH_M  = 128.8     # AFL ground width in metres
a = FIELD_LENGTH_M / 2.0   # semi-major axis
b = FIELD_WIDTH_M  / 2.0   # semi-minor axis

NX, NY = 200, 150          # heatmap grid resolution
SIGMA  = 2.0               # default blur level

OUT_ROOT = "outputs"       # output folder name
WEIGHT_COLS = ["confidence", "conf"]  # columns used for weighting

# Standard unified columns
UNIFIED_COLS = [
    "frame_id","player_id","timestamp_s",
    "x1","y1","x2","y2","cx","cy","w","h",
    "confidence","class_id","visibility"
]

# -------------------------
# CSV schema definitions
# -------------------------
# Schema A: Bounding box-style (x1,y1,x2,y2,cx,cy,w,h)
BOUNDINGBOX_REQUIRED = [
    "frame_id","player_id","timestamp_s",
    "x1","y1","x2","y2","cx","cy","w","h","confidence"
]

# Schema B: Tracking-style (x,y,width,height,track_id)
TRACKING_REQUIRED = [
    "frame_id","track_id","x","y",
    "width","height","conf","class_id","visibility"
]

# -------------------------
# Helper Functions
# -------------------------
def raw_bbox(xs, ys, pad_ratio=0.02):
    """Get bounding box for raw coordinates with padding."""
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    dx, dy = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    return (xmin - dx*pad_ratio, xmax + dx*pad_ratio,
            ymin - dy*pad_ratio, ymax + dy*pad_ratio)

def raw_to_metres(x, y, bbox_raw, a, b):
    """Convert raw coords to metre coords scaled to AFL oval size."""
    xmin, xmax, ymin, ymax = bbox_raw
    x_m = ((x - xmin) / max(1e-9, (xmax - xmin))) * (2*a) - a
    y_m = ((y - ymin) / max(1e-9, (ymax - ymin))) * (2*b) - b
    return x_m, y_m

def make_oval_mask_metres(nx, ny, a, b):
    """Create oval-shaped mask for field area."""
    x_edges = np.linspace(-a, a, nx + 1)
    y_edges = np.linspace(-b, b, ny + 1)
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2
    Xc, Yc = np.meshgrid(xc, yc, indexing="xy")
    mask = (Xc**2)/(a**2) + (Yc**2)/(b**2) <= 1.0
    return x_edges, y_edges, mask

def heatmap_from_metres(x_m, y_m, a, b, nx=NX, ny=NY, sigma=SIGMA, weights=None):
    """Make heatmap grid from metre coords with Gaussian blur applied."""
    x_edges, y_edges, mask = make_oval_mask_metres(nx, ny, a, b)
    H, _, _ = np.histogram2d(x_m, y_m, bins=[x_edges, y_edges], weights=weights)
    H = H.T
    if sigma and sigma > 0:
        H = gaussian_filter(H, sigma=sigma)
    H = np.where(mask, H, np.nan)
    return H, x_edges, y_edges

def choose_weights_unified(df):
    """Pick weight column (confidence/conf) if available."""
    for c in WEIGHT_COLS:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return None

# -------------------------
# CSV Loader
# -------------------------
def _coerce_numeric(df, cols):
    """Force selected columns into numeric type."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _to_unified_columns(df):
    """Convert dataframe to unified schema with cx,cy and player_id."""
    out = df.copy()
    if "player_id" not in out.columns:
        if "track_id" in out.columns:
            out["player_id"] = out["track_id"]
        else:
            out["player_id"] = np.nan
    if "cx" not in out.columns and all(c in out.columns for c in ["x","y","width","height"]):
        out["cx"] = pd.to_numeric(out["x"], errors="coerce") + pd.to_numeric(out["width"], errors="coerce")/2.0
        out["cy"] = pd.to_numeric(out["y"], errors="coerce") + pd.to_numeric(out["height"], errors="coerce")/2.0
    if "confidence" not in out.columns and "conf" in out.columns:
        out["confidence"] = out["conf"]
    for c in UNIFIED_COLS:
        if c not in out.columns:
            out[c] = np.nan
    out = _coerce_numeric(out, UNIFIED_COLS)
    out = out.dropna(subset=["cx","cy"])
    out = out[np.isfinite(out["cx"]) & np.isfinite(out["cy"])]
    return out.reset_index(drop=True)

def load_events_csv_any(path: str) -> pd.DataFrame:
    """Load CSV with auto-detected schema (Bounding Box or Tracking)."""
    df = pd.read_csv(path, sep=None, engine="python")  # auto-detect delimiter
    return _to_unified_columns(df)

# -------------------------
# Heatmap Plotting
# -------------------------
def plot_heatmap(H, x_edges, y_edges, a, b, out_path, title="Heatmap", alpha_img=0.9):
    """Plot one heatmap over AFL oval background and save PNG."""
    fig, ax = plt.subplots(figsize=(11, 8))
    t = np.linspace(0, 2*np.pi, 600)
    ax.fill(a*np.cos(t), b*np.sin(t), color=(0.05, 0.35, 0.05), alpha=1.0, zorder=0)
    ax.set_xlim([-a, a]); ax.set_ylim([-b, b])
    ax.set_aspect("equal"); ax.set_axis_off()

    finite_vals = H[np.isfinite(H)]
    vmin, vmax = 0.0, (np.nanpercentile(finite_vals, 99) if finite_vals.size else 1.0)
    extent = [x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()]
    im = ax.imshow(H, origin="lower", extent=extent, aspect="equal",
                   interpolation="bilinear", cmap="viridis",
                   norm=Normalize(vmin=vmin, vmax=vmax, clip=True),
                   alpha=alpha_img, zorder=2)

    ax.set_title(title, color="white")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intensity")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def split_zones_circular(df_m):
    """Split data into Back 50, Midfield, and Forward 50 zones."""
    dist_left  = np.sqrt((df_m["x_m"] + a)**2 + (df_m["y_m"])**2)
    dist_right = np.sqrt((df_m["x_m"] - a)**2 + (df_m["y_m"])**2)
    back50     = df_m[dist_left  <= 50].copy()
    fwd50      = df_m[dist_right <= 50].copy()
    mid        = df_m[(dist_left > 50) & (dist_right > 50)].copy()
    return {"Back 50": back50, "Midfield": mid, "Forward 50": fwd50}

# -------------------------
# Dataset Processing
# -------------------------
def process_dataset(path, label):
    """Run heatmap generation for one dataset."""
    print(f"\n=== Processing: {label} ===")
    out_base = os.path.join(OUT_ROOT, label)

    df = load_events_csv_any(path)
    if df.empty:
        print(f"({label}) Empty after load; skipping.")
        return

    bbox = raw_bbox(df["cx"].values, df["cy"].values)
    df["x_m"], df["y_m"] = raw_to_metres(df["cx"].values, df["cy"].values, bbox, a, b)
    weights = choose_weights_unified(df)

    # Overall heatmap
    H, xe, ye = heatmap_from_metres(df["x_m"], df["y_m"], a, b, NX, NY, SIGMA, weights)
    plot_heatmap(H, xe, ye, a, b, os.path.join(out_base, "overall", "overall.png"))

    # Per-player heatmaps
    for pid, sub in df.groupby("player_id"):
        if pd.isna(pid) or sub.empty: continue
        w = choose_weights_unified(sub)
        H_i, xe_i, ye_i = heatmap_from_metres(sub["x_m"], sub["y_m"], a, b, NX, NY, SIGMA, w)
        out_path = os.path.join(out_base, "per_id", f"id_{pid}.png")
        plot_heatmap(H_i, xe_i, ye_i, a, b, out_path, title=f"Player {pid}")

    # Zone heatmaps
    zones = split_zones_circular(df)
    for zname, zdf in zones.items():
        if zdf.empty: continue
        w = choose_weights_unified(zdf)
        H_z, xe_z, ye_z = heatmap_from_metres(zdf["x_m"], zdf["y_m"], a, b, NX, NY, SIGMA, w)
        fname = zname.lower().replace(" ", "_") + ".png"
        plot_heatmap(H_z, xe_z, ye_z, a, b, os.path.join(out_base, "zones", fname), title=zname)

# -------------------------
# Pipeline Entry Point
# -------------------------
def run_pipeline(inputs, out_root="outputs", sigma=2.0):
    """Run pipeline for multiple datasets."""
    global OUT_ROOT, SIGMA
    OUT_ROOT = out_root
    SIGMA = sigma

    for path, label in inputs:
        try:
            process_dataset(path, label)
        except Exception as e:
            print(f"!! Failed {label}: {e}")
    print(f"\nDone. Outputs in: {os.path.abspath(OUT_ROOT)}")

# -------------------------
# Console Interface
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AFL Heatmap Pipeline - generate heatmaps from tracking/event CSVs"
    )
    parser.add_argument(
        "datasets", nargs="+",
        help="Datasets in the form path:label (e.g., tracking.csv:tracking)"
    )
    parser.add_argument(
        "--out-dir", default="outputs",
        help="Output root directory (default: outputs)"
    )
    parser.add_argument(
        "--sigma", type=float, default=2.0,
        help="Gaussian blur sigma for smoothing (default: 2.0)"
    )
    args = parser.parse_args()

    # Parse dataset inputs
    inputs = []
    for ds in args.datasets:
        if ":" in ds:
            path, label = ds.split(":", 1)
        else:
            path = ds
            label = os.path.splitext(os.path.basename(ds))[0]
        inputs.append((path, label))

    # Run the pipeline
    run_pipeline(inputs, out_root=args.out_dir, sigma=args.sigma)

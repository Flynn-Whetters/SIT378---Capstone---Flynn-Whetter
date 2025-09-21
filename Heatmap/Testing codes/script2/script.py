# -*- coding: utf-8 -*-
"""
HeatMapPipeline — AFL Oval Heatmap Generator
===========================================

Generates AFL oval heatmaps from event CSVs and saves:
- PNG images with heatmaps overlaid on a stylised AFL field
- CSV files containing smoothed density grids (x_m, y_m, density)

---------------------------------------------------
Quick Start
---------------------------------------------------
# Single file
python3 heatmappipeline.py --inputs "events.csv:session1" --out-dir outputs

# Multiple files
python3 heatmappipeline.py \
    --inputs "kick.csv:kick" "mark.csv:mark" "tackle.csv:tackle" \
    --out-dir outputs

# Per-class heatmaps
python3 heatmappipeline.py \
    --inputs "events.csv" \
    --sigma 3.0 \
    --group-by class_id \
    --out-dir outputs
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

# -------------------------
# Defaults and schema
# -------------------------
REQUIRED_COLS = [
    "frame_id", "track_id", "x", "y", "width", "height", "conf", "class_id", "visibility"
]

DEFAULT_FIELD_LENGTH_M = 165.0
DEFAULT_FIELD_WIDTH_M  = 135.0
DEFAULT_NX = 200
DEFAULT_NY = 150
DEFAULT_SIGMA = 2.0

# -------------------------
# Schema + loading
# -------------------------
def assert_schema(df: pd.DataFrame, path: str):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Tip: ensure your header row is exactly: {REQUIRED_COLS}"
        )

def load_events_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert_schema(df, path)

    for c in ["x", "y", "width", "height", "conf", "visibility", "class_id", "track_id", "frame_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    df = df[np.isfinite(df["x"]) & np.isfinite(df["y"])]
    return df.reset_index(drop=True)

def choose_weights(df: pd.DataFrame, mode: str | None):
    if not mode:
        return None
    m = mode.lower().strip()
    if m == "conf":
        return df["conf"].to_numpy(dtype=float)
    if m == "visibility":
        return df["visibility"].to_numpy(dtype=float)
    if m in ("conf*visibility", "visibility*conf"):
        conf = df["conf"].fillna(0).to_numpy(dtype=float)
        vis  = df["visibility"].fillna(0).to_numpy(dtype=float)
        return conf * vis
    return None

# -------------------------
# Field + mapping helpers
# -------------------------
def raw_bbox(xs, ys, pad_ratio=0.02):
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    dx, dy = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    return (xmin - dx*pad_ratio, xmax + dx*pad_ratio,
            ymin - dy*pad_ratio, ymax + dy*pad_ratio)

def raw_to_metres(x, y, bbox_raw, a, b):
    xmin, xmax, ymin, ymax = bbox_raw
    x_m = ((x - xmin) / max(1e-9, (xmax - xmin))) * (2*a) - a
    y_m = ((y - ymin) / max(1e-9, (ymax - ymin))) * (2*b) - b
    return x_m, y_m

def make_oval_mask_metres(nx, ny, a, b):
    x_edges = np.linspace(-a, a, nx + 1)
    y_edges = np.linspace(-b, b, ny + 1)
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2
    Xc, Yc = np.meshgrid(xc, yc)
    mask = (Xc**2) / (a**2) + (Yc**2) / (b**2) <= 1.0
    return x_edges, y_edges, mask

def heatmap_in_metres(x_raw, y_raw, bbox_raw, a, b, nx, ny, sigma, weights=None):
    x_m, y_m = raw_to_metres(np.asarray(x_raw, float), np.asarray(y_raw, float), bbox_raw, a, b)
    x_edges, y_edges, mask = make_oval_mask_metres(nx, ny, a, b)
    H, _, _ = np.histogram2d(x_m, y_m, bins=[x_edges, y_edges], weights=weights)
    H = H.T
    Hs = gaussian_filter(H, sigma=sigma)
    Hs_masked = np.where(mask, Hs, np.nan)
    return Hs_masked, x_edges, y_edges

# -------------------------
# AFL field drawing
# -------------------------
def draw_afl_field_metres(ax, a, b,
                          centre_square=50.0,
                          centre_inner_d=3.0,
                          centre_outer_d=10.0,
                          goal_square_depth=9.0,
                          goal_square_width=6.4,
                          arc_r=50.0,
                          line_color="white", lw=1.8, alpha=0.9,
                          show_ticks=False):
    t = np.linspace(0, 2*np.pi, 800)
    ax.plot(a*np.cos(t), b*np.sin(t), color=line_color, lw=lw, alpha=alpha)
    ax.plot([0, 0], [-b, b], color=line_color, lw=lw, alpha=alpha)
    cs = centre_square/2.0
    ax.plot([-cs, cs, cs, -cs, -cs], [-cs, -cs, cs, cs, -cs],
            color=line_color, lw=lw-0.5, alpha=alpha)
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot((centre_outer_d/2.0)*np.cos(th), (centre_outer_d/2.0)*np.sin(th),
            color=line_color, lw=lw-0.5, alpha=alpha)
    ax.plot((centre_inner_d/2.0)*np.cos(th), (centre_inner_d/2.0)*np.sin(th),
            color=line_color, lw=lw-0.5, alpha=alpha)
    phi_L = np.linspace(-np.pi/2, np.pi/2, 400)
    phi_R = np.linspace(np.pi/2, 3*np.pi/2, 400)
    ax.plot(-a + arc_r*np.cos(phi_L), arc_r*np.sin(phi_L),
            color=line_color, lw=lw-0.5, alpha=alpha)
    ax.plot(a + arc_r*np.cos(phi_R), arc_r*np.sin(phi_R),
            color=line_color, lw=lw-0.5, alpha=alpha)
    gs_w = goal_square_width/2.0
    ax.plot([-a, -a+goal_square_depth, -a+goal_square_depth, -a, -a],
            [-gs_w, -gs_w, gs_w, gs_w, -gs_w],
            color=line_color, lw=lw-0.5, alpha=alpha)
    ax.plot([a, a-goal_square_depth, a-goal_square_depth, a, a],
            [-gs_w, -gs_w, gs_w, gs_w, -gs_w],
            color=line_color, lw=lw-0.5, alpha=alpha)
    tick = 4.0
    for xg in (-a, a):
        ax.plot([xg, xg], [-tick, tick], color=line_color, lw=lw, alpha=alpha)
    if show_ticks:
        ax.set_xlabel("Metres (X)")
        ax.set_ylabel("Metres (Y)")
        ax.set_xticks(np.arange(-a, a+1e-6, 20))
        ax.set_yticks(np.arange(-b, b+1e-6, 20))
        ax.grid(alpha=0.15, linewidth=0.8)
    else:
        ax.set_axis_off()
    ax.set_aspect('equal')

def plot_heatmap_on_field_metres(
    H, x_edges, y_edges, a, b, title="", alpha_img=0.88, out_path=None
):
    fig, ax = plt.subplots(figsize=(11, 8))
    t = np.linspace(0, 2*np.pi, 600)
    ax.fill(a*np.cos(t), b*np.sin(t), color="#2d6a4f", alpha=0.95, zorder=0)
    ax.set_facecolor("#1b4332")
    extent = [x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()]
    finite_vals = H[np.isfinite(H)]
    vmax = (np.nanpercentile(finite_vals, 99) if finite_vals.size else np.nanmax(H)) or 1.0

    im = ax.imshow(
        H, origin="lower", extent=extent, aspect="equal",
        interpolation="bilinear", alpha=alpha_img,
        cmap="plasma", norm=Normalize(vmin=0.0, vmax=vmax), zorder=2
    )
    draw_afl_field_metres(ax, a, b, show_ticks=False)
    sb_y = -b + 8
    sb_x0, sb_x1 = -a + 12, -a + 32
    ax.plot([sb_x0, sb_x1], [sb_y, sb_y], color="white", lw=4, alpha=0.95)
    ax.text((sb_x0+sb_x1)/2, sb_y-4, "20 m", ha="center", va="top",
            color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=15)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Event Density", fontsize=12, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=300, metadata={
            "Title": title,
            "Author": "HeatMapPipeline",
            "Description": "AFL heatmap generated from tracking data"
        })
    plt.close(fig)

def save_density_csv(H, x_edges, y_edges, out_csv_path):
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2
    Xc, Yc = np.meshgrid(xc, yc)
    df = pd.DataFrame({
        "x_m": Xc.flatten().round(2),
        "y_m": Yc.flatten().round(2),
        "density": np.nan_to_num(H, nan=0.0).flatten().round(4)
    })
    df.to_csv(out_csv_path, index=False)

# -------------------------
# Main
# -------------------------
def parse_inputs(pairs):
    out = []
    for p in pairs:
        if ":" in p:
            path, label = p.split(":", 1)
        else:
            path, label = p, os.path.splitext(os.path.basename(p))[0]
        out.append((path, label))
    return out

def main():
    ap = argparse.ArgumentParser(description="AFL oval heatmap generator")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help='List like "file.csv:label". Label optional.')
    ap.add_argument("--out-dir", default="heatmap_outputs", help="Output directory")
    ap.add_argument("--weight-mode", default="conf*visibility",
                    choices=["none", "conf", "visibility", "conf*visibility"],
                    help="Weighting for histogram bins")
    ap.add_argument("--field-length", type=float, default=DEFAULT_FIELD_LENGTH_M,
                    help="Oval length in metres (tip-to-tip)")
    ap.add_argument("--field-width", type=float, default=DEFAULT_FIELD_WIDTH_M,
                    help="Oval width in metres (wing-to-wing)")
    ap.add_argument("--nx", type=int, default=DEFAULT_NX, help="Grid bins in X")
    ap.add_argument("--ny", type=int, default=DEFAULT_NY, help="Grid bins in Y")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                    help="Gaussian smoothing sigma (bins)")
    ap.add_argument("--group-by", default=None,
                    choices=[None, "class_id"],
                    help="Optional grouping per class")
    args = ap.parse_args()

    inputs = parse_inputs(args.inputs)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    a = args.field_length / 2.0
    b = args.field_width  / 2.0

    loaded = []
    all_x, all_y = [], []
    for path, label in inputs:
        df = load_events_csv(path)
        loaded.append((df, label, path))
        all_x.append(df["x"].to_numpy())
        all_y.append(df["y"].to_numpy())

    all_x = np.concatenate(all_x) if len(all_x) else np.array([0.0])
    all_y = np.concatenate(all_y) if len(all_y) else np.array([0.0])
    shared_bbox_raw = raw_bbox(all_x, all_y)

    for df, label, path in loaded:
        weights_base = None if args.weight_mode == "none" else choose_weights(df, args.weight_mode)
        if args.group_by == "class_id":
            classes = sorted(df["class_id"].dropna().unique().tolist())
            for cid in classes:
                sub = df[df["class_id"] == cid].reset_index(drop=True)
                if sub.empty:
                    continue
                w = None if weights_base is None else choose_weights(sub, args.weight_mode)
                H, xe, ye = heatmap_in_metres(
                    sub["x"], sub["y"], shared_bbox_raw, a, b,
                    args.nx, args.ny, args.sigma, w
                )
                title = f"{label} — class_id {cid} ({int(args.field_length)}×{int(args.field_width)} m)"
                img_out = os.path.join(out_dir, "images", f"heatmap_{label}_class{int(cid)}.png")
                csv_out = os.path.join(out_dir, "csv", f"density_{label}_class{int(cid)}.csv")
                plot_heatmap_on_field_metres(H, xe, ye, a, b, title, out_path=img_out)
                save_density_csv(H, xe, ye, csv_out)
        else:
            H, xe, ye = heatmap_in_metres(
                df["x"], df["y"], shared_bbox_raw, a, b,
                args.nx, args.ny, args.sigma, weights_base
            )
            title = f"{label} — Density ({int(args.field_length)}×{int(args.field_width)} m)"
            img_out = os.path.join(out_dir, "images", f"heatmap_{label}.png")
            csv_out = os.path.join(out_dir, "csv", f"density_{label}.csv")
            plot_heatmap_on_field_metres(H, xe, ye, a, b, title, out_path=img_out)
            save_density_csv(H, xe, ye, csv_out)

    print(f"Done. Outputs in: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()

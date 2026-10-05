#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDMX saved-result fit diagnostics
================================

Standalone downstream analysis for results written by hdmx_all_salt_organized_v2.py.

IMPORTANT
---------
- This script DOES NOT rerun labeling, quench, LC, ESI, or any optimization.
- It only reads the already-saved final result files under `hdmx_results_all_salt/`.
- Experimental data are taken from the saved `Uptake` column.
- Final model predictions are taken from the saved `Uptake_ESI` column.

Main outputs
------------
1. Publication-style stage/fit plots for every peptide and salt condition,
   modeled after the original HDMX plot:
       Labeling -> Quench -> LC -> final ESI + experimental data by charge state.
2. Numerical validation tables with RMSE, MAE, bias, median/max absolute error,
   R^2, Pearson correlation, and normalized RMSE.
3. Metrics stratified by peptide, salt condition, exposure time, and charge state.
4. Exposure-dependent error plots to test whether prediction error decreases as
   exposure time increases.
5. Trend tables reporting first-to-last changes, slopes versus log10(exposure),
   and Spearman correlation. Negative slopes/correlations indicate decreasing
   error with increasing exposure; they do not by themselves establish causality.

Expected input structure
------------------------
hdmx_results_all_salt/
  0mM/
    exact_objects/dynamic_results.pkl
    tables/row_results_scalar.csv
  150mM_NaCl/
  150mM_CsCl/
  ...

Run this script from the directory containing `hdmx_results_all_salt`, or edit
AnalysisConfig.results_root below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class AnalysisConfig:
    results_root: str = "hdmx_results_all_salt"
    output_root: str = "hdmx_saved_fit_diagnostics"
    dpi: int = 300
    image_format: str = "png"
    show_plots: bool = False

    # Metrics compare these saved columns.
    experimental_col: str = "Uptake"
    predicted_col: str = "Uptake_ESI"

    # Smallest number of finite observations required for R^2 / correlation.
    min_points_for_association: int = 2

    # If True, make metric heatmaps in addition to the main requested plots.
    make_metric_heatmaps: bool = True


# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

def safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text or "unnamed"


def parse_state(state: str) -> tuple[float, str]:
    """Return salt concentration in mM and salt identity."""
    s = str(state).strip()
    if s == "0mM":
        return 0.0, "Baseline"

    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*(mM|M)(?:[_\-\s]+(.+))?\s*$", s, re.I)
    if not m:
        return np.nan, "Unknown"

    value = float(m.group(1))
    unit = m.group(2).lower()
    ion = (m.group(3) or "Unknown").strip()
    if unit == "m":
        value *= 1000.0
    return value, ion


def state_sort_key(state: str):
    c, ion = parse_state(state)
    ion_order = {"Baseline": 0, "NaCl": 1, "CsCl": 2, "Unknown": 9}
    c = c if np.isfinite(c) else float("inf")
    return (c, ion_order.get(ion, 8), str(state))


def add_state_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["State"].astype(str).map(parse_state)
    out["Salt_mM"] = [p[0] for p in parsed]
    out["Salt_type"] = [p[1] for p in parsed]
    return out


def save_figure(fig, path: Path, cfg: AnalysisConfig):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)


def make_output_dirs(cfg: AnalysisConfig) -> dict[str, Path]:
    root = Path(cfg.output_root)
    dirs = {
        "root": root,
        "tables": root / "00_tables",
        "stage_fit": root / "01_stage_fit_plots",
        "error_exposure": root / "02_error_vs_exposure",
        "heatmaps": root / "03_metric_heatmaps",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def finite_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


# =============================================================================
# 3. LOAD SAVED FINAL RESULTS ONLY
# =============================================================================

def discover_state_dirs(root: Path) -> list[tuple[str, Path]]:
    if not root.exists():
        raise FileNotFoundError(
            f"Saved-results folder not found: {root.resolve()}\n"
            "Run from the directory containing hdmx_results_all_salt, or change "
            "AnalysisConfig.results_root."
        )

    found: list[tuple[str, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        pkl = child / "exact_objects" / "dynamic_results.pkl"
        scalar = child / "tables" / "row_results_scalar.csv"
        manifest = child / "manifest.json"
        if not (pkl.exists() or scalar.exists() or manifest.exists()):
            continue

        state = child.name
        if manifest.exists():
            try:
                with open(manifest, "r", encoding="utf-8") as fh:
                    state = str(json.load(fh).get("state", state))
            except Exception:
                pass
        found.append((state, child))

    found.sort(key=lambda x: state_sort_key(x[0]))
    return found


def load_saved_results(cfg: AnalysisConfig) -> pd.DataFrame:
    """
    Load final saved rows. Prefer exact pickle; fall back to scalar CSV.
    No model or optimizer is imported or executed.
    """
    root = Path(cfg.results_root)
    state_dirs = discover_state_dirs(root)
    if not state_dirs:
        raise RuntimeError(f"No saved state folders found in {root.resolve()}")

    frames = []
    for state, state_dir in state_dirs:
        pkl = state_dir / "exact_objects" / "dynamic_results.pkl"
        scalar = state_dir / "tables" / "row_results_scalar.csv"

        if pkl.exists():
            df = pd.read_pickle(pkl)
            source = "exact pickle"
        elif scalar.exists():
            df = pd.read_csv(scalar)
            source = "scalar CSV"
        else:
            continue

        df = df.copy()
        df.attrs.clear()
        df["State"] = str(state)
        if "Sequence" in df.columns:
            df["Sequence"] = df["Sequence"].astype(str).str.strip()
        frames.append(df)
        print(f"Loaded {state}: {len(df)} rows ({source})")

    if not frames:
        raise RuntimeError("No readable saved result tables were found.")

    rows = pd.concat(frames, ignore_index=True)
    rows.attrs.clear()
    return add_state_metadata(rows)


# =============================================================================
# 4. METRICS
# =============================================================================

def calculate_metrics(
    df: pd.DataFrame,
    exp_col: str = "Uptake",
    pred_col: str = "Uptake_ESI",
    min_assoc_n: int = 2,
) -> dict[str, float]:
    """Calculate fit metrics from saved experimental and final predicted uptake."""
    if not {exp_col, pred_col} <= set(df.columns):
        return {}

    valid = df[[exp_col, pred_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if valid.empty:
        return {
            "N": 0,
            "RMSE_Da": np.nan,
            "RMSE_frac_max": np.nan,
            "NRMSE_range": np.nan,
            "MAE_Da": np.nan,
            "MeanBias_Da": np.nan,
            "MedianAE_Da": np.nan,
            "MaxAE_Da": np.nan,
            "R2": np.nan,
            "Pearson_r": np.nan,
        }

    exp_vals = valid[exp_col].to_numpy(dtype=float)
    pred_vals = valid[pred_col].to_numpy(dtype=float)
    residual = pred_vals - exp_vals
    abs_res = np.abs(residual)

    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(abs_res))
    bias = float(np.mean(residual))
    medae = float(np.median(abs_res))
    maxae = float(np.max(abs_res))

    max_abs_exp = float(np.max(np.abs(exp_vals))) if len(exp_vals) else np.nan
    rmse_frac_max = rmse / max_abs_exp if np.isfinite(max_abs_exp) and max_abs_exp > 0 else np.nan

    exp_range = float(np.max(exp_vals) - np.min(exp_vals)) if len(exp_vals) else np.nan
    nrmse_range = rmse / exp_range if np.isfinite(exp_range) and exp_range > 0 else np.nan

    if len(exp_vals) >= min_assoc_n:
        ss_res = float(np.sum((exp_vals - pred_vals) ** 2))
        ss_tot = float(np.sum((exp_vals - np.mean(exp_vals)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        if np.std(exp_vals) > 0 and np.std(pred_vals) > 0:
            pearson = float(np.corrcoef(exp_vals, pred_vals)[0, 1])
        else:
            pearson = np.nan
    else:
        r2 = np.nan
        pearson = np.nan

    return {
        "N": int(len(valid)),
        "RMSE_Da": rmse,
        "RMSE_frac_max": float(rmse_frac_max) if np.isfinite(rmse_frac_max) else np.nan,
        "NRMSE_range": float(nrmse_range) if np.isfinite(nrmse_range) else np.nan,
        "MAE_Da": mae,
        "MeanBias_Da": bias,
        "MedianAE_Da": medae,
        "MaxAE_Da": maxae,
        "R2": float(r2) if np.isfinite(r2) else np.nan,
        "Pearson_r": float(pearson) if np.isfinite(pearson) else np.nan,
    }


def grouped_metrics(
    rows: pd.DataFrame,
    group_cols: list[str],
    cfg: AnalysisConfig,
) -> pd.DataFrame:
    records = []
    available_groups = [c for c in group_cols if c in rows.columns]
    if not available_groups:
        return pd.DataFrame()

    for key, grp in rows.groupby(available_groups, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(available_groups, key_tuple))
        rec.update(
            calculate_metrics(
                grp,
                exp_col=cfg.experimental_col,
                pred_col=cfg.predicted_col,
                min_assoc_n=cfg.min_points_for_association,
            )
        )
        records.append(rec)
    return pd.DataFrame(records)


def add_row_errors(rows: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    out = rows.copy()
    exp = pd.to_numeric(out[cfg.experimental_col], errors="coerce")
    pred = pd.to_numeric(out[cfg.predicted_col], errors="coerce")
    out["Residual_Da"] = pred - exp
    out["AbsoluteError_Da"] = (pred - exp).abs()
    out["SquaredError_Da2"] = (pred - exp) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        out["RelativeError_frac"] = (pred - exp) / exp.replace(0, np.nan)
    return out


def spearman_corr(x, y) -> float:
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2:
        return np.nan
    xr = valid["x"].rank(method="average")
    yr = valid["y"].rank(method="average")
    if xr.std(ddof=0) == 0 or yr.std(ddof=0) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def exposure_trend_table(
    exposure_metrics: pd.DataFrame,
    grouping: list[str],
    metric_cols: tuple[str, ...] = ("RMSE_Da", "MAE_Da"),
) -> pd.DataFrame:
    """
    Summarize whether error metrics tend to decrease as exposure increases.

    Slope is fitted against log10(exposure_ms). A negative slope and negative
    Spearman rho indicate a decreasing metric with increasing exposure.
    """
    if exposure_metrics.empty or "Exposure" not in exposure_metrics.columns:
        return pd.DataFrame()

    group_cols = [c for c in grouping if c in exposure_metrics.columns]
    records = []

    for key, grp in exposure_metrics.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key_tuple))
        g = grp.copy()
        g["Exposure"] = pd.to_numeric(g["Exposure"], errors="coerce")
        g = g[g["Exposure"] > 0].sort_values("Exposure")
        if g.empty:
            continue

        x = np.log10(g["Exposure"].to_numpy(dtype=float))
        for metric in metric_cols:
            if metric not in g.columns:
                continue
            y = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            xv, yv = x[mask], y[mask]
            if len(yv) == 0:
                continue

            if len(yv) >= 2 and np.unique(xv).size >= 2:
                slope, intercept = np.polyfit(xv, yv, 1)
                rho = spearman_corr(xv, yv)
            else:
                slope = intercept = rho = np.nan

            first = float(yv[0])
            last = float(yv[-1])
            delta = last - first
            pct = 100.0 * delta / first if first != 0 else np.nan

            rec = dict(base)
            rec.update({
                "Metric": metric,
                "N_exposures": int(len(yv)),
                "First_exposure_ms": float(10 ** xv[0]),
                "Last_exposure_ms": float(10 ** xv[-1]),
                "First_value": first,
                "Last_value": last,
                "Delta_last_minus_first": delta,
                "Percent_change_first_to_last": pct,
                "Slope_per_log10_ms": float(slope) if np.isfinite(slope) else np.nan,
                "Spearman_rho_vs_logExposure": float(rho) if np.isfinite(rho) else np.nan,
                "Decreases_first_to_last": bool(last < first),
            })
            records.append(rec)

    return pd.DataFrame(records)


# =============================================================================
# 5. STAGE / FIT PLOTS (MODELED AFTER THE UPLOADED EXAMPLE)
# =============================================================================

def plot_stage_fit_all_states(rows: pd.DataFrame, outdir: Path, cfg: AnalysisConfig):
    needed = {
        "State", "Sequence", "Exposure", "z",
        cfg.experimental_col, cfg.predicted_col,
    }
    if not needed <= set(rows.columns):
        missing = sorted(needed - set(rows.columns))
        warnings.warn(f"Stage/fit plots skipped; missing columns: {missing}")
        return

    stage_specs = [
        ("Uptake_Labeling", "*", "-.", "1. Labeling"),
        ("Uptake_Quenched", "^", "-.", "2. After Quench"),
        ("Uptake_LC", "P", "--", "3. After LC"),
    ]

    for (state, seq), df_plot in rows.groupby(["State", "Sequence"], dropna=False):
        df_plot = df_plot.copy().sort_values("Exposure")
        valid_main = df_plot.dropna(subset=["Exposure", cfg.experimental_col, cfg.predicted_col, "z"])
        if valid_main.empty:
            continue

        protein_names = (
            df_plot["Protein"].dropna().astype(str).unique().tolist()
            if "Protein" in df_plot.columns else []
        )
        protein_name = protein_names[0] if protein_names else "Unknown protein"

        fig, ax = plt.subplots(figsize=(12, 7))

        # Physical-stage curves are shown as mean value at each exposure so that
        # repeated charge-state rows do not draw duplicate overlapping lines.
        stage_colors = ["black", "orange", "deepskyblue"]
        for (col, marker, style, label), color in zip(stage_specs, stage_colors):
            if col not in df_plot.columns:
                continue
            valid = df_plot.dropna(subset=["Exposure", col])
            if valid.empty:
                continue
            mean_vals = valid.groupby("Exposure", as_index=False)[col].mean(numeric_only=True)
            ax.plot(
                mean_vals["Exposure"], mean_vals[col],
                marker=marker, linestyle=style, linewidth=1.5,
                markersize=7, alpha=0.75, label=label, color=color,
            )

        unique_z = sorted(pd.to_numeric(valid_main["z"], errors="coerce").dropna().unique())
        cmap = plt.get_cmap("viridis")
        charge_colors = [cmap(v) for v in np.linspace(0.05, 0.80, max(len(unique_z), 1))]

        for idx, z_val in enumerate(unique_z):
            z_group = valid_main[pd.to_numeric(valid_main["z"], errors="coerce").eq(float(z_val))].copy()
            if z_group.empty:
                continue

            # Average replicates at each exposure before drawing the line/points.
            z_mean = (
                z_group.groupby("Exposure", as_index=False)[[cfg.experimental_col, cfg.predicted_col]]
                .mean(numeric_only=True)
                .sort_values("Exposure")
            )
            color = charge_colors[idx]

            ax.plot(
                z_mean["Exposure"], z_mean[cfg.predicted_col],
                color=color, linestyle="-", linewidth=2.5, alpha=0.95,
                label=f"After ESI (z={int(z_val) if float(z_val).is_integer() else z_val:g})",
            )
            ax.scatter(
                z_mean["Exposure"], z_mean[cfg.predicted_col],
                color=color, marker="D", s=58, alpha=1.0,
                edgecolor="white", linewidth=0.8,
            )
            ax.scatter(
                z_mean["Exposure"], z_mean[cfg.experimental_col],
                facecolors="none", edgecolors=color, marker="o", s=76,
                linewidth=1.6, alpha=0.9,
                label=f"Exp Data (z={int(z_val) if float(z_val).is_integer() else z_val:g})",
            )

            last_row = z_mean.iloc[-1]
            ax.text(
                float(last_row["Exposure"]) * 1.12,
                float(last_row[cfg.predicted_col]),
                f"z={int(z_val) if float(z_val).is_integer() else z_val:g}",
                color=color, fontweight="bold", va="center", fontsize=9,
            )

        metrics = calculate_metrics(
            df_plot,
            exp_col=cfg.experimental_col,
            pred_col=cfg.predicted_col,
            min_assoc_n=cfg.min_points_for_association,
        )
        metric_text = f"RMSE = {metrics.get('RMSE_Da', np.nan):.4f} Da"
        ax.text(
            0.035, 0.965, metric_text,
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.16),
        )

        exposures = pd.to_numeric(df_plot["Exposure"], errors="coerce").dropna()
        if not exposures.empty and (exposures > 0).all():
            ax.set_xscale("log")
            ax.set_xlim(float(exposures.min()) * 0.8, float(exposures.max()) * 4.0)

        ax.set_xlabel("Exposure Time (ms)", fontsize=13)
        ax.set_ylabel("Deuterium Uptake (Da)", fontsize=13)
        ax.set_title(
            f"Protein: {protein_name} , {seq} | State: {state}",
            fontsize=17, fontweight="bold", pad=18,
        )
        ax.grid(True, which="both", linestyle="--", alpha=0.25)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(), by_label.keys(),
            loc="lower right", ncol=2, fontsize=8,
            frameon=True, framealpha=0.85,
        )

        save_figure(
            fig,
            outdir / safe_name(state) / f"{safe_name(seq)}.{cfg.image_format}",
            cfg,
        )


# =============================================================================
# 6. EXPOSURE-DEPENDENT ERROR PLOTS
# =============================================================================

def plot_rmse_vs_exposure_across_states(
    metrics_exposure: pd.DataFrame,
    outdir: Path,
    cfg: AnalysisConfig,
):
    """For each peptide: pooled-charge RMSE vs exposure, one line per salt state."""
    if metrics_exposure.empty or not {"Sequence", "State", "Exposure", "RMSE_Da"} <= set(metrics_exposure.columns):
        return

    for seq, seq_df in metrics_exposure.groupby("Sequence"):
        fig, ax = plt.subplots(figsize=(9.2, 5.8))
        drawn = False
        for state in sorted(seq_df["State"].astype(str).unique(), key=state_sort_key):
            g = seq_df[seq_df["State"].astype(str).eq(state)].copy()
            g = g.sort_values("Exposure")
            if g.empty:
                continue
            ax.plot(g["Exposure"], g["RMSE_Da"], marker="o", linewidth=1.8, label=state)
            drawn = True
        if not drawn:
            plt.close(fig)
            continue

        if (pd.to_numeric(seq_df["Exposure"], errors="coerce") > 0).all():
            ax.set_xscale("log")
        ax.set_xlabel("Exposure time (ms)")
        ax.set_ylabel("RMSE across charge states / replicates (Da)")
        ax.set_title(f"RMSE vs exposure | {seq} | all salt conditions")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        save_figure(fig, outdir / "pooled_charge_by_state" / f"{safe_name(seq)}.{cfg.image_format}", cfg)


def plot_error_vs_exposure_by_charge(
    metrics_charge_exposure: pd.DataFrame,
    outdir: Path,
    cfg: AnalysisConfig,
):
    """
    For every peptide x state, plot RMSE and MAE vs exposure with one line per z.

    Note: if a charge/exposure group contains one observation, RMSE equals the
    absolute error for that point. The table records N so this is transparent.
    """
    needed = {"Sequence", "State", "z", "Exposure", "RMSE_Da", "MAE_Da"}
    if metrics_charge_exposure.empty or not needed <= set(metrics_charge_exposure.columns):
        return

    for (state, seq), grp in metrics_charge_exposure.groupby(["State", "Sequence"], dropna=False):
        for metric, ylabel in [
            ("RMSE_Da", "RMSE within charge state (Da)"),
            ("MAE_Da", "MAE within charge state (Da)"),
        ]:
            fig, ax = plt.subplots(figsize=(8.8, 5.6))
            drawn = False
            for z, g in grp.groupby("z", dropna=False):
                g = g.sort_values("Exposure")
                if g.empty:
                    continue
                label = f"z={int(z)}" if pd.notna(z) and float(z).is_integer() else f"z={z}"
                ax.plot(g["Exposure"], g[metric], marker="o", linewidth=1.7, label=label)
                drawn = True
            if not drawn:
                plt.close(fig)
                continue

            if (pd.to_numeric(grp["Exposure"], errors="coerce") > 0).all():
                ax.set_xscale("log")
            ax.set_xlabel("Exposure time (ms)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{metric} vs exposure by charge state | {seq} | {state}")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(title="Charge")
            save_figure(
                fig,
                outdir / "by_state_charge" / safe_name(state) / f"{safe_name(seq)}_{metric}.{cfg.image_format}",
                cfg,
            )


def plot_pointwise_absolute_error(
    row_errors: pd.DataFrame,
    outdir: Path,
    cfg: AnalysisConfig,
):
    """Experimental pointwise absolute error versus exposure, separated by charge."""
    needed = {"Sequence", "State", "z", "Exposure", "AbsoluteError_Da"}
    if not needed <= set(row_errors.columns):
        return

    for (state, seq), grp in row_errors.groupby(["State", "Sequence"], dropna=False):
        fig, ax = plt.subplots(figsize=(8.8, 5.6))
        drawn = False
        for z, g in grp.groupby("z", dropna=False):
            g = g.dropna(subset=["Exposure", "AbsoluteError_Da"]).copy()
            if g.empty:
                continue
            # If replicates exist, show mean absolute error and SD error bars.
            summary = (
                g.groupby("Exposure", as_index=False)["AbsoluteError_Da"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("Exposure")
            )
            label = f"z={int(z)}" if pd.notna(z) and float(z).is_integer() else f"z={z}"
            ax.plot(summary["Exposure"], summary["mean"], marker="o", linewidth=1.6, label=label)
            if summary["count"].max() > 1:
                std = summary["std"].fillna(0.0).to_numpy(dtype=float)
                ax.fill_between(
                    summary["Exposure"].to_numpy(dtype=float),
                    np.maximum(summary["mean"].to_numpy(dtype=float) - std, 0.0),
                    summary["mean"].to_numpy(dtype=float) + std,
                    alpha=0.12,
                )
            drawn = True
        if not drawn:
            plt.close(fig)
            continue

        if (pd.to_numeric(grp["Exposure"], errors="coerce") > 0).all():
            ax.set_xscale("log")
        ax.set_xlabel("Exposure time (ms)")
        ax.set_ylabel("Mean absolute prediction error (Da)")
        ax.set_title(f"Absolute error vs exposure by charge state | {seq} | {state}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(title="Charge")
        save_figure(
            fig,
            outdir / "absolute_error_by_state_charge" / safe_name(state) / f"{safe_name(seq)}.{cfg.image_format}",
            cfg,
        )


# =============================================================================
# 7. OPTIONAL METRIC HEATMAPS
# =============================================================================

def plot_metric_heatmap(
    table: pd.DataFrame,
    metric: str,
    outdir: Path,
    cfg: AnalysisConfig,
):
    if table.empty or not {"Sequence", "State", metric} <= set(table.columns):
        return

    pivot = table.pivot_table(index="Sequence", columns="State", values=metric, aggfunc="mean")
    if pivot.empty:
        return
    pivot = pivot.reindex(columns=sorted(pivot.columns.astype(str), key=state_sort_key))

    fig, ax = plt.subplots(
        figsize=(max(8.0, 1.15 * len(pivot.columns)), max(4.0, 0.75 * len(pivot.index) + 2.0))
    )
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Salt condition")
    ax.set_ylabel("Peptide")
    ax.set_title(f"{metric} by peptide and salt condition")
    fig.colorbar(im, ax=ax, label=metric)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    save_figure(fig, outdir / f"{safe_name(metric)}_peptide_state.{cfg.image_format}", cfg)


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    cfg = AnalysisConfig()
    dirs = make_output_dirs(cfg)

    print("=" * 92)
    print("HDMX SAVED-RESULT FIT DIAGNOSTICS")
    print("=" * 92)
    print(f"Reading saved results from: {Path(cfg.results_root).resolve()}")
    print("No model equations, forward calculations, or optimization will be rerun.\n")

    rows = load_saved_results(cfg)

    required = {cfg.experimental_col, cfg.predicted_col, "State", "Sequence", "Exposure", "z"}
    missing = required - set(rows.columns)
    if missing:
        raise KeyError(f"Saved results are missing required columns: {sorted(missing)}")

    row_errors = add_row_errors(rows, cfg)

    # -------------------------------------------------------------------------
    # Validation tables at several resolutions.
    # -------------------------------------------------------------------------
    metrics_peptide_state = grouped_metrics(
        rows,
        ["State", "Salt_mM", "Salt_type", "Protein", "Sequence"],
        cfg,
    )

    metrics_peptide_state_charge = grouped_metrics(
        rows,
        ["State", "Salt_mM", "Salt_type", "Protein", "Sequence", "z"],
        cfg,
    )

    # Pooled across charge states (and replicates) at each exposure.
    metrics_peptide_state_exposure = grouped_metrics(
        rows,
        ["State", "Salt_mM", "Salt_type", "Protein", "Sequence", "Exposure"],
        cfg,
    )

    # Most granular requested table: peptide x salt x charge x exposure.
    metrics_peptide_state_charge_exposure = grouped_metrics(
        rows,
        ["State", "Salt_mM", "Salt_type", "Protein", "Sequence", "z", "Exposure"],
        cfg,
    )

    trends_pooled_charge = exposure_trend_table(
        metrics_peptide_state_exposure,
        grouping=["State", "Salt_mM", "Salt_type", "Protein", "Sequence"],
    )
    if not trends_pooled_charge.empty:
        trends_pooled_charge.insert(0, "Trend_level", "Pooled across charge states")

    trends_by_charge = exposure_trend_table(
        metrics_peptide_state_charge_exposure,
        grouping=["State", "Salt_mM", "Salt_type", "Protein", "Sequence", "z"],
    )
    if not trends_by_charge.empty:
        trends_by_charge.insert(0, "Trend_level", "Within charge state")

    exposure_trends = pd.concat(
        [x for x in [trends_pooled_charge, trends_by_charge] if not x.empty],
        ignore_index=True,
    ) if (not trends_pooled_charge.empty or not trends_by_charge.empty) else pd.DataFrame()

    tables = {
        "row_level_prediction_errors.csv": row_errors,
        "metrics_by_peptide_and_state.csv": metrics_peptide_state,
        "metrics_by_peptide_state_and_charge.csv": metrics_peptide_state_charge,
        "metrics_by_peptide_state_and_exposure.csv": metrics_peptide_state_exposure,
        "metrics_by_peptide_state_charge_and_exposure.csv": metrics_peptide_state_charge_exposure,
        "exposure_error_trends.csv": exposure_trends,
    }
    for filename, df in tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(dirs["tables"] / filename, index=False)

    # -------------------------------------------------------------------------
    # Figures.
    # -------------------------------------------------------------------------
    print("Creating full stage/fit plots for every peptide and salt condition...")
    plot_stage_fit_all_states(rows, dirs["stage_fit"], cfg)

    print("Creating exposure-dependent RMSE/error plots...")
    plot_rmse_vs_exposure_across_states(metrics_peptide_state_exposure, dirs["error_exposure"], cfg)
    plot_error_vs_exposure_by_charge(metrics_peptide_state_charge_exposure, dirs["error_exposure"], cfg)
    plot_pointwise_absolute_error(row_errors, dirs["error_exposure"], cfg)

    if cfg.make_metric_heatmaps:
        print("Creating metric heatmaps...")
        for metric in ["RMSE_Da", "MAE_Da", "MeanBias_Da", "R2"]:
            plot_metric_heatmap(metrics_peptide_state, metric, dirs["heatmaps"], cfg)

    manifest = f"""
HDMX saved-result fit diagnostics completed.

SOURCE
------
{Path(cfg.results_root).resolve()}

No HDMX forward model or optimizer was rerun. All metrics use the saved final
experimental uptake column `{cfg.experimental_col}` and saved final model column
`{cfg.predicted_col}`.

OUTPUTS
-------
00_tables/
  row_level_prediction_errors.csv
      Every saved measurement with residual, absolute error, and squared error.

  metrics_by_peptide_and_state.csv
      Overall RMSE, MAE, bias, R2, etc. for each peptide in each salt condition.

  metrics_by_peptide_state_and_charge.csv
      Same metrics separately for every peptide x salt x charge state.

  metrics_by_peptide_state_and_exposure.csv
      Metrics at each exposure, pooling available charge states/replicates. This is
      the preferred table for asking whether RMSE changes with exposure.

  metrics_by_peptide_state_charge_and_exposure.csv
      Most granular requested table. If N=1, RMSE equals the absolute error for
      that single observation; inspect N before interpreting it as a distributional
      RMSE.

  exposure_error_trends.csv
      First-to-last change, log-exposure slope, and Spearman correlation for RMSE
      and MAE. Negative values indicate decreasing error with increasing exposure.

01_stage_fit_plots/
  One plot per peptide per salt state, showing Labeling -> Quench -> LC -> ESI,
  experimental points by charge state, and an annotation box with RMSE only.

02_error_vs_exposure/
  pooled_charge_by_state/
      For each peptide, RMSE vs exposure with one line per salt condition.

  by_state_charge/
      For each peptide x salt condition, RMSE and MAE vs exposure with one line
      per charge state.

  absolute_error_by_state_charge/
      Pointwise/replicate-mean absolute error vs exposure by charge state.

03_metric_heatmaps/
  Peptide x salt-condition heatmaps for selected overall metrics.

INTERPRETATION NOTE
-------------------
A decreasing RMSE/MAE curve with exposure is descriptive evidence that the saved
model predictions are closer to the experimental uptake at later exposure times.
It should not be assumed in advance; inspect the trend table and plots. At the
charge-state + exposure level, groups with N=1 have RMSE == absolute error.
""".strip() + "\n"

    (dirs["root"] / "README.txt").write_text(manifest, encoding="utf-8")

    print("\n" + "=" * 92)
    print("DIAGNOSTICS FINISHED")
    print(f"Output directory: {dirs['root'].resolve()}")
    print("Key outputs:")
    print("  - 01_stage_fit_plots/")
    print("  - 00_tables/metrics_by_peptide_and_state.csv")
    print("  - 00_tables/metrics_by_peptide_state_and_exposure.csv")
    print("  - 00_tables/metrics_by_peptide_state_charge_and_exposure.csv")
    print("  - 00_tables/exposure_error_trends.csv")
    print("  - 02_error_vs_exposure/")
    print("=" * 92)


if __name__ == "__main__":
    main()

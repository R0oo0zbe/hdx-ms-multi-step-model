#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDMX charge-state scientific diagnostics from SAVED all-salt results.

This script is intentionally downstream-only. It DOES NOT rerun the HDMX model,
optimization, labeling, quench, LC, or ESI calculations. It reads the files saved
by hdmx_all_salt_organized_v2.py and creates charge-state-focused tables and figures.

Main scientific questions addressed
-----------------------------------
1. How does experimental uptake change with charge state, and does the final model
   reproduce that charge dependence?
2. Which physical/model stage contributes most to deuterium loss as charge changes?
3. How does prediction error (RMSE, MAE, bias, absolute error) depend on charge state
   and exposure time?
4. Does the error decrease as exposure time increases?
5. How does charge-state separation (high-z minus low-z uptake) vary with exposure,
   salt identity, and salt concentration?
6. How do charge-state-specific model errors and ESI losses vary with salt concentration?

Expected input directory
------------------------
hdmx_results_all_salt/
    0mM/
    150mM_NaCl/
    150mM_CsCl/
    500mM_NaCl/
    500mM_CsCl/
    1M_NaCl/
    1M_CsCl/
    ...

For each state the script prefers:
    exact_objects/dynamic_results.pkl
and falls back to:
    tables/row_results_scalar.csv

Authoring principle
-------------------
All quantities in this script are calculated from already-saved final model outputs.
No fitted parameter or theoretical equation is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
class ChargeDiagnosticsConfig:
    results_root: str = "hdmx_results_all_salt"
    output_root: str = "hdmx_charge_state_diagnostics"
    baseline_state: str = "0mM"

    dpi: int = 300
    image_format: str = "png"
    show_plots: bool = False

    # Stage-loss vs charge plots can be numerous. "representative" gives the
    # first, middle, and last exposure with >= 2 charge states; "all" gives all.
    exposure_plot_mode: str = "representative"  # representative | all

    # Numerical tolerance only for describing a nearly flat exposure-error trend.
    trend_flat_tolerance_da: float = 1e-4


SCALAR_STAGE_COLUMNS = [
    "Uptake",
    "Uptake_Labeling",
    "Uptake_Quenched",
    "Uptake_LC",
    "Uptake_ESI",
]


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

    match = re.match(
        r"^\s*([0-9]*\.?[0-9]+)\s*(mM|M)(?:[_\-\s]+(.+))?\s*$",
        s,
        flags=re.IGNORECASE,
    )
    if not match:
        return np.nan, "Unknown"

    value = float(match.group(1))
    unit = match.group(2).lower()
    salt_type = (match.group(3) or "Unknown").strip()
    if unit == "m":
        value *= 1000.0
    return value, salt_type


def state_sort_key(state: str):
    c, ion = parse_state(state)
    ion_order = {"Baseline": 0, "NaCl": 1, "CsCl": 2, "Unknown": 9}
    c = c if np.isfinite(c) else float("inf")
    return (c, ion_order.get(ion, 8), str(state))


def add_state_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["State"].astype(str).map(parse_state)
    out["Salt_mM"] = [x[0] for x in parsed]
    out["Salt_type"] = [x[1] for x in parsed]
    return out


def make_output_dirs(cfg: ChargeDiagnosticsConfig) -> dict[str, Path]:
    root = Path(cfg.output_root)
    dirs = {
        "root": root,
        "tables": root / "00_tables",
        "uptake": root / "01_charge_uptake_validation",
        "loss": root / "02_stage_loss_vs_charge",
        "differential": root / "03_charge_differential_vs_exposure",
        "error_exposure": root / "04_error_vs_exposure",
        "heatmaps": root / "05_charge_exposure_heatmaps",
        "separation": root / "06_charge_separation",
        "slope": root / "07_charge_effect_slope",
        "salt_error": root / "08_error_vs_salt_concentration",
        "salt_loss": root / "09_esi_loss_vs_salt_concentration",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_figure(fig, path: Path, cfg: ChargeDiagnosticsConfig):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)


def style_axis(ax):
    ax.grid(True, which="major", linestyle="--", alpha=0.28)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15)
    ax.spines["top"].set_alpha(0.35)
    ax.spines["right"].set_alpha(0.35)


def charge_colors(z_values) -> dict[float, object]:
    vals = sorted(float(z) for z in pd.Series(z_values).dropna().unique())
    if not vals:
        return {}
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(vals)))
    return {z: colors[i] for i, z in enumerate(vals)}


def representative_exposures(values) -> list[float]:
    vals = sorted(set(float(v) for v in values if np.isfinite(v)))
    if len(vals) <= 3:
        return vals
    return [vals[0], vals[len(vals) // 2], vals[-1]]


def common_limits(x, y):
    a = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy(dtype=float)
    vals = np.concatenate([a, b]) if len(a) + len(b) else np.array([])
    if len(vals) == 0:
        return 0.0, 1.0
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if math.isclose(lo, hi):
        pad = 1.0 if lo == 0 else 0.1 * abs(lo)
        return lo - pad, hi + pad
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


# =============================================================================
# 3. LOAD SAVED RESULTS
# =============================================================================

def discover_state_dirs(root: Path) -> list[tuple[str, Path]]:
    if not root.exists():
        raise FileNotFoundError(
            f"Saved-results folder not found: {root.resolve()}\n"
            "Run this script from the directory containing hdmx_results_all_salt "
            "or change ChargeDiagnosticsConfig.results_root."
        )

    found = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        pkl = child / "exact_objects" / "dynamic_results.pkl"
        scalar = child / "tables" / "row_results_scalar.csv"
        manifest = child / "manifest.json"
        if not (pkl.exists() or scalar.exists() or manifest.exists()):
            continue
        state = child.name
        found.append((state, child))

    found.sort(key=lambda x: state_sort_key(x[0]))
    return found


def load_saved_rows(cfg: ChargeDiagnosticsConfig) -> pd.DataFrame:
    root = Path(cfg.results_root)
    state_dirs = discover_state_dirs(root)
    if not state_dirs:
        raise RuntimeError(f"No saved state directories found in {root.resolve()}")

    frames = []
    for state, state_dir in state_dirs:
        pkl = state_dir / "exact_objects" / "dynamic_results.pkl"
        scalar = state_dir / "tables" / "row_results_scalar.csv"

        if pkl.exists():
            df = pd.read_pickle(pkl)
        elif scalar.exists():
            df = pd.read_csv(scalar)
        else:
            warnings.warn(f"Skipping {state}: no dynamic_results.pkl or row_results_scalar.csv")
            continue

        df = df.copy()
        df.attrs.clear()
        df["State"] = str(state)
        if "Sequence" in df.columns:
            df["Sequence"] = df["Sequence"].astype(str).str.strip()
        frames.append(df)

    if not frames:
        raise RuntimeError("No saved result rows could be loaded.")

    rows = pd.concat(frames, ignore_index=True)
    rows.attrs.clear()
    rows = add_state_metadata(rows)

    required = {"State", "Sequence", "Exposure", "z", "Uptake", "Uptake_ESI"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Saved results are missing required columns: {sorted(missing)}")

    for col in ["Exposure", "z", "Salt_mM"] + SCALAR_STAGE_COLUMNS:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")

    return rows


# =============================================================================
# 4. CORE DIAGNOSTIC TABLES
# =============================================================================

def add_error_and_loss_columns(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()

    out["Residual_Model_minus_Experimental"] = out["Uptake_ESI"] - out["Uptake"]
    out["Absolute_Error_Da"] = np.abs(out["Residual_Model_minus_Experimental"])
    out["Squared_Error_Da2"] = out["Residual_Model_minus_Experimental"] ** 2

    if {"Uptake_Labeling", "Uptake_Quenched"} <= set(out.columns):
        out["Loss_Quench_Da"] = out["Uptake_Labeling"] - out["Uptake_Quenched"]
    if {"Uptake_Quenched", "Uptake_LC"} <= set(out.columns):
        out["Loss_LC_Da"] = out["Uptake_Quenched"] - out["Uptake_LC"]
    if {"Uptake_LC", "Uptake_ESI"} <= set(out.columns):
        out["Loss_ESI_Da"] = out["Uptake_LC"] - out["Uptake_ESI"]
        out["ESI_Loss_Fraction"] = (
            out["Loss_ESI_Da"] / out["Uptake_LC"].replace(0, np.nan)
        )
    if {"Uptake_Labeling", "Uptake_ESI"} <= set(out.columns):
        out["Loss_Total_PostLabel_Da"] = out["Uptake_Labeling"] - out["Uptake_ESI"]
        out["Total_Loss_Fraction"] = (
            out["Loss_Total_PostLabel_Da"] / out["Uptake_Labeling"].replace(0, np.nan)
        )

    return out


def aggregate_for_visualization(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in [
        "State", "Salt_mM", "Salt_type", "Protein", "Sequence", "Exposure", "z"
    ] if c in rows.columns]

    value_cols = [c for c in [
        "Uptake", "Uptake_Labeling", "Uptake_Quenched", "Uptake_LC", "Uptake_ESI",
        "Residual_Model_minus_Experimental", "Absolute_Error_Da", "Squared_Error_Da2",
        "Loss_Quench_Da", "Loss_LC_Da", "Loss_ESI_Da", "Loss_Total_PostLabel_Da",
        "ESI_Loss_Fraction", "Total_Loss_Fraction",
    ] if c in rows.columns]

    return (
        rows[group_cols + value_cols]
        .groupby(group_cols, dropna=False, as_index=False)
        .mean(numeric_only=True)
    )


def calculate_fit_metrics(df: pd.DataFrame) -> dict[str, float]:
    valid = df.dropna(subset=["Uptake", "Uptake_ESI"]).copy()
    n = int(len(valid))
    if n == 0:
        return {
            "N": 0,
            "RMSE_Da": np.nan,
            "NRMSE_max": np.nan,
            "NRMSE_percent": np.nan,
            "MAE_Da": np.nan,
            "Bias_Da": np.nan,
            "MedianAE_Da": np.nan,
            "MaxAE_Da": np.nan,
            "R2": np.nan,
            "Pearson_r": np.nan,
        }

    exp = valid["Uptake"].to_numpy(dtype=float)
    pred = valid["Uptake_ESI"].to_numpy(dtype=float)
    residual = pred - exp
    abs_err = np.abs(residual)

    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(abs_err))
    bias = float(np.mean(residual))
    medae = float(np.median(abs_err))
    maxae = float(np.max(abs_err))

    denom = float(np.nanmax(np.abs(exp))) if len(exp) else np.nan
    nrmse = float(rmse / denom) if np.isfinite(denom) and denom > 0 else np.nan

    ss_res = float(np.sum((pred - exp) ** 2))
    ss_tot = float(np.sum((exp - np.mean(exp)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if n >= 2 and ss_tot > 0 else np.nan
    pearson = float(np.corrcoef(exp, pred)[0, 1]) if n >= 2 and np.std(exp) > 0 and np.std(pred) > 0 else np.nan

    return {
        "N": n,
        "RMSE_Da": rmse,
        "NRMSE_max": nrmse,
        "NRMSE_percent": nrmse * 100.0 if np.isfinite(nrmse) else np.nan,
        "MAE_Da": mae,
        "Bias_Da": bias,
        "MedianAE_Da": medae,
        "MaxAE_Da": maxae,
        "R2": r2,
        "Pearson_r": pearson,
    }


def metrics_by_group(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    records = []
    for key, grp in rows.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(group_cols, key_tuple))
        rec.update(calculate_fit_metrics(grp))
        records.append(rec)
    return pd.DataFrame(records)


def build_stage_loss_summary(agg: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in [
        "Loss_Quench_Da", "Loss_LC_Da", "Loss_ESI_Da", "Loss_Total_PostLabel_Da",
        "ESI_Loss_Fraction", "Total_Loss_Fraction",
    ] if c in agg.columns]
    keep = [c for c in [
        "State", "Salt_mM", "Salt_type", "Protein", "Sequence", "Exposure", "z"
    ] if c in agg.columns]
    return agg[keep + cols].copy()


def build_charge_separation(agg: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in [
        "State", "Salt_mM", "Salt_type", "Protein", "Sequence", "Exposure"
    ] if c in agg.columns]
    records = []

    for key, grp in agg.groupby(group_cols, dropna=False):
        g = grp.dropna(subset=["z"]).copy()
        if g["z"].nunique() < 2:
            continue

        z_low = float(g["z"].min())
        z_high = float(g["z"].max())
        lo = g[g["z"].eq(z_low)]
        hi = g[g["z"].eq(z_high)]

        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rec["z_low"] = z_low
        rec["z_high"] = z_high

        def diff_high_low(col):
            if col not in g.columns:
                return np.nan
            return float(hi[col].mean() - lo[col].mean())

        rec["ChargeSeparation_Experimental_Da"] = diff_high_low("Uptake")
        rec["ChargeSeparation_Model_Da"] = diff_high_low("Uptake_ESI")
        rec["ChargeSeparation_Error_Da"] = (
            rec["ChargeSeparation_Model_Da"] - rec["ChargeSeparation_Experimental_Da"]
        )

        for col in ["Loss_Quench_Da", "Loss_LC_Da", "Loss_ESI_Da", "Loss_Total_PostLabel_Da"]:
            if col in g.columns:
                rec[f"Delta_{col}_high_minus_low"] = diff_high_low(col)

        records.append(rec)

    return pd.DataFrame(records)


def build_charge_differentials(agg: pd.DataFrame) -> pd.DataFrame:
    """For every z, calculate uptake/loss change relative to the lowest charge state."""
    group_cols = [c for c in [
        "State", "Salt_mM", "Salt_type", "Protein", "Sequence", "Exposure"
    ] if c in agg.columns]
    records = []

    for key, grp in agg.groupby(group_cols, dropna=False):
        g = grp.dropna(subset=["z"]).sort_values("z")
        if g["z"].nunique() < 2:
            continue
        z_ref = float(g["z"].min())
        ref = g[g["z"].eq(z_ref)].mean(numeric_only=True)

        for _, row in g.iterrows():
            z = float(row["z"])
            if math.isclose(z, z_ref):
                continue
            rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
            rec["z_reference"] = z_ref
            rec["z"] = z
            rec["Delta_z"] = z - z_ref
            for col in [
                "Uptake", "Uptake_ESI", "Loss_Quench_Da", "Loss_LC_Da",
                "Loss_ESI_Da", "Loss_Total_PostLabel_Da",
            ]:
                if col in g.columns and col in ref.index:
                    rec[f"Delta_{col}_vs_lowest_z"] = float(row[col] - ref[col])
            if {"Delta_Uptake_vs_lowest_z", "Delta_Uptake_ESI_vs_lowest_z"} <= set(rec):
                rec["ChargeDifferential_Model_minus_Experimental_Da"] = (
                    rec["Delta_Uptake_ESI_vs_lowest_z"] - rec["Delta_Uptake_vs_lowest_z"]
                )
            records.append(rec)

    return pd.DataFrame(records)


def build_charge_effect_slopes(agg: pd.DataFrame) -> pd.DataFrame:
    """Linear uptake-vs-charge slope at each state/peptide/exposure."""
    group_cols = [c for c in [
        "State", "Salt_mM", "Salt_type", "Protein", "Sequence", "Exposure"
    ] if c in agg.columns]
    records = []

    for key, grp in agg.groupby(group_cols, dropna=False):
        g = grp.dropna(subset=["z", "Uptake", "Uptake_ESI"]).copy()
        if g["z"].nunique() < 2:
            continue

        z = g["z"].to_numpy(dtype=float)
        exp = g["Uptake"].to_numpy(dtype=float)
        model = g["Uptake_ESI"].to_numpy(dtype=float)

        exp_slope, exp_intercept = np.polyfit(z, exp, 1)
        model_slope, model_intercept = np.polyfit(z, model, 1)

        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rec.update({
            "N_charge_states": int(g["z"].nunique()),
            "Experimental_dUptake_dz_Da_per_charge": float(exp_slope),
            "Model_dUptake_dz_Da_per_charge": float(model_slope),
            "Slope_Error_Da_per_charge": float(model_slope - exp_slope),
            "Experimental_intercept": float(exp_intercept),
            "Model_intercept": float(model_intercept),
        })
        records.append(rec)

    return pd.DataFrame(records)


def build_exposure_error_trends(exposure_metrics: pd.DataFrame, cfg: ChargeDiagnosticsConfig) -> pd.DataFrame:
    """Quantify whether RMSE tends to decrease or increase with exposure."""
    if exposure_metrics.empty:
        return pd.DataFrame()

    group_cols = [c for c in ["State", "Salt_mM", "Salt_type", "Protein", "Sequence", "z"] if c in exposure_metrics.columns]
    records = []

    for key, grp in exposure_metrics.groupby(group_cols, dropna=False):
        g = grp.dropna(subset=["Exposure", "RMSE_Da"]).copy()
        g = g[g["Exposure"] > 0].sort_values("Exposure")
        if len(g) < 2:
            continue

        x = np.log10(g["Exposure"].to_numpy(dtype=float))
        y = g["RMSE_Da"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)

        x_rank = pd.Series(x).rank().to_numpy(dtype=float)
        y_rank = pd.Series(y).rank().to_numpy(dtype=float)
        spearman = (
            float(np.corrcoef(x_rank, y_rank)[0, 1])
            if len(g) >= 2 and np.std(x_rank) > 0 and np.std(y_rank) > 0
            else np.nan
        )

        first_rmse = float(g.iloc[0]["RMSE_Da"])
        last_rmse = float(g.iloc[-1]["RMSE_Da"])
        delta = last_rmse - first_rmse
        tol = cfg.trend_flat_tolerance_da
        if delta < -tol:
            trend = "lower_error_at_last_exposure"
        elif delta > tol:
            trend = "higher_error_at_last_exposure"
        else:
            trend = "approximately_flat_first_to_last"

        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rec.update({
            "N_exposures": int(len(g)),
            "First_exposure_ms": float(g.iloc[0]["Exposure"]),
            "Last_exposure_ms": float(g.iloc[-1]["Exposure"]),
            "RMSE_first_Da": first_rmse,
            "RMSE_last_Da": last_rmse,
            "Delta_RMSE_last_minus_first_Da": delta,
            "RMSE_slope_per_log10_exposure": float(slope),
            "RMSE_intercept": float(intercept),
            "Spearman_rho_RMSE_vs_exposure": spearman,
            "First_to_last_summary": trend,
        })
        records.append(rec)

    return pd.DataFrame(records)


# =============================================================================
# 5. SCIENTIFIC VISUALIZATIONS
# =============================================================================

def plot_charge_uptake_validation(agg: pd.DataFrame, outdir: Path, cfg: ChargeDiagnosticsConfig):
    """Experimental points and model lines vs exposure for every charge state."""
    needed = {"State", "Sequence", "Exposure", "z", "Uptake", "Uptake_ESI"}
    if not needed <= set(agg.columns):
        return

    for (state, seq), grp in agg.groupby(["State", "Sequence"], dropna=False):
        valid = grp.dropna(subset=["Exposure", "z", "Uptake", "Uptake_ESI"])
        if valid.empty:
            continue

        colors = charge_colors(valid["z"])
        fig, ax = plt.subplots(figsize=(9.2, 6.0))

        for z in sorted(valid["z"].unique()):
            g = valid[valid["z"].eq(z)].sort_values("Exposure")
            color = colors[float(z)]
            ax.plot(
                g["Exposure"], g["Uptake_ESI"], marker="D", markersize=5,
                linewidth=2.0, color=color, label=f"Model z={z:g}"
            )
            ax.scatter(
                g["Exposure"], g["Uptake"], s=58, facecolors="none",
                edgecolors=color, linewidths=1.6, label=f"Experiment z={z:g}"
            )

        if (valid["Exposure"] > 0).all():
            ax.set_xscale("log")
        ax.set_xlabel("Exposure time (ms)")
        ax.set_ylabel("Deuterium uptake (Da)")
        ax.set_title(f"Charge-state uptake validation | {seq} | {state}")
        style_axis(ax)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, ncol=2)
        save_figure(fig, outdir / safe_name(state) / f"{safe_name(seq)}.{cfg.image_format}", cfg)


def plot_stage_loss_vs_charge(stage: pd.DataFrame, outdir: Path, cfg: ChargeDiagnosticsConfig):
    loss_cols = [c for c in [
        "Loss_Quench_Da", "Loss_LC_Da", "Loss_ESI_Da", "Loss_Total_PostLabel_Da"
    ] if c in stage.columns]
    if not loss_cols:
        return

    labels = {
        "Loss_Quench_Da": "Quench loss",
        "Loss_LC_Da": "LC loss",
        "Loss_ESI_Da": "ESI loss",
        "Loss_Total_PostLabel_Da": "Total post-label loss",
    }
    markers = ["o", "s", "^", "D"]

    for (state, seq), grp in stage.groupby(["State", "Sequence"], dropna=False):
        # Require at least two charge states at an exposure for a charge-effect plot.
        exposure_counts = grp.groupby("Exposure")["z"].nunique()
        exposures = exposure_counts[exposure_counts >= 2].index.to_numpy(dtype=float)
        if len(exposures) == 0:
            continue
        if cfg.exposure_plot_mode == "representative":
            exposures = representative_exposures(exposures)
        else:
            exposures = sorted(float(x) for x in exposures)

        for exposure in exposures:
            g = grp[grp["Exposure"].eq(exposure)].sort_values("z")
            if g["z"].nunique() < 2:
                continue

            fig, ax = plt.subplots(figsize=(8.2, 5.5))
            for i, col in enumerate(loss_cols):
                ax.plot(
                    g["z"], g[col], marker=markers[i % len(markers)],
                    linewidth=1.8, markersize=6, label=labels[col]
                )
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            ax.set_xticks(sorted(g["z"].dropna().unique()))
            ax.set_xlabel("Charge state z")
            ax.set_ylabel("Deuterium loss (Da)")
            ax.set_title(f"Stage-specific deuterium loss vs charge | {seq} | {state} | {exposure:g} ms")
            style_axis(ax)
            ax.legend(fontsize=8)
            save_figure(
                fig,
                outdir / safe_name(state) / safe_name(seq) / f"{exposure:g}ms.{cfg.image_format}",
                cfg,
            )


def plot_charge_differential_vs_exposure(diff: pd.DataFrame, outdir: Path, cfg: ChargeDiagnosticsConfig):
    needed = {
        "State", "Sequence", "Exposure", "z", "z_reference",
        "Delta_Uptake_vs_lowest_z", "Delta_Uptake_ESI_vs_lowest_z",
    }
    if diff.empty or not needed <= set(diff.columns):
        return

    for (state, seq), grp in diff.groupby(["State", "Sequence"], dropna=False):
        if grp.empty:
            continue
        colors = charge_colors(grp["z"])
        fig, ax = plt.subplots(figsize=(9.0, 5.8))

        for z in sorted(grp["z"].unique()):
            g = grp[grp["z"].eq(z)].sort_values("Exposure")
            ref = g["z_reference"].dropna().iloc[0]
            color = colors[float(z)]
            ax.plot(
                g["Exposure"], g["Delta_Uptake_ESI_vs_lowest_z"],
                marker="D", linewidth=2.0, color=color,
                label=f"Model z={z:g} − z={ref:g}"
            )
            ax.scatter(
                g["Exposure"], g["Delta_Uptake_vs_lowest_z"],
                s=55, facecolors="none", edgecolors=color, linewidths=1.5,
                label=f"Experiment z={z:g} − z={ref:g}"
            )

        if (grp["Exposure"] > 0).all():
            ax.set_xscale("log")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.set_xlabel("Exposure time (ms)")
        ax.set_ylabel("Uptake difference relative to lowest z (Da)")
        ax.set_title(f"Charge-state differential uptake | {seq} | {state}")
        style_axis(ax)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, ncol=2)
        save_figure(fig, outdir / safe_name(state) / f"{safe_name(seq)}.{cfg.image_format}", cfg)


def plot_error_vs_exposure(exposure_metrics: pd.DataFrame, outdir: Path, cfg: ChargeDiagnosticsConfig):
    if exposure_metrics.empty:
        return

    specs = [
        ("RMSE_Da", "RMSE (Da)"),
        ("MAE_Da", "MAE (Da)"),
        ("Bias_Da", "Mean signed error: model − experiment (Da)"),
    ]

    for (state, seq), grp in exposure_metrics.groupby(["State", "Sequence"], dropna=False):
        colors = charge_colors(grp["z"])
        for metric, ylabel in specs:
            if metric not in grp.columns:
                continue
            fig, ax = plt.subplots(figsize=(8.8, 5.6))
            for z in sorted(grp["z"].dropna().unique()):
                g = grp[grp["z"].eq(z)].dropna(subset=["Exposure", metric]).sort_values("Exposure")
                if g.empty:
                    continue
                ax.plot(
                    g["Exposure"], g[metric], marker="o", linewidth=1.8,
                    color=colors[float(z)], label=f"z={z:g}"
                )
            if (pd.to_numeric(grp["Exposure"], errors="coerce").dropna() > 0).all():
                ax.set_xscale("log")
            if metric == "Bias_Da":
                ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
            ax.set_xlabel("Exposure time (ms)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs exposure by charge state | {seq} | {state}")
            style_axis(ax)
            ax.legend(fontsize=8, title="Charge")
            save_figure(
                fig,
                outdir / metric / safe_name(state) / f"{safe_name(seq)}.{cfg.image_format}",
                cfg,
            )


def _plot_heatmap(pivot: pd.DataFrame, title: str, colorbar_label: str, path: Path, cfg: ChargeDiagnosticsConfig):
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8.0, 0.72 * len(pivot.columns) + 3.0), max(4.5, 0.7 * len(pivot.index) + 2.2)))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{float(x):g}" for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"z={float(x):g}" for x in pivot.index])
    ax.set_xlabel("Exposure time (ms)")
    ax.set_ylabel("Charge state")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=colorbar_label)

    # Numeric annotations are useful for this small z x exposure matrix.
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)

    save_figure(fig, path, cfg)


def plot_charge_exposure_heatmaps(agg: pd.DataFrame, outdir: Path, cfg: ChargeDiagnosticsConfig):
    specs = [
        ("Absolute_Error_Da", "Mean absolute model error (Da)"),
        ("Residual_Model_minus_Experimental", "Signed model error (Da)"),
        ("Loss_ESI_Da", "ESI deuterium loss (Da)"),
        ("Loss_Total_PostLabel_Da", "Total post-label deuterium loss (Da)"),
    ]

    for (state, seq), grp in agg.groupby(["State", "Sequence"], dropna=False):
        if grp["z"].nunique() < 2:
            continue
        for col, label in specs:
            if col not in grp.columns:
                continue
            pivot = grp.pivot_table(index="z", columns="Exposure", values=col, aggfunc="mean")
            pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
            _plot_heatmap(
                pivot,
                title=f"{label}: charge × exposure | {seq} | {state}",
                colorbar_label=label,
                path=outdir / col / safe_name(state) / f"{safe_name(seq)}.{cfg.image_format}",
                cfg=cfg,
            )


def plot_charge_separation(sep: pd.DataFrame, outdir: Path, cfg: ChargeDiagnosticsConfig):
    if sep.empty:
        return

    for (state, seq), grp in sep.groupby(["State", "Sequence"], dropna=False):
        g = grp.dropna(subset=["Exposure", "ChargeSeparation_Experimental_Da", "ChargeSeparation_Model_Da"]).sort_values("Exposure")
        if g.empty:
            continue

        fig, ax = plt.subplots(figsize=(8.8, 5.5))
        ax.plot(g["Exposure"], g["ChargeSeparation_Model_Da"], marker="D", linewidth=2.0, label="Model")
        ax.scatter(g["Exposure"], g["ChargeSeparation_Experimental_Da"], s=60, facecolors="none", edgecolors="black", linewidths=1.4, label="Experiment")
        if (g["Exposure"] > 0).all():
            ax.set_xscale("log")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        zlo = g["z_low"].iloc[0]
        zhi = g["z_high"].iloc[0]
        ax.set_xlabel("Exposure time (ms)")
        ax.set_ylabel(f"Uptake(z={zhi:g}) − Uptake(z={zlo:g}) (Da)")
        ax.set_title(f"High-vs-low charge separation | {seq} | {state}")
        style_axis(ax)
        ax.legend()
        save_figure(fig, outdir / "vs_exposure" / safe_name(state) / f"{safe_name(seq)}.{cfg.image_format}", cfg)

    valid = sep.dropna(subset=["ChargeSeparation_Experimental_Da", "ChargeSeparation_Model_Da"])
    if not valid.empty:
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        for state in sorted(valid["State"].astype(str).unique(), key=state_sort_key):
            g = valid[valid["State"].astype(str).eq(state)]
            ax.scatter(g["ChargeSeparation_Experimental_Da"], g["ChargeSeparation_Model_Da"], s=35, alpha=0.72, label=state)
        lo, hi = common_limits(valid["ChargeSeparation_Experimental_Da"], valid["ChargeSeparation_Model_Da"])
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, color="black")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Experimental high-z − low-z separation (Da)")
        ax.set_ylabel("Model high-z − low-z separation (Da)")
        ax.set_title("Charge-state separation parity")
        style_axis(ax)
        ax.legend(fontsize=7)
        save_figure(fig, outdir / f"charge_separation_parity.{cfg.image_format}", cfg)


def plot_charge_effect_slopes(slopes: pd.DataFrame, outdir: Path, cfg: ChargeDiagnosticsConfig):
    if slopes.empty:
        return

    for (state, seq), grp in slopes.groupby(["State", "Sequence"], dropna=False):
        g = grp.dropna(subset=[
            "Exposure", "Experimental_dUptake_dz_Da_per_charge", "Model_dUptake_dz_Da_per_charge"
        ]).sort_values("Exposure")
        if g.empty:
            continue

        fig, ax = plt.subplots(figsize=(8.8, 5.5))
        ax.plot(g["Exposure"], g["Model_dUptake_dz_Da_per_charge"], marker="D", linewidth=2.0, label="Model")
        ax.scatter(g["Exposure"], g["Experimental_dUptake_dz_Da_per_charge"], s=60, facecolors="none", edgecolors="black", linewidths=1.4, label="Experiment")
        if (g["Exposure"] > 0).all():
            ax.set_xscale("log")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.set_xlabel("Exposure time (ms)")
        ax.set_ylabel("d(Uptake)/dz (Da per charge)")
        ax.set_title(f"Quantified charge-state effect on uptake | {seq} | {state}")
        style_axis(ax)
        ax.legend()
        save_figure(fig, outdir / safe_name(state) / f"{safe_name(seq)}.{cfg.image_format}", cfg)


def _baseline_metrics_for_ion(metrics: pd.DataFrame, baseline_state: str, seq: str, z: float) -> pd.DataFrame:
    b = metrics[
        metrics["State"].astype(str).eq(str(baseline_state))
        & metrics["Sequence"].astype(str).eq(str(seq))
        & pd.to_numeric(metrics["z"], errors="coerce").eq(float(z))
    ].copy()
    return b


def plot_error_vs_salt(charge_metrics: pd.DataFrame, outdir: Path, cfg: ChargeDiagnosticsConfig):
    if charge_metrics.empty:
        return

    specs = [("RMSE_Da", "RMSE (Da)"), ("MAE_Da", "MAE (Da)"), ("Bias_Da", "Bias: model − experiment (Da)")]

    for seq, seq_df in charge_metrics.groupby("Sequence", dropna=False):
        for ion in ["NaCl", "CsCl"]:
            ion_df = seq_df[seq_df["Salt_type"].eq(ion)].copy()
            if ion_df.empty:
                continue

            z_values = sorted(seq_df["z"].dropna().unique())
            colors = charge_colors(z_values)

            for metric, ylabel in specs:
                fig, ax = plt.subplots(figsize=(8.2, 5.4))
                drawn = False
                for z in z_values:
                    g = ion_df[pd.to_numeric(ion_df["z"], errors="coerce").eq(float(z))].copy()
                    base = _baseline_metrics_for_ion(charge_metrics, cfg.baseline_state, str(seq), float(z))
                    if not base.empty:
                        g = pd.concat([base, g], ignore_index=True)
                    g = g.dropna(subset=["Salt_mM", metric]).sort_values("Salt_mM")
                    if g.empty:
                        continue
                    drawn = True
                    ax.plot(g["Salt_mM"], g[metric], marker="o", linewidth=1.8, color=colors[float(z)], label=f"z={z:g}")

                if not drawn:
                    plt.close(fig)
                    continue
                if metric == "Bias_Da":
                    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
                ax.set_xlabel(f"{ion} concentration (mM)")
                ax.set_ylabel(ylabel)
                ax.set_title(f"Charge-state-specific {ylabel} vs {ion} | {seq}")
                style_axis(ax)
                ax.legend(title="Charge", fontsize=8)
                save_figure(fig, outdir / metric / ion / f"{safe_name(seq)}.{cfg.image_format}", cfg)


def plot_esi_loss_vs_salt(agg: pd.DataFrame, outdir: Path, cfg: ChargeDiagnosticsConfig):
    if "Loss_ESI_Da" not in agg.columns:
        return

    # Average over exposure here deliberately: this plot asks how the overall ESI
    # loss experienced by each charge state varies across salt concentration.
    group_cols = [c for c in ["State", "Salt_mM", "Salt_type", "Protein", "Sequence", "z"] if c in agg.columns]
    summary = (
        agg[group_cols + ["Loss_ESI_Da"]]
        .groupby(group_cols, dropna=False, as_index=False)
        .agg(Mean_ESI_Loss_Da=("Loss_ESI_Da", "mean"), SD_ESI_Loss_Da=("Loss_ESI_Da", "std"), N_exposures=("Loss_ESI_Da", "count"))
    )

    base_all = summary[summary["State"].astype(str).eq(str(cfg.baseline_state))]

    for seq, seq_df in summary.groupby("Sequence", dropna=False):
        z_values = sorted(seq_df["z"].dropna().unique())
        colors = charge_colors(z_values)
        for ion in ["NaCl", "CsCl"]:
            fig, ax = plt.subplots(figsize=(8.2, 5.4))
            drawn = False
            for z in z_values:
                g = seq_df[
                    seq_df["Salt_type"].eq(ion)
                    & pd.to_numeric(seq_df["z"], errors="coerce").eq(float(z))
                ].copy()
                base = base_all[
                    base_all["Sequence"].astype(str).eq(str(seq))
                    & pd.to_numeric(base_all["z"], errors="coerce").eq(float(z))
                ].copy()
                if not base.empty:
                    g = pd.concat([base, g], ignore_index=True)
                g = g.dropna(subset=["Salt_mM", "Mean_ESI_Loss_Da"]).sort_values("Salt_mM")
                if g.empty:
                    continue
                drawn = True
                ax.errorbar(
                    g["Salt_mM"], g["Mean_ESI_Loss_Da"], yerr=g["SD_ESI_Loss_Da"],
                    marker="o", linewidth=1.7, capsize=3, color=colors[float(z)], label=f"z={z:g}"
                )
            if not drawn:
                plt.close(fig)
                continue
            ax.set_xlabel(f"{ion} concentration (mM)")
            ax.set_ylabel("Mean ESI deuterium loss across exposures (Da)")
            ax.set_title(f"Charge-dependent ESI loss vs {ion} | {seq}")
            style_axis(ax)
            ax.legend(title="Charge", fontsize=8)
            save_figure(fig, outdir / ion / f"{safe_name(seq)}.{cfg.image_format}", cfg)

    return summary


# =============================================================================
# 6. README / METRIC DEFINITIONS
# =============================================================================

def metric_definitions_table() -> pd.DataFrame:
    return pd.DataFrame([
        ("Residual_Model_minus_Experimental", "Uptake_ESI - Uptake", "Da", "Positive means model overpredicts experimental uptake."),
        ("Absolute_Error_Da", "abs(Uptake_ESI - Uptake)", "Da", "Unsigned pointwise prediction error."),
        ("RMSE_Da", "sqrt(mean((Uptake_ESI-Uptake)^2))", "Da", "Penalizes larger errors more strongly than MAE."),
        ("MAE_Da", "mean(abs(Uptake_ESI-Uptake))", "Da", "Average absolute prediction error."),
        ("Bias_Da", "mean(Uptake_ESI-Uptake)", "Da", "Signed model bias; positive=overprediction, negative=underprediction."),
        ("NRMSE_max", "RMSE_Da / max(abs(experimental uptake))", "fraction", "Same normalization convention used by the saved model metrics."),
        ("R2", "1-SSE/SST", "dimensionless", "May be negative when predictions are worse than using the experimental mean."),
        ("Loss_Quench_Da", "Uptake_Labeling - Uptake_Quenched", "Da", "Deuterium lost during quench stage."),
        ("Loss_LC_Da", "Uptake_Quenched - Uptake_LC", "Da", "Deuterium lost during LC stage."),
        ("Loss_ESI_Da", "Uptake_LC - Uptake_ESI", "Da", "Model-predicted deuterium loss during ESI stage."),
        ("Loss_Total_PostLabel_Da", "Uptake_Labeling - Uptake_ESI", "Da", "Total model-predicted loss after labeling."),
        ("ChargeSeparation_Experimental_Da", "Uptake(z_high)-Uptake(z_low)", "Da", "Observed high-vs-low charge-state separation."),
        ("ChargeSeparation_Model_Da", "Uptake_ESI(z_high)-Uptake_ESI(z_low)", "Da", "Predicted high-vs-low charge-state separation."),
        ("Experimental_dUptake_dz_Da_per_charge", "linear slope of experimental uptake vs z", "Da/charge", "Negative means uptake decreases as charge state increases."),
        ("Model_dUptake_dz_Da_per_charge", "linear slope of model uptake vs z", "Da/charge", "Model counterpart of experimental charge-state slope."),
    ], columns=["Metric", "Definition", "Unit", "Interpretation"])


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    cfg = ChargeDiagnosticsConfig()
    dirs = make_output_dirs(cfg)

    print("=" * 94)
    print("HDMX CHARGE-STATE SCIENTIFIC DIAGNOSTICS FROM SAVED RESULTS")
    print("=" * 94)
    print(f"Reading saved results: {Path(cfg.results_root).resolve()}")
    print("No HDMX model or optimizer will be rerun.\n")

    raw = load_saved_rows(cfg)
    raw_diag = add_error_and_loss_columns(raw)
    agg = aggregate_for_visualization(raw_diag)

    # Metrics are calculated from the original saved observation rows, not from
    # the replicate-averaged plotting table.
    state_peptide_charge_metrics = metrics_by_group(
        raw_diag,
        [c for c in ["State", "Salt_mM", "Salt_type", "Protein", "Sequence", "z"] if c in raw_diag.columns],
    )
    state_peptide_charge_exposure_metrics = metrics_by_group(
        raw_diag,
        [c for c in ["State", "Salt_mM", "Salt_type", "Protein", "Sequence", "z", "Exposure"] if c in raw_diag.columns],
    )
    state_peptide_exposure_metrics = metrics_by_group(
        raw_diag,
        [c for c in ["State", "Salt_mM", "Salt_type", "Protein", "Sequence", "Exposure"] if c in raw_diag.columns],
    )

    stage = build_stage_loss_summary(agg)
    separation = build_charge_separation(agg)
    differential = build_charge_differentials(agg)
    slopes = build_charge_effect_slopes(agg)
    exposure_trends = build_exposure_error_trends(state_peptide_charge_exposure_metrics, cfg)

    # -------------------------------------------------------------------------
    # Save core numerical tables first.
    # -------------------------------------------------------------------------
    tables = {
        "rowwise_charge_errors_and_losses.csv": raw_diag,
        "charge_state_fit_metrics.csv": state_peptide_charge_metrics,
        "charge_state_exposure_fit_metrics.csv": state_peptide_charge_exposure_metrics,
        "peptide_exposure_fit_metrics_all_charges.csv": state_peptide_exposure_metrics,
        "stage_loss_by_charge_and_exposure.csv": stage,
        "charge_separation_by_exposure.csv": separation,
        "charge_differential_vs_lowest_z.csv": differential,
        "charge_effect_slope_by_exposure.csv": slopes,
        "exposure_error_trend_by_charge.csv": exposure_trends,
        "metric_definitions.csv": metric_definitions_table(),
    }
    for filename, df in tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(dirs["tables"] / filename, index=False)

    print("Creating charge-state experimental/model validation plots...")
    plot_charge_uptake_validation(agg, dirs["uptake"], cfg)

    print("Creating stage-specific deuterium-loss vs charge plots...")
    plot_stage_loss_vs_charge(stage, dirs["loss"], cfg)

    print("Creating charge-differential uptake plots...")
    plot_charge_differential_vs_exposure(differential, dirs["differential"], cfg)

    print("Creating charge-specific error-vs-exposure plots...")
    plot_error_vs_exposure(state_peptide_charge_exposure_metrics, dirs["error_exposure"], cfg)

    print("Creating charge x exposure heatmaps...")
    plot_charge_exposure_heatmaps(agg, dirs["heatmaps"], cfg)

    print("Creating charge-separation diagnostics...")
    plot_charge_separation(separation, dirs["separation"], cfg)

    print("Creating quantitative charge-effect slope plots...")
    plot_charge_effect_slopes(slopes, dirs["slope"], cfg)

    print("Creating charge-specific error vs salt-concentration plots...")
    plot_error_vs_salt(state_peptide_charge_metrics, dirs["salt_error"], cfg)

    print("Creating charge-specific ESI-loss vs salt-concentration plots...")
    esi_salt_summary = plot_esi_loss_vs_salt(agg, dirs["salt_loss"], cfg)
    if isinstance(esi_salt_summary, pd.DataFrame) and not esi_salt_summary.empty:
        esi_salt_summary.to_csv(dirs["tables"] / "esi_loss_summary_by_state_peptide_charge.csv", index=False)

    readme = f"""
HDMX CHARGE-STATE SCIENTIFIC DIAGNOSTICS
========================================

SOURCE
------
{Path(cfg.results_root).resolve()}

This analysis reads saved HDMX results only. No optimization or forward model is rerun.

OUTPUT FOLDERS
--------------
00_tables/
    Row-level prediction errors and stage losses; charge-specific RMSE/MAE/bias;
    exposure-specific metrics; charge separation; differential uptake; and
    quantitative uptake-vs-charge slopes.

01_charge_uptake_validation/
    Experimental uptake points and final ESI model curves vs exposure for all z.

02_stage_loss_vs_charge/
    Quench, LC, ESI, and total post-label deuterium loss vs z at selected exposures.
    Current exposure_plot_mode = {cfg.exposure_plot_mode!r}.

03_charge_differential_vs_exposure/
    Experimental and model uptake differences relative to the lowest charge state.

04_error_vs_exposure/
    RMSE, MAE and signed bias vs exposure separately for each charge state.

05_charge_exposure_heatmaps/
    Charge x exposure heatmaps for absolute error, signed error, ESI loss and total loss.

06_charge_separation/
    High-z minus low-z uptake separation vs exposure, plus experiment-vs-model parity.

07_charge_effect_slope/
    d(Uptake)/dz vs exposure for experiment and model. Negative slopes mean uptake
    decreases as charge state increases; positive slopes mean uptake increases.

08_error_vs_salt_concentration/
    RMSE, MAE and bias vs salt concentration, separately for NaCl and CsCl and by z.
    0mM is included as a shared baseline when available.

09_esi_loss_vs_salt_concentration/
    Mean model-predicted ESI deuterium loss vs salt concentration for each charge state.
    Error bars show SD across exposure times.

IMPORTANT STATISTICAL NOTE
--------------------------
At a peptide/state/charge/exposure combination with only one saved observation,
RMSE_Da equals the absolute error for that point. Always inspect N in
charge_state_exposure_fit_metrics.csv when interpreting exposure-specific RMSE.

The exposure-error trend table does not assume that error must decrease with exposure.
It reports first-vs-last RMSE, the RMSE slope versus log10(exposure), and a Spearman
rank correlation so the observed direction is visible from the saved data.
""".strip() + "\n"
    (dirs["root"] / "README.txt").write_text(readme, encoding="utf-8")

    print("\n" + "=" * 94)
    print("CHARGE-STATE DIAGNOSTICS FINISHED")
    print(f"Output: {dirs['root'].resolve()}")
    print("Recommended first outputs:")
    print("  - 00_tables/charge_state_fit_metrics.csv")
    print("  - 00_tables/exposure_error_trend_by_charge.csv")
    print("  - 03_charge_differential_vs_exposure/")
    print("  - 05_charge_exposure_heatmaps/")
    print("  - 06_charge_separation/")
    print("  - 07_charge_effect_slope/")
    print("=" * 94)


if __name__ == "__main__":
    main()

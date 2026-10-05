#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDMX downstream analysis of SAVED all-salt results.

This script is intentionally independent of the HDMX fitting/model code.
It reads the files written by hdmx_all_salt_organized_v2.py and creates
cross-salt tables, diagnostics and figures without rerunning optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import shutil
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
    analysis_root: str = "hdmx_downstream_analysis"
    baseline_state: str = "0mM"
    dpi: int = 300
    image_format: str = "png"
    show_plots: bool = False

    # "representative": first/middle/last common exposure per peptide/charge
    # "all": every matched peptide/exposure/charge combination in >= 2 states
    # "off": no residue/vector plots
    vector_plot_mode: str = "representative"

    # A fitted value is considered near a bound if it lies within this fraction
    # of the full allowed interval.
    bound_fraction_tolerance: float = 0.005


VECTOR_COLUMNS = [
    "Uptake_Vector",
    "Uptake_Vector_Quenched",
    "Uptake_Vector_LC",
    "Uptake_Vector_ESI",
    "k_int_label_vector",
    "K_eq_label_vector",
    "k_int_quench_vector",
    "K_eq_quench_vector",
    "LC_Integral_Vector",
    "k_LC_at_RT_vector",
    "Q_ESI_Vector",
    "k_ESI_vector",
]


# =============================================================================
# 2. HELPERS
# =============================================================================

def safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text or "unnamed"


def parse_state(state: str) -> tuple[float, str]:
    """Return numeric salt concentration in mM and salt identity."""
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


def make_output_dirs(cfg: AnalysisConfig) -> dict[str, Path]:
    root = Path(cfg.analysis_root)
    dirs = {
        "root": root,
        "tables": root / "00_tables",
        "experimental": root / "01_experimental_salt_effects",
        "validation": root / "02_model_validation",
        "stage": root / "03_stage_losses",
        "parameters": root / "04_optimization_parameters",
        "rates": root / "05_rate_consistency",
        "vectors": root / "06_residue_diagnostics",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def reset_generated_directory(path: Path) -> None:
    """Remove stale generated files from a section that this version replaces."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig, path: Path, cfg: AnalysisConfig):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)


def vector_from_value(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float)
    if value is None:
        return np.array([], dtype=float)
    if isinstance(value, float) and np.isnan(value):
        return np.array([], dtype=float)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.array([], dtype=float)
        try:
            return np.asarray(json.loads(text), dtype=float)
        except Exception:
            return np.array([], dtype=float)
    return np.array([], dtype=float)


def mean_vector(series: pd.Series) -> np.ndarray:
    arrays = [vector_from_value(v) for v in series]
    arrays = [a for a in arrays if len(a) > 0]
    if not arrays:
        return np.array([], dtype=float)
    n = min(len(a) for a in arrays)
    return np.nanmean(np.vstack([a[:n] for a in arrays]), axis=0)


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
            "or change AnalysisConfig.results_root."
        )

    found = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        manifest = child / "manifest.json"
        pkl = child / "exact_objects" / "dynamic_results.pkl"
        csv = child / "tables" / "row_results_scalar.csv"
        if not (manifest.exists() or pkl.exists() or csv.exists()):
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


def read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_saved_results(cfg: AnalysisConfig) -> dict[str, object]:
    root = Path(cfg.results_root)
    state_dirs = discover_state_dirs(root)
    if not state_dirs:
        raise RuntimeError(f"No state result folders were found in {root.resolve()}")

    row_frames = []
    metric_frames = []
    global_frames = []
    local_frames = []
    rate_frames = []
    settings = {}
    exact_states = []

    for state, state_dir in state_dirs:
        pkl = state_dir / "exact_objects" / "dynamic_results.pkl"
        scalar = state_dir / "tables" / "row_results_scalar.csv"

        if pkl.exists():
            df = pd.read_pickle(pkl)
            exact_states.append(state)
        elif scalar.exists():
            df = pd.read_csv(scalar)
        else:
            continue

        df = df.copy()
        df["State"] = str(state)
        if "Sequence" in df.columns:
            df["Sequence"] = df["Sequence"].astype(str).str.strip()
        row_frames.append(df)

        metrics = read_optional_csv(state_dir / "tables" / "fit_metrics.csv")
        if not metrics.empty:
            metrics["State"] = str(state)
            metric_frames.append(metrics)

        glob = read_optional_csv(state_dir / "optimization" / "global_optimization.csv")
        if glob.empty:
            gp = state_dir / "global_parameters.json"
            if gp.exists():
                with open(gp, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                glob = pd.DataFrame([{
                    "State": str(state),
                    "gamma_lc_opt": d.get("gamma_lc", np.nan),
                    "tau_label_ms_opt": d.get("tau_label_ms", np.nan),
                    "global_gamma_tau_objective": d.get("global_gamma_tau_objective", np.nan),
                }])
        if not glob.empty:
            glob["State"] = str(state)
            global_frames.append(glob)

        local = read_optional_csv(state_dir / "optimization" / "all_optimized_parameters.csv")
        if local.empty:
            local = read_optional_csv(state_dir / "tables" / "peptide_parameters.csv")
            if not local.empty:
                local = local.rename(columns={
                    "gamma_prime_p": "gamma_prime_p_opt",
                    "t_ESI": "t_ESI_opt",
                    "lambda_ESI": "lambda_ESI_opt",
                })
        if not local.empty:
            local["State"] = str(state)
            if "Sequence" in local.columns:
                local["Sequence"] = local["Sequence"].astype(str).str.strip()
            local_frames.append(local)

        rates = read_optional_csv(state_dir / "tables" / "intrinsic_rate_vectors.csv")
        if not rates.empty:
            rates["State"] = str(state)
            rate_frames.append(rates)

        settings_path = state_dir / "optimization" / "optimization_settings.json"
        if settings_path.exists():
            with open(settings_path, "r", encoding="utf-8") as fh:
                settings[str(state)] = json.load(fh)

    rows = add_state_metadata(pd.concat(row_frames, ignore_index=True))
    metrics = add_state_metadata(pd.concat(metric_frames, ignore_index=True)) if metric_frames else pd.DataFrame()
    global_opt = add_state_metadata(pd.concat(global_frames, ignore_index=True)) if global_frames else pd.DataFrame()
    local_opt = add_state_metadata(pd.concat(local_frames, ignore_index=True)) if local_frames else pd.DataFrame()
    rates = add_state_metadata(pd.concat(rate_frames, ignore_index=True)) if rate_frames else pd.DataFrame()

    return {
        "rows": rows,
        "metrics": metrics,
        "global_opt": global_opt,
        "local_opt": local_opt,
        "rates": rates,
        "settings": settings,
        "states": [x[0] for x in state_dirs],
        "exact_states": exact_states,
    }


# =============================================================================
# 4. NUMERICAL COMPARISON TABLES
# =============================================================================

def aggregate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in [
        "State", "Salt_mM", "Salt_type", "Protein", "Sequence", "Exposure", "z"
    ] if c in rows.columns]
    value_cols = [c for c in [
        "Uptake", "Uptake_Labeling", "Uptake_Quenched", "Uptake_LC", "Uptake_ESI", "RT"
    ] if c in rows.columns]
    return (
        rows[group_cols + value_cols]
        .groupby(group_cols, dropna=False, as_index=False)
        .mean(numeric_only=True)
    )


def baseline_deltas(agg: pd.DataFrame, baseline_state: str) -> pd.DataFrame:
    baseline = agg[agg["State"].astype(str).eq(str(baseline_state))].copy()
    other = agg[~agg["State"].astype(str).eq(str(baseline_state))].copy()
    keys = [c for c in ["Protein", "Sequence", "Exposure", "z"] if c in agg.columns]
    measures = [c for c in [
        "Uptake", "Uptake_Labeling", "Uptake_Quenched", "Uptake_LC", "Uptake_ESI"
    ] if c in agg.columns]
    baseline = baseline[keys + measures].rename(columns={c: f"{c}_0mM" for c in measures})
    merged = other.merge(baseline, on=keys, how="inner")
    for col in measures:
        merged[f"Delta_{col}_vs_0mM"] = merged[col] - merged[f"{col}_0mM"]
    return merged


def ion_differences(agg: pd.DataFrame) -> pd.DataFrame:
    work = agg[agg["Salt_type"].isin(["NaCl", "CsCl"]) & agg["Salt_mM"].gt(0)].copy()
    keys = [c for c in ["Protein", "Sequence", "Exposure", "z", "Salt_mM"] if c in work.columns]
    measures = [c for c in [
        "Uptake", "Uptake_Labeling", "Uptake_Quenched", "Uptake_LC", "Uptake_ESI"
    ] if c in work.columns]
    nacl = work[work["Salt_type"].eq("NaCl")][keys + measures].rename(columns={c: f"{c}_NaCl" for c in measures})
    cscl = work[work["Salt_type"].eq("CsCl")][keys + measures].rename(columns={c: f"{c}_CsCl" for c in measures})
    merged = cscl.merge(nacl, on=keys, how="inner")
    for col in measures:
        merged[f"Delta_{col}_CsCl_minus_NaCl"] = merged[f"{col}_CsCl"] - merged[f"{col}_NaCl"]
    return merged


def stage_losses(agg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = agg.copy()
    if {"Uptake_Labeling", "Uptake_Quenched"} <= set(out.columns):
        out["Loss_Quench"] = out["Uptake_Labeling"] - out["Uptake_Quenched"]
    if {"Uptake_Quenched", "Uptake_LC"} <= set(out.columns):
        out["Loss_LC"] = out["Uptake_Quenched"] - out["Uptake_LC"]
        out["Retention_LC"] = out["Uptake_LC"] / out["Uptake_Quenched"].replace(0, np.nan)
    if {"Uptake_LC", "Uptake_ESI"} <= set(out.columns):
        out["Loss_ESI"] = out["Uptake_LC"] - out["Uptake_ESI"]
        out["Retention_ESI"] = out["Uptake_ESI"] / out["Uptake_LC"].replace(0, np.nan)

    group_cols = [c for c in ["State", "Salt_mM", "Salt_type", "Protein", "Sequence"] if c in out.columns]
    value_cols = [c for c in ["Loss_Quench", "Loss_LC", "Loss_ESI", "Retention_LC", "Retention_ESI"] if c in out.columns]
    summary = (
        out[group_cols + value_cols]
        .groupby(group_cols, dropna=False, as_index=False)
        .mean(numeric_only=True)
    ) if value_cols else pd.DataFrame()
    return out, summary


def residual_table(agg: pd.DataFrame) -> pd.DataFrame:
    out = agg.copy()
    if {"Uptake", "Uptake_ESI"} <= set(out.columns):
        out["Residual_ESI_minus_Experimental"] = out["Uptake_ESI"] - out["Uptake"]
        out["AbsResidual"] = np.abs(out["Residual_ESI_minus_Experimental"])
    return out


def charge_separation(agg: pd.DataFrame) -> pd.DataFrame:
    if "z" not in agg.columns:
        return pd.DataFrame()
    group_cols = [c for c in ["State", "Salt_mM", "Salt_type", "Protein", "Sequence", "Exposure"] if c in agg.columns]
    rows = []
    for key, grp in agg.groupby(group_cols, dropna=False):
        g = grp.dropna(subset=["z"])
        if g["z"].nunique() < 2:
            continue
        zlo, zhi = float(g["z"].min()), float(g["z"].max())
        lo = g[g["z"].eq(zlo)]
        hi = g[g["z"].eq(zhi)]
        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rec["z_low"] = zlo
        rec["z_high"] = zhi
        if "Uptake" in g.columns:
            rec["ChargeSeparation_Experimental"] = float(hi["Uptake"].mean() - lo["Uptake"].mean())
        if "Uptake_ESI" in g.columns:
            rec["ChargeSeparation_Model"] = float(hi["Uptake_ESI"].mean() - lo["Uptake_ESI"].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def global_parameter_deltas(global_opt: pd.DataFrame, baseline_state: str) -> pd.DataFrame:
    if global_opt.empty:
        return pd.DataFrame()
    out = global_opt.copy()
    baseline = out[out["State"].astype(str).eq(str(baseline_state))]
    if baseline.empty:
        return out
    for col in ["gamma_lc_opt", "tau_label_ms_opt", "global_gamma_tau_objective"]:
        if col in out.columns:
            b = pd.to_numeric(baseline[col], errors="coerce").iloc[0]
            out[f"Delta_{col}_vs_0mM"] = pd.to_numeric(out[col], errors="coerce") - b
    return out


def local_parameter_deltas(local_opt: pd.DataFrame, baseline_state: str) -> pd.DataFrame:
    if local_opt.empty:
        return pd.DataFrame()
    params = [c for c in ["gamma_prime_p_opt", "t_ESI_opt", "lambda_ESI_opt", "ESI_objective_at_optimum"] if c in local_opt.columns]
    keys = [c for c in ["Protein", "Sequence"] if c in local_opt.columns]
    baseline = local_opt[local_opt["State"].astype(str).eq(str(baseline_state))][keys + params].copy()
    baseline = baseline.rename(columns={c: f"{c}_0mM" for c in params})
    other = local_opt[~local_opt["State"].astype(str).eq(str(baseline_state))].copy()
    merged = other.merge(baseline, on=keys, how="left")
    for col in params:
        merged[f"Delta_{col}_vs_0mM"] = merged[col] - merged[f"{col}_0mM"]
    return merged


def local_parameter_ion_differences(local_opt: pd.DataFrame) -> pd.DataFrame:
    if local_opt.empty:
        return pd.DataFrame()
    work = local_opt[local_opt["Salt_type"].isin(["NaCl", "CsCl"]) & local_opt["Salt_mM"].gt(0)].copy()
    params = [c for c in ["gamma_prime_p_opt", "t_ESI_opt", "lambda_ESI_opt", "ESI_objective_at_optimum"] if c in work.columns]
    keys = [c for c in ["Protein", "Sequence", "Salt_mM"] if c in work.columns]
    nacl = work[work["Salt_type"].eq("NaCl")][keys + params].rename(columns={c: f"{c}_NaCl" for c in params})
    cscl = work[work["Salt_type"].eq("CsCl")][keys + params].rename(columns={c: f"{c}_CsCl" for c in params})
    merged = cscl.merge(nacl, on=keys, how="inner")
    for col in params:
        merged[f"Delta_{col}_CsCl_minus_NaCl"] = merged[f"{col}_CsCl"] - merged[f"{col}_NaCl"]
    return merged


def linear_fit(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.unique(x).size < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(r2)


def parameter_slopes(global_opt: pd.DataFrame, local_opt: pd.DataFrame, baseline_state: str) -> pd.DataFrame:
    """Simple descriptive slopes vs salt concentration, reported per 100 mM."""
    rows = []

    if not global_opt.empty:
        base = global_opt[global_opt["State"].astype(str).eq(str(baseline_state))]
        for ion in ["NaCl", "CsCl"]:
            subset = global_opt[global_opt["Salt_type"].eq(ion)].copy()
            if not base.empty:
                subset = pd.concat([base, subset], ignore_index=True)
            for p in ["gamma_lc_opt", "tau_label_ms_opt", "global_gamma_tau_objective"]:
                if p not in subset.columns:
                    continue
                slope, intercept, r2 = linear_fit(subset["Salt_mM"], subset[p])
                rows.append({
                    "Level": "Global", "Protein": "", "Sequence": "",
                    "Salt_type": ion, "Parameter": p,
                    "Slope_per_100mM": slope * 100 if np.isfinite(slope) else np.nan,
                    "Intercept": intercept, "R2_linear": r2,
                    "N_points": int(subset[["Salt_mM", p]].dropna().shape[0]),
                })

    if not local_opt.empty:
        base_all = local_opt[local_opt["State"].astype(str).eq(str(baseline_state))]
        params = [c for c in ["gamma_prime_p_opt", "t_ESI_opt", "lambda_ESI_opt", "ESI_objective_at_optimum"] if c in local_opt.columns]
        for seq, seq_df in local_opt.groupby("Sequence"):
            base = base_all[base_all["Sequence"].eq(seq)]
            protein = str(seq_df["Protein"].dropna().iloc[0]) if "Protein" in seq_df.columns and not seq_df["Protein"].dropna().empty else ""
            for ion in ["NaCl", "CsCl"]:
                subset = seq_df[seq_df["Salt_type"].eq(ion)].copy()
                if not base.empty:
                    subset = pd.concat([base, subset], ignore_index=True)
                for p in params:
                    slope, intercept, r2 = linear_fit(subset["Salt_mM"], subset[p])
                    rows.append({
                        "Level": "Peptide", "Protein": protein, "Sequence": seq,
                        "Salt_type": ion, "Parameter": p,
                        "Slope_per_100mM": slope * 100 if np.isfinite(slope) else np.nan,
                        "Intercept": intercept, "R2_linear": r2,
                        "N_points": int(subset[["Salt_mM", p]].dropna().shape[0]),
                    })

    return pd.DataFrame(rows)


def bound_diagnostics(global_opt: pd.DataFrame, local_opt: pd.DataFrame, settings: dict, tol: float) -> pd.DataFrame:
    rows = []

    def add(state, level, protein, sequence, parameter, value, bounds):
        if bounds is None or len(bounds) != 2:
            return
        lo, hi = float(bounds[0]), float(bounds[1])
        val = float(value) if pd.notna(value) else np.nan
        fixed = math.isclose(lo, hi)
        if not np.isfinite(val):
            pos = np.nan
            near_lo = near_hi = False
        elif fixed:
            pos = np.nan
            near_lo = near_hi = math.isclose(val, lo)
        else:
            pos = (val - lo) / (hi - lo)
            near_lo = pos <= tol
            near_hi = pos >= (1.0 - tol)
        rows.append({
            "State": state, "Level": level, "Protein": protein, "Sequence": sequence,
            "Parameter": parameter, "Value": val, "Lower_bound": lo, "Upper_bound": hi,
            "Normalized_bound_position": pos,
            "Near_lower_bound": bool(near_lo), "Near_upper_bound": bool(near_hi),
            "Fixed_by_equal_bounds": bool(fixed),
        })

    for _, r in global_opt.iterrows():
        state = str(r["State"])
        s = settings.get(state, {})
        if "gamma_lc_opt" in r.index:
            add(state, "Global", "", "", "gamma_lc_opt", r["gamma_lc_opt"], s.get("gamma_bounds"))
        if "tau_label_ms_opt" in r.index:
            add(state, "Global", "", "", "tau_label_ms_opt", r["tau_label_ms_opt"], s.get("tau_label_ms_bounds"))

    for _, r in local_opt.iterrows():
        state = str(r["State"])
        s = settings.get(state, {})
        protein, seq = str(r.get("Protein", "")), str(r.get("Sequence", ""))
        if "gamma_prime_p_opt" in r.index:
            add(state, "Peptide", protein, seq, "gamma_prime_p_opt", r["gamma_prime_p_opt"], s.get("gamma_prime_p_bounds"))
        if "t_ESI_opt" in r.index:
            add(state, "Peptide", protein, seq, "t_ESI_opt", r["t_ESI_opt"], s.get("t_esi_bounds"))
        if "lambda_ESI_opt" in r.index:
            add(state, "Peptide", protein, seq, "lambda_ESI_opt", r["lambda_ESI_opt"], s.get("lambda_esi_bounds"))

    return pd.DataFrame(rows)


def parameter_effect_associations(local_opt: pd.DataFrame, stage_summary: pd.DataFrame, metrics: pd.DataFrame):
    if local_opt.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = [c for c in ["State", "Protein", "Sequence"] if c in local_opt.columns]
    joined = local_opt.copy()

    if not stage_summary.empty:
        cols = [c for c in ["Loss_LC", "Loss_ESI", "Retention_LC", "Retention_ESI"] if c in stage_summary.columns]
        joined = joined.merge(stage_summary[keys + cols], on=keys, how="left")
    if not metrics.empty:
        cols = [c for c in ["RMSE_Da", "RMSE_frac"] if c in metrics.columns]
        joined = joined.merge(metrics[keys + cols], on=keys, how="left")

    pairs = [
        ("gamma_prime_p_opt", "Loss_LC"),
        ("gamma_prime_p_opt", "Retention_LC"),
        ("t_ESI_opt", "Loss_ESI"),
        ("t_ESI_opt", "Retention_ESI"),
        ("lambda_ESI_opt", "Loss_ESI"),
        ("lambda_ESI_opt", "Retention_ESI"),
        ("gamma_prime_p_opt", "RMSE_frac"),
        ("t_ESI_opt", "RMSE_frac"),
        ("lambda_ESI_opt", "RMSE_frac"),
        ("t_ESI_opt", "lambda_ESI_opt"),
    ]
    corr_rows = []
    for x, y in pairs:
        if x not in joined.columns or y not in joined.columns:
            continue
        tmp = joined[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        corr_rows.append({
            "X": x, "Y": y,
            "Pearson_r": tmp[x].corr(tmp[y]) if len(tmp) >= 2 else np.nan,
            "N": int(len(tmp)),
            "Interpretation": "Descriptive association only; not causal.",
        })
    return joined, pd.DataFrame(corr_rows)


def rate_consistency(rates: pd.DataFrame, baseline_state: str) -> pd.DataFrame:
    if rates.empty:
        return pd.DataFrame()
    base = rates[rates["State"].astype(str).eq(str(baseline_state))]
    cols = [c for c in ["label_k_int_vector", "quench_k_int_vector", "label_K_eq_vector", "quench_K_eq_vector"] if c in rates.columns]
    rows = []
    for _, r in rates.iterrows():
        seq = str(r["Sequence"])
        b = base[base["Sequence"].astype(str).eq(seq)]
        if b.empty:
            continue
        br = b.iloc[0]
        for col in cols:
            a, ref = vector_from_value(r[col]), vector_from_value(br[col])
            n = min(len(a), len(ref))
            if n:
                diff = a[:n] - ref[:n]
                mx = float(np.nanmax(np.abs(diff)))
                rms = float(np.sqrt(np.nanmean(diff ** 2)))
            else:
                mx = rms = np.nan
            rows.append({
                "State": r["State"], "Sequence": seq, "Rate_vector": col,
                "Vector_length_compared": n,
                "Max_abs_difference_vs_0mM": mx,
                "RMS_difference_vs_0mM": rms,
            })
    return pd.DataFrame(rows)


# =============================================================================
# 5. EXPERIMENTAL PLOTS
# =============================================================================

def plot_experimental_kinetics(agg: pd.DataFrame, outdir: Path, cfg: AnalysisConfig):
    needed = {"Sequence", "z", "Exposure", "Uptake", "State"}
    if not needed <= set(agg.columns):
        return
    for (seq, z), grp in agg.groupby(["Sequence", "z"], dropna=False):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for state in sorted(grp["State"].astype(str).unique(), key=state_sort_key):
            g = grp[grp["State"].astype(str).eq(state)].sort_values("Exposure")
            ax.plot(g["Exposure"], g["Uptake"], marker="o", linewidth=1.5, label=state)
        if (pd.to_numeric(grp["Exposure"], errors="coerce") > 0).all():
            ax.set_xscale("log")
        ax.set_xlabel("Exposure time (ms)")
        ax.set_ylabel("Experimental deuterium uptake (Da)")
        ax.set_title(f"Experimental uptake vs exposure | {seq} | z={z:g}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        save_figure(fig, outdir / "kinetics" / f"{safe_name(seq)}_z{safe_name(z)}.{cfg.image_format}", cfg)


def _common_last_exposure_for_peptide_ion(
    peptide_df: pd.DataFrame,
    ion: str,
    baseline_state: str,
) -> tuple[pd.DataFrame, float | None, list[float], list[str]]:
    """
    Select the highest exposure that is present for EVERY common charge state
    across the baseline state and all available concentrations of one salt type.

    This prevents comparisons in which one salt or charge state is evaluated at
    a different exposure time from another.
    """
    if peptide_df.empty:
        return pd.DataFrame(), None, [], []

    baseline = peptide_df[
        peptide_df["State"].astype(str).eq(str(baseline_state))
    ].copy()
    ion_df = peptide_df[peptide_df["Salt_type"].eq(ion)].copy()

    if ion_df.empty:
        return pd.DataFrame(), None, [], []

    relevant = pd.concat([baseline, ion_df], ignore_index=True) if not baseline.empty else ion_df.copy()
    relevant = relevant.dropna(subset=["State", "Exposure", "z", "Uptake"]).copy()
    if relevant.empty:
        return pd.DataFrame(), None, [], []

    states = sorted(relevant["State"].astype(str).unique(), key=state_sort_key)

    # Only compare charge states that exist in every included state.
    charge_sets = []
    for state in states:
        charges = set(
            pd.to_numeric(
                relevant[relevant["State"].astype(str).eq(state)]["z"],
                errors="coerce",
            ).dropna().astype(float)
        )
        if charges:
            charge_sets.append(charges)

    if not charge_sets:
        return pd.DataFrame(), None, [], states

    common_charges = sorted(set.intersection(*charge_sets))
    if not common_charges:
        warnings.warn(
            f"No charge state is shared across all {ion} states for peptide "
            f"{peptide_df['Sequence'].iloc[0]!r}; summary plot skipped."
        )
        return pd.DataFrame(), None, [], states

    # Find exposure times shared by every state x common-charge combination.
    exposure_sets = []
    for state in states:
        for z in common_charges:
            g = relevant[
                relevant["State"].astype(str).eq(state)
                & pd.to_numeric(relevant["z"], errors="coerce").eq(float(z))
            ]
            exposures = {
                round(float(x), 6)
                for x in pd.to_numeric(g["Exposure"], errors="coerce").dropna()
            }
            if not exposures:
                return pd.DataFrame(), None, common_charges, states
            exposure_sets.append(exposures)

    common_exposures = set.intersection(*exposure_sets) if exposure_sets else set()
    if not common_exposures:
        warnings.warn(
            f"No common exposure is shared across all {ion} state/charge "
            f"combinations for peptide {peptide_df['Sequence'].iloc[0]!r}; "
            "summary plot skipped rather than mixing exposure times."
        )
        return pd.DataFrame(), None, common_charges, states

    last_exposure = float(max(common_exposures))
    exposure_values = pd.to_numeric(relevant["Exposure"], errors="coerce")
    charge_values = pd.to_numeric(relevant["z"], errors="coerce")

    selected = relevant[
        np.isclose(exposure_values, last_exposure, rtol=0.0, atol=1e-5)
        & charge_values.isin(common_charges)
    ].copy()
    selected["Selected_common_exposure"] = last_exposure
    selected["Ion_family"] = ion

    return selected, last_exposure, common_charges, states


def build_last_common_exposure_tables(
    agg: pd.DataFrame,
    baseline_state: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the exact data used by the two new experimental summary plot families."""
    if agg.empty or "Sequence" not in agg.columns:
        return pd.DataFrame(), pd.DataFrame()

    selected_frames = []
    summary_rows = []

    for seq, seq_df in agg.groupby("Sequence", dropna=False):
        for ion in ["NaCl", "CsCl"]:
            selected, exposure, charges, states = _common_last_exposure_for_peptide_ion(
                seq_df,
                ion,
                baseline_state,
            )
            if selected.empty or exposure is None:
                continue

            selected_frames.append(selected)
            summary_rows.append(
                {
                    "Sequence": seq,
                    "Salt_type": ion,
                    "Selected_common_exposure_ms": exposure,
                    "Common_charge_states": ",".join(f"{z:g}" for z in charges),
                    "States_included": ",".join(states),
                }
            )

    points = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return points, summary


def plot_last_exposure_vs_salt(
    selected: pd.DataFrame,
    outdir: Path,
    cfg: AnalysisConfig,
):
    """
    For each peptide and salt family, plot raw experimental uptake at the highest
    common exposure versus salt concentration, with one line per charge state.
    """
    needed = {"Sequence", "Ion_family", "Salt_mM", "z", "Uptake", "Selected_common_exposure"}
    if selected.empty or not needed <= set(selected.columns):
        return

    for (seq, ion), grp in selected.groupby(["Sequence", "Ion_family"], dropna=False):
        exposure = float(grp["Selected_common_exposure"].iloc[0])
        fig, ax = plt.subplots(figsize=(8.2, 5.5))

        for z, g in grp.groupby("z", dropna=False):
            g = g.sort_values("Salt_mM")
            ax.plot(
                g["Salt_mM"],
                g["Uptake"],
                marker="o",
                linewidth=1.7,
                markersize=6,
                label=f"z={float(z):g}",
            )

        concentrations = sorted(pd.to_numeric(grp["Salt_mM"], errors="coerce").dropna().unique())
        if concentrations:
            ax.set_xticks(concentrations)
        ax.set_xlabel(f"{ion} concentration (mM)")
        ax.set_ylabel("Experimental deuterium uptake (Da)")
        ax.set_title(
            f"Experimental uptake vs {ion} concentration | {seq}\n"
            f"highest common exposure = {exposure:g} ms"
        )
        ax.grid(True, alpha=0.25)
        ax.legend(title="Charge state")

        save_figure(
            fig,
            outdir / "last_exposure_vs_salt" /
            f"{safe_name(seq)}_{ion}_last_exposure.{cfg.image_format}",
            cfg,
        )


def plot_charge_state_effect(
    selected: pd.DataFrame,
    outdir: Path,
    cfg: AnalysisConfig,
):
    """
    For each peptide and salt family, put charge state on the x-axis so the
    charge-state dependence can be read directly. Each line is one concentration.
    """
    needed = {"Sequence", "Ion_family", "State", "Salt_mM", "z", "Uptake", "Selected_common_exposure"}
    if selected.empty or not needed <= set(selected.columns):
        return

    for (seq, ion), grp in selected.groupby(["Sequence", "Ion_family"], dropna=False):
        charges = sorted(pd.to_numeric(grp["z"], errors="coerce").dropna().unique())
        if len(charges) < 2:
            # A charge-state effect cannot be assessed when only one charge state
            # exists for the peptide across the compared conditions.
            continue

        exposure = float(grp["Selected_common_exposure"].iloc[0])
        fig, ax = plt.subplots(figsize=(8.2, 5.5))

        for state in sorted(grp["State"].astype(str).unique(), key=state_sort_key):
            g = grp[grp["State"].astype(str).eq(state)].sort_values("z")
            if g.empty:
                continue
            concentration = float(pd.to_numeric(g["Salt_mM"], errors="coerce").iloc[0])
            concentration_label = f"{concentration:g} mM"
            ax.plot(
                g["z"],
                g["Uptake"],
                marker="o",
                linewidth=1.7,
                markersize=6,
                label=concentration_label,
            )

        ax.set_xticks(charges)
        ax.set_xlabel("Charge state (z)")
        ax.set_ylabel("Experimental deuterium uptake (Da)")
        ax.set_title(
            f"Charge-state dependence of experimental uptake | {seq} | {ion}\n"
            f"highest common exposure = {exposure:g} ms"
        )
        ax.grid(True, alpha=0.25)
        ax.legend(title=f"{ion} concentration")

        save_figure(
            fig,
            outdir / "charge_state_effect" /
            f"{safe_name(seq)}_{ion}_charge_state_effect.{cfg.image_format}",
            cfg,
        )


# =============================================================================
# 6. VALIDATION PLOTS
# =============================================================================

def plot_parity(residuals: pd.DataFrame, outdir: Path, cfg: AnalysisConfig):
    if not {"Uptake", "Uptake_ESI"} <= set(residuals.columns):
        return
    valid = residuals.dropna(subset=["Uptake", "Uptake_ESI"])
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for state in sorted(valid["State"].astype(str).unique(), key=state_sort_key):
        g = valid[valid["State"].astype(str).eq(state)]
        ax.scatter(g["Uptake"], g["Uptake_ESI"], s=28, alpha=0.7, label=state)
    lo, hi = common_limits(valid["Uptake"], valid["Uptake_ESI"])
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Experimental uptake (Da)")
    ax.set_ylabel("Predicted final ESI uptake (Da)")
    ax.set_title("Experimental vs predicted uptake | all salt conditions")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    save_figure(fig, outdir / f"parity_all_states.{cfg.image_format}", cfg)

    for state, grp in valid.groupby("State"):
        fig, ax = plt.subplots(figsize=(6.3, 5.8))
        for seq, g in grp.groupby("Sequence"):
            ax.scatter(g["Uptake"], g["Uptake_ESI"], s=30, alpha=0.75, label=seq)
        lo, hi = common_limits(grp["Uptake"], grp["Uptake_ESI"])
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Experimental uptake (Da)")
        ax.set_ylabel("Predicted final ESI uptake (Da)")
        ax.set_title(f"Experimental vs predicted uptake | {state}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        save_figure(fig, outdir / "parity_by_state" / f"{safe_name(state)}.{cfg.image_format}", cfg)


def plot_residuals(residuals: pd.DataFrame, outdir: Path, cfg: AnalysisConfig):
    y = "Residual_ESI_minus_Experimental"
    if y not in residuals.columns:
        return
    for state, grp in residuals.groupby("State"):
        fig, ax = plt.subplots(figsize=(8.0, 5.3))
        for seq, g in grp.groupby("Sequence"):
            ax.scatter(g["Exposure"], g[y], s=32, alpha=0.75, label=seq)
        if (pd.to_numeric(grp["Exposure"], errors="coerce") > 0).all():
            ax.set_xscale("log")
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_xlabel("Exposure time (ms)")
        ax.set_ylabel("Predicted - experimental uptake (Da)")
        ax.set_title(f"Model residuals vs exposure | {state}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        save_figure(fig, outdir / "residuals_by_state" / f"{safe_name(state)}.{cfg.image_format}", cfg)


def plot_rmse_heatmap(metrics: pd.DataFrame, outdir: Path, cfg: AnalysisConfig):
    if metrics.empty or "RMSE_frac" not in metrics.columns:
        return
    pivot = metrics.pivot_table(index="Sequence", columns="State", values="RMSE_frac", aggfunc="mean")
    pivot = pivot.reindex(columns=sorted(pivot.columns.astype(str), key=state_sort_key))
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(pivot.columns)), max(4, 0.8 * len(pivot.index) + 2)))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Salt condition"); ax.set_ylabel("Peptide")
    ax.set_title("Normalized RMSE by peptide and salt condition")
    fig.colorbar(im, ax=ax, label="RMSE fraction")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    save_figure(fig, outdir / f"RMSE_frac_heatmap.{cfg.image_format}", cfg)


def plot_charge_separation(table: pd.DataFrame, outdir: Path, cfg: AnalysisConfig):
    need = {"ChargeSeparation_Experimental", "ChargeSeparation_Model"}
    if table.empty or not need <= set(table.columns):
        return
    valid = table.dropna(subset=list(need))
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for state, g in valid.groupby("State"):
        ax.scatter(g["ChargeSeparation_Experimental"], g["ChargeSeparation_Model"], s=34, alpha=0.75, label=state)
    lo, hi = common_limits(valid["ChargeSeparation_Experimental"], valid["ChargeSeparation_Model"])
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Experimental charge-state separation (Da)")
    ax.set_ylabel("Model charge-state separation (Da)")
    ax.set_title("Charge-state separation: experiment vs model")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=7)
    save_figure(fig, outdir / f"charge_state_separation_parity.{cfg.image_format}", cfg)


# =============================================================================
# 7. STAGE-LOSS AND PARAMETER PLOTS
# =============================================================================

def plot_stage_losses(summary: pd.DataFrame, outdir: Path, cfg: AnalysisConfig, baseline_state: str):
    if summary.empty:
        return
    base_all = summary[summary["State"].astype(str).eq(str(baseline_state))]
    for seq, seq_df in summary.groupby("Sequence"):
        base = base_all[base_all["Sequence"].eq(seq)]
        for col in [c for c in ["Loss_Quench", "Loss_LC", "Loss_ESI"] if c in summary.columns]:
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            for ion in ["NaCl", "CsCl"]:
                g = seq_df[seq_df["Salt_type"].eq(ion)].copy()
                if not base.empty:
                    g = pd.concat([base, g], ignore_index=True)
                g = g.sort_values("Salt_mM")
                if not g.empty:
                    ax.plot(g["Salt_mM"], g[col], marker="o", linewidth=1.6, label=ion)
            ax.set_xlabel("Salt concentration (mM)")
            ax.set_ylabel(f"{col} (Da)")
            ax.set_title(f"{col} vs salt concentration | {seq}")
            ax.grid(True, alpha=0.25); ax.legend()
            save_figure(fig, outdir / col / f"{safe_name(seq)}.{cfg.image_format}", cfg)


def _concentration_column_name(value: float) -> str:
    value = float(value)
    if math.isclose(value, round(value)):
        return f"{int(round(value))}mM"
    return f"{value:g}mM"


def _local_parameter_table(
    local_opt: pd.DataFrame,
    parameter: str,
    ion: str,
    baseline_state: str,
) -> pd.DataFrame:
    """
    Peptide x concentration table for one local optimized parameter.

    Raw fitted values are followed by explicit changes relative to 0mM.
    """
    if local_opt.empty or parameter not in local_opt.columns:
        return pd.DataFrame()

    baseline = local_opt[
        local_opt["State"].astype(str).eq(str(baseline_state))
    ].copy()
    ion_rows = local_opt[local_opt["Salt_type"].eq(ion)].copy()

    if ion_rows.empty:
        return pd.DataFrame()

    work = pd.concat([baseline, ion_rows], ignore_index=True)
    index_cols = [c for c in ["Protein", "Sequence"] if c in work.columns]
    if "Sequence" not in index_cols:
        return pd.DataFrame()

    pivot = work.pivot_table(
        index=index_cols,
        columns="Salt_mM",
        values=parameter,
        aggfunc="mean",
    ).sort_index(axis=1)

    if pivot.empty:
        return pd.DataFrame()

    concentrations = [float(c) for c in pivot.columns]
    result = pivot.copy()
    result.columns = [_concentration_column_name(c) for c in concentrations]

    zero_col = None
    for c in concentrations:
        if math.isclose(c, 0.0):
            zero_col = _concentration_column_name(c)
            break

    if zero_col is not None and zero_col in result.columns:
        for c in concentrations:
            if math.isclose(c, 0.0):
                continue
            cname = _concentration_column_name(c)
            result[f"Delta_{cname}_vs_0mM"] = result[cname] - result[zero_col]

    result = result.reset_index()
    return result


def _bounds_status_for_parameter(
    parameter: str,
    settings: dict[str, dict],
    preferred_state: str,
) -> tuple[str, str]:
    """Return a human-readable optimization status and bounds string."""
    settings_row = settings.get(str(preferred_state), {})

    if parameter == "gamma_lc_opt":
        bounds = settings_row.get("gamma_bounds")
    elif parameter == "tau_label_ms_opt":
        bounds = settings_row.get("tau_label_ms_bounds")
    else:
        bounds = None

    if bounds is None or len(bounds) != 2:
        if parameter == "global_gamma_tau_objective":
            return "Objective value (not directly bounded)", ""
        return "Bounds not available", ""

    lo, hi = float(bounds[0]), float(bounds[1])
    bounds_text = f"[{lo:g}, {hi:g}]"
    if math.isclose(lo, hi):
        return f"Fixed by equal bounds {bounds_text}", bounds_text
    return f"Optimized within bounds {bounds_text}", bounds_text


def _global_parameter_table(
    global_opt: pd.DataFrame,
    ion: str,
    baseline_state: str,
    settings: dict[str, dict],
) -> pd.DataFrame:
    """One human-readable global-parameter table for NaCl or CsCl."""
    if global_opt.empty:
        return pd.DataFrame()

    baseline = global_opt[
        global_opt["State"].astype(str).eq(str(baseline_state))
    ].copy()
    ion_rows = global_opt[global_opt["Salt_type"].eq(ion)].copy()

    if ion_rows.empty:
        return pd.DataFrame()

    work = pd.concat([baseline, ion_rows], ignore_index=True)
    work = work.sort_values("Salt_mM")

    specs = [
        ("gamma_lc_opt", "Global LC gamma"),
        ("tau_label_ms_opt", "Global effective labeling-time offset (ms)"),
        ("global_gamma_tau_objective", "Global optimization objective"),
    ]

    concentrations = sorted(
        float(x)
        for x in pd.to_numeric(work["Salt_mM"], errors="coerce").dropna().unique()
    )

    preferred_state = str(baseline_state)
    if preferred_state not in settings and not ion_rows.empty:
        preferred_state = str(ion_rows.iloc[0]["State"])

    rows = []
    for parameter, label in specs:
        if parameter not in work.columns:
            continue

        record = {
            "Parameter": label,
            "Internal_name": parameter,
        }

        values_by_conc = {}
        for concentration in concentrations:
            vals = pd.to_numeric(
                work[np.isclose(
                    pd.to_numeric(work["Salt_mM"], errors="coerce"),
                    concentration,
                    rtol=0.0,
                    atol=1e-8,
                )][parameter],
                errors="coerce",
            ).dropna()
            value = float(vals.mean()) if not vals.empty else np.nan
            cname = _concentration_column_name(concentration)
            record[cname] = value
            values_by_conc[concentration] = value

        baseline_value = next(
            (v for c, v in values_by_conc.items() if math.isclose(c, 0.0)),
            np.nan,
        )
        if np.isfinite(baseline_value):
            for concentration in concentrations:
                if math.isclose(concentration, 0.0):
                    continue
                cname = _concentration_column_name(concentration)
                value = values_by_conc.get(concentration, np.nan)
                record[f"Delta_{cname}_vs_0mM"] = (
                    value - baseline_value if np.isfinite(value) else np.nan
                )

        status, bounds_text = _bounds_status_for_parameter(
            parameter,
            settings,
            preferred_state,
        )
        record["Optimization_status"] = status
        record["Bounds"] = bounds_text
        rows.append(record)

    return pd.DataFrame(rows)


def save_optimization_parameter_tables(
    global_opt: pd.DataFrame,
    local_opt: pd.DataFrame,
    bounds: pd.DataFrame,
    slopes: pd.DataFrame,
    local_ion_diff: pd.DataFrame,
    param_effects: pd.DataFrame,
    param_corr: pd.DataFrame,
    settings: dict[str, dict],
    outdir: Path,
    baseline_state: str,
) -> None:
    """
    Replace optimization-parameter plots with organized CSV comparison tables.

    The main local tables are peptide x concentration and are separated into
    NaCl and CsCl. Each includes raw fitted values and changes from 0mM.
    """
    local_specs = [
        ("lambda_ESI_opt", "lambda_ESI"),
        ("t_ESI_opt", "t_ESI"),
        ("gamma_prime_p_opt", "gamma_prime_p"),
        ("ESI_objective_at_optimum", "ESI_objective_at_optimum"),
    ]

    written = []

    for ion in ["NaCl", "CsCl"]:
        ion_dir = outdir / "local" / ion
        ion_dir.mkdir(parents=True, exist_ok=True)

        for parameter, filename_stem in local_specs:
            table = _local_parameter_table(
                local_opt,
                parameter,
                ion,
                baseline_state,
            )
            if table.empty:
                continue

            path = ion_dir / f"{filename_stem}_by_concentration.csv"
            table.to_csv(path, index=False)
            written.append(path)

        global_table = _global_parameter_table(
            global_opt,
            ion,
            baseline_state,
            settings,
        )
        if not global_table.empty:
            global_dir = outdir / "global"
            global_dir.mkdir(parents=True, exist_ok=True)
            path = global_dir / f"{ion}_global_parameters_by_concentration.csv"
            global_table.to_csv(path, index=False)
            written.append(path)

    # Preserve useful optimization diagnostics as tables, not plots.
    diagnostics_dir = outdir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_tables = {
        "optimization_bound_diagnostics.csv": bounds,
        "parameter_salt_slopes.csv": slopes,
        "local_parameter_CsCl_minus_NaCl.csv": local_ion_diff,
        "parameter_effect_associations.csv": param_effects,
        "parameter_effect_correlations.csv": param_corr,
    }
    for filename, table in diagnostic_tables.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            path = diagnostics_dir / filename
            table.to_csv(path, index=False)
            written.append(path)

    readme = [
        "HDMX optimization-parameter comparison tables",
        "",
        "Local tables are separated into NaCl and CsCl.",
        "Rows = peptides; columns = 0mM and available salt concentrations,",
        "followed by explicit parameter changes relative to 0mM.",
        "",
        "Local parameters:",
        "  - lambda_ESI",
        "  - t_ESI",
        "  - gamma_prime_p",
        "  - ESI_objective_at_optimum",
        "",
        "Global tables contain gamma_lc, tau_label_ms and the global objective.",
        "If tau_label_ms bounds are equal (for example [0, 0]), the table marks",
        "tau_label_ms as fixed rather than interpreting it as a freely optimized trend.",
        "",
        "Files written:",
    ]
    readme.extend(f"  - {path.relative_to(outdir)}" for path in written)
    (outdir / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")


# =============================================================================
# 8. RATE AND RESIDUE/VECTOR DIAGNOSTICS
# =============================================================================

def plot_rates(rates: pd.DataFrame, outdir: Path, cfg: AnalysisConfig):
    if rates.empty:
        return
    specs = [
        ("label_k_int_vector", "Labeling intrinsic k_int"),
        ("quench_k_int_vector", "Quench intrinsic k_int"),
    ]
    for seq, seq_df in rates.groupby("Sequence"):
        for col, label in specs:
            if col not in seq_df.columns:
                continue
            fig, ax = plt.subplots(figsize=(8.0, 5.3))
            for _, r in seq_df.iterrows():
                vec = vector_from_value(r[col])
                if len(vec):
                    ax.plot(np.arange(len(vec)), vec, marker="o", markersize=3, linewidth=1.1, label=str(r["State"]))
            ax.set_xlabel("Residue / model-vector position"); ax.set_ylabel(label)
            ax.set_title(f"{label} across salt conditions | {seq}")
            ax.grid(True, alpha=0.25); ax.legend(fontsize=7)
            save_figure(fig, outdir / col / f"{safe_name(seq)}.{cfg.image_format}", cfg)


def representative_exposures(values) -> list[float]:
    vals = sorted(set(float(v) for v in values if np.isfinite(v)))
    if len(vals) <= 3:
        return vals
    return [vals[0], vals[len(vals)//2], vals[-1]]


def vector_combinations(rows: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "off":
        return pd.DataFrame()
    need = {"Sequence", "Exposure", "z", "State"}
    if not need <= set(rows.columns):
        return pd.DataFrame()
    combos = (
        rows[["Sequence", "Exposure", "z", "State"]]
        .dropna().drop_duplicates()
        .groupby(["Sequence", "Exposure", "z"], as_index=False)
        .agg(N_states=("State", "nunique"))
    )
    combos = combos[combos["N_states"] >= 2]
    if mode == "all":
        return combos
    selected = []
    for (seq, z), g in combos.groupby(["Sequence", "z"]):
        exps = representative_exposures(g["Exposure"].tolist())
        selected.append(g[g["Exposure"].isin(exps)])
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def plot_vectors(rows: pd.DataFrame, outdir: Path, cfg: AnalysisConfig):
    if cfg.vector_plot_mode == "off":
        return
    available = [c for c in VECTOR_COLUMNS if c in rows.columns]
    if not available:
        warnings.warn("No exact vector columns found; residue diagnostics skipped.")
        return
    combos = vector_combinations(rows, cfg.vector_plot_mode)
    if combos.empty:
        return
    specs = [
        ("Uptake_Vector_ESI", "Final residue-level ESI uptake"),
        ("Q_ESI_Vector", "Local ESI charge density Q"),
        ("k_ESI_vector", "Residue-level ESI rate"),
        ("LC_Integral_Vector", "Integrated LC rate"),
    ]
    specs = [s for s in specs if s[0] in rows.columns]

    for _, combo in combos.iterrows():
        seq, exposure, z = str(combo["Sequence"]), float(combo["Exposure"]), float(combo["z"])
        subset = rows[
            rows["Sequence"].astype(str).eq(seq)
            & pd.to_numeric(rows["Exposure"], errors="coerce").eq(exposure)
            & pd.to_numeric(rows["z"], errors="coerce").eq(z)
        ]
        for col, label in specs:
            fig, ax = plt.subplots(figsize=(8.5, 5.3))
            drawn = False
            for state in sorted(subset["State"].astype(str).unique(), key=state_sort_key):
                g = subset[subset["State"].astype(str).eq(state)]
                vec = mean_vector(g[col])
                if not len(vec):
                    continue
                drawn = True
                ax.plot(np.arange(len(vec)), vec, marker="o", markersize=3, linewidth=1.2, label=state)
            if not drawn:
                plt.close(fig)
                continue
            ax.set_xlabel("Residue / model-vector position"); ax.set_ylabel(label)
            ax.set_title(f"{label} by salt | {seq} | {exposure:g} ms | z={z:g}")
            ax.grid(True, alpha=0.25); ax.legend(fontsize=7)
            save_figure(fig, outdir / col / f"{safe_name(seq)}_{exposure:g}ms_z{z:g}.{cfg.image_format}", cfg)


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    cfg = AnalysisConfig()
    dirs = make_output_dirs(cfg)

    print("=" * 90)
    print("HDMX SAVED-RESULTS DOWNSTREAM ANALYSIS")
    print("=" * 90)
    print(f"Reading: {Path(cfg.results_root).resolve()}")
    print("No HDMX optimization or forward-model recalculation will be run.\n")

    data = load_saved_results(cfg)

    # These two sections changed structure in this version. Clear only their
    # generated outputs after the saved source data have loaded successfully, so
    # obsolete delta plots / parameter plots from a previous run cannot be
    # mistaken for current outputs.
    reset_generated_directory(dirs["experimental"])
    reset_generated_directory(dirs["parameters"])

    rows = data["rows"]
    metrics = data["metrics"]
    global_opt = data["global_opt"]
    local_opt = data["local_opt"]
    rates = data["rates"]
    settings = data["settings"]

    print("States discovered:")
    for state in data["states"]:
        print(f"  - {state}")

    agg = aggregate_rows(rows)
    delta = baseline_deltas(agg, cfg.baseline_state)
    ion_diff = ion_differences(agg)
    stage_rows, stage_summary = stage_losses(agg)
    residuals = residual_table(agg)
    charge_sep = charge_separation(agg)

    global_delta = global_parameter_deltas(global_opt, cfg.baseline_state)
    local_delta = local_parameter_deltas(local_opt, cfg.baseline_state)
    local_ion_diff = local_parameter_ion_differences(local_opt)
    slopes = parameter_slopes(global_opt, local_opt, cfg.baseline_state)
    bounds = bound_diagnostics(global_opt, local_opt, settings, cfg.bound_fraction_tolerance)
    param_effects, param_corr = parameter_effect_associations(local_opt, stage_summary, metrics)
    rate_check = rate_consistency(rates, cfg.baseline_state)

    last_common_points, last_common_summary = build_last_common_exposure_tables(
        agg, cfg.baseline_state
    )

    tables = {
        "aggregated_measurements.csv": agg,
        "salt_changes_vs_0mM.csv": delta,
        "CsCl_minus_NaCl_matched.csv": ion_diff,
        "stage_losses_row_level.csv": stage_rows,
        "stage_losses_summary.csv": stage_summary,
        "model_residuals.csv": residuals,
        "charge_state_separation.csv": charge_sep,
        "global_optimization_vs_salt.csv": global_delta,
        "local_optimization_vs_salt.csv": local_opt,
        "local_parameter_changes_vs_0mM.csv": local_delta,
        "local_parameter_CsCl_minus_NaCl.csv": local_ion_diff,
        "parameter_salt_slopes.csv": slopes,
        "optimization_bound_diagnostics.csv": bounds,
        "parameter_effect_associations.csv": param_effects,
        "parameter_effect_correlations.csv": param_corr,
        "intrinsic_rate_consistency_vs_0mM.csv": rate_check,
        "fit_metrics_all_states.csv": metrics,
        "global_optimization_raw.csv": global_opt,
        "intrinsic_rate_vectors_all_states.csv": rates,
        "experimental_last_common_exposure_points.csv": last_common_points,
        "experimental_last_common_exposure_selection.csv": last_common_summary,
    }
    for filename, df in tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(dirs["tables"] / filename, index=False)

    print("Creating experimental salt-effect plots...")
    # Keep the original full kinetic plots (one peptide x charge-state figure).
    plot_experimental_kinetics(agg, dirs["experimental"], cfg)

    # New raw-uptake summaries at the highest exposure common to all compared
    # state x charge combinations. No delta uptake is used on the y-axis.
    plot_last_exposure_vs_salt(last_common_points, dirs["experimental"], cfg)
    plot_charge_state_effect(last_common_points, dirs["experimental"], cfg)

    print("Creating model-validation plots...")
    plot_parity(residuals, dirs["validation"], cfg)
    plot_residuals(residuals, dirs["validation"], cfg)
    plot_rmse_heatmap(metrics, dirs["validation"], cfg)
    plot_charge_separation(charge_sep, dirs["validation"], cfg)

    print("Creating stage-loss plots...")
    plot_stage_losses(stage_summary, dirs["stage"], cfg, cfg.baseline_state)

    print("Creating global/local optimized-parameter tables...")
    save_optimization_parameter_tables(
        global_opt=global_opt,
        local_opt=local_opt,
        bounds=bounds,
        slopes=slopes,
        local_ion_diff=local_ion_diff,
        param_effects=param_effects,
        param_corr=param_corr,
        settings=settings,
        outdir=dirs["parameters"],
        baseline_state=cfg.baseline_state,
    )

    print("Creating intrinsic-rate consistency diagnostics...")
    plot_rates(rates, dirs["rates"], cfg)

    print("Creating saved residue/vector diagnostics...")
    plot_vectors(rows, dirs["vectors"], cfg)

    manifest = f"""
HDMX downstream analysis completed.

SOURCE
------
{Path(cfg.results_root).resolve()}

No HDMX optimizer or forward model was rerun.

STATES
------
{chr(10).join('  - ' + str(s) for s in data['states'])}

OUTPUTS
-------
00_tables/
  Numerical data behind all downstream comparisons.

01_experimental_salt_effects/
  Original experimental uptake-vs-exposure curves for each peptide x charge state.
  Raw experimental uptake vs NaCl/CsCl concentration at the highest common exposure.
  Charge-state-effect plots at the same highest common exposure.

02_model_validation/
  Experimental-vs-predicted parity.
  Residuals.
  RMSE heatmap.
  Charge-state-separation validation.

03_stage_losses/
  Quench, LC and ESI deuterium-loss trends vs salt concentration.

04_optimization_parameters/
  Human-readable CSV tables instead of parameter plots.
  Separate NaCl and CsCl tables for lambda_ESI, t_ESI, gamma_prime_p and ESI objective.
  Columns contain 0mM / salt concentrations and explicit changes vs 0mM.
  Global parameter tables contain gamma_lc, tau_label_ms and the global objective.
  Optimization diagnostics are retained as CSV tables.

05_rate_consistency/
  Saved intrinsic k_int curves across salt conditions.

06_residue_diagnostics/
  Saved residue-level ESI uptake, Q_ESI, k_ESI and LC integral profiles.

KEY TABLES
----------
parameter_salt_slopes.csv
  Simple descriptive fitted-parameter slopes per 100 mM, separately for NaCl
  and CsCl. These are descriptive, not causal estimates.

optimization_bound_diagnostics.csv
  Shows whether a fitted parameter is near a lower/upper bound or is fixed by
  equal bounds. Inspect this before giving fitted parameters physical meaning.

parameter_effect_associations.csv
  Joins local fitted parameters to mean LC/ESI loss and RMSE.

parameter_effect_correlations.csv
  Pearson associations for selected parameter/output pairs. These are not
  causal sensitivity coefficients.

intrinsic_rate_consistency_vs_0mM.csv
  Checks whether saved labeling/quench intrinsic-rate vectors differ from 0mM.

IMPORTANT
---------
If tau_label_ms_bounds were saved as [0, 0], tau_label_ms was fixed rather
than freely optimized. The bound diagnostics will explicitly mark that.

Residue/vector plot mode: {cfg.vector_plot_mode}
Set AnalysisConfig.vector_plot_mode = "all" to generate every matched vector
combination, or "off" to skip those plots.
""".strip() + "\n"
    (dirs["root"] / "analysis_manifest.txt").write_text(manifest, encoding="utf-8")

    print("\n" + "=" * 90)
    print("DOWNSTREAM ANALYSIS FINISHED")
    print(f"Output: {dirs['root'].resolve()}")
    print("Recommended first tables:")
    print("  - 00_tables/salt_changes_vs_0mM.csv")
    print("  - 00_tables/global_optimization_vs_salt.csv")
    print("  - 00_tables/local_parameter_changes_vs_0mM.csv")
    print("  - 00_tables/parameter_salt_slopes.csv")
    print("  - 00_tables/optimization_bound_diagnostics.csv")
    print("=" * 90)


if __name__ == "__main__":
    main()

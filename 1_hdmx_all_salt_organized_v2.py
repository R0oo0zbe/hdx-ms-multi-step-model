#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored HDMX workflow

Goals of this version:
1. Keep all parameters in one place.
2. Remove duplicated rate-generation code.
3. Replace slow row-wise uptake calculations with vectorized operations where possible.
4. Avoid global mutable state where possible.
5. Make each physical step explicit: labeling -> quench -> LC -> ESI -> optimization -> plotting.

Assumptions preserved from your original code:
- Exposure is stored in milliseconds.
- calculate_kint expects temperature in Kelvin.
- Quench/LC temperatures are entered in Celsius and converted using 273 + T_C, as in your code.
- The experimental uptake is Mass - MHP.
"""


from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
import json
import pickle
import re
from typing import Callable

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import minimize, minimize_scalar

from utils_kint import calculate_kint


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ExperimentConfig:
    input_csv: str = "PepMix_ClusterData.csv"
    state: str = "0mM"
    excluded_proteins: tuple[str, ...] = () #"Bradykinin",

    m_H: float = 1.0078
    m_D: float = 2.0141

    # Labeling conditions
    T_label_C: float = 20.0
    pH_label: float = 7.0
    x_label: float = 0.95

    # Quench conditions
    T_quench_C: float = 0.0
    pH_quench: float = 2.55
    x_quench: float = 0.475
    t_quench_sec: float = 180.0

    # Reference files for M0 uptake comparison
    high_file: str = "220718_pephigh_0_01"
    medium_file: str = "220718_pepmedium_0_01"

    # Batch selection only; this does NOT change the physical/model equations.
    # None = run every unique value found in the State column.
    states_to_run: tuple[str, ...] | None = None


@dataclass(frozen=True)
class OutputConfig:
    """Controls result organization only; it does not affect the model."""
    root_dir: str = "hdmx_results_all_salt"
    save_full_pickle: bool = True
    save_human_readable_csv: bool = True
    show_plots_during_batch: bool = False


@dataclass
class LCParams:
    V0: float = 0.15
    V_TAU: float = 0.40
    TAU_GRADIENT: float = 4.0


@dataclass
class LCChemParams:
    eta: float = 1.0
    gamma: float = 19.0

@dataclass
class OptimizationConfig:
    initial_gamma: float = 15.0
    gamma_bounds: tuple[float, float] = (-10.0, 25.0)

    # Global effective labeling-time offset fitted for the whole experiment.
    # Effective exposure is Exposure_eff = Exposure + tau_label_ms.
    # Bound is constrained to 0--50 ms.
    tau_label_ms_bounds: tuple[float, float] = (0.0, 00.0)
    tau_label_ms_initial: float = 0.0
    tau_label_ms_reg_weight: float = 0.0
    # Final ESI optimization: alpha and xi_ESI are fixed, only t_ESI and lambda_ESI are fitted.
    fixed_alpha_ESI: float = 1.0
    fixed_xi_ESI: float = 1.0
    t_esi_bounds: tuple[float, float] = (0.0, 5000.0)
    lambda_esi_bounds: tuple[float, float] = (-10.0, 10.0)
    t_esi_initial: float = 100.0
    lambda_esi_initial: float = 0.0
    lc_n_steps: int = 100
    gamma_prime_p_bounds: tuple[float, float] = (-10.0, 10.0)
    gamma_prime_p_reg_weight: float = 0.01


# =============================================================================
# 2. SMALL NUMERICAL HELPERS
# =============================================================================

def celsius_to_kelvin(T_C: float) -> float:
    """Convert Celsius to Kelvin using the same offset used in the original code."""
    return 273.0 + T_C


def flow(k: np.ndarray, K: np.ndarray, x0: np.ndarray, t: float) -> np.ndarray:
    """
    Two-state exchange flow model.

    D(t) = x0 * exp(-k t) + D_eq * (1 - exp(-k t))
    where D_eq = K / (1 + K)
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        D_eq = K / (1.0 + K)
        exp_term = np.exp(-k * t)
        return x0 * exp_term + D_eq * (1.0 - exp_term)


def rates_to_k_and_K(rates_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert [kforw, kback] matrix into k_int and K_eq vectors."""
    k_forw = rates_matrix[:, 0]
    k_back = rates_matrix[:, 1]
    k_int = k_forw + k_back
    with np.errstate(divide="ignore", invalid="ignore"):
        K_eq = k_forw / k_back
    return k_int, K_eq


# =============================================================================
# 3. DATA LOADING AND SIMPLE VECTORIZED COLUMNS
# =============================================================================

def discover_states(cfg: ExperimentConfig) -> list[str]:
    """Return salt/state values to run, preserving the order in the input CSV."""
    df = pd.read_csv(cfg.input_csv, usecols=["State", "Protein"])
    df = df[~df["Protein"].isin(cfg.excluded_proteins)].copy()

    available = df["State"].dropna().astype(str).drop_duplicates().tolist()
    if cfg.states_to_run is None:
        return available

    requested = [str(x) for x in cfg.states_to_run]
    missing = [x for x in requested if x not in available]
    if missing:
        raise ValueError(
            f"Requested State values not found in input CSV: {missing}. "
            f"Available values: {available}"
        )
    return requested


def load_and_prepare_data(
    cfg: ExperimentConfig,
    state: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load one state/salt condition, then calculate mass and experimental uptake.

    This intentionally preserves the original single-state filtering logic.
    Batch execution simply calls this function once per State value.
    """
    selected_state = cfg.state if state is None else str(state)
    df = pd.read_csv(cfg.input_csv)

    df = df[df["State"].astype(str).eq(selected_state)].copy()
    df = df[~df["Protein"].isin(cfg.excluded_proteins)].copy()
    df.reset_index(drop=True, inplace=True)

    df["Mass"] = df["Center"] * df["z"] - df["z"] * (cfg.m_D - cfg.m_H)
    df["Uptake"] = df["Mass"] - df["MHP"]

    m0 = df[df["Exposure"].eq(0.0)].copy()
    dynamic = df[df["Exposure"].gt(0.0)].copy()

    return df, m0, dynamic


def add_reference_uptakes(
    dynamic: pd.DataFrame,
    m0: pd.DataFrame,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """
    Add Uptake_pephigh and Uptake_pepmedium without row-wise apply.
    This replaces the original calculate_uptakes(row) function.
    """
    out = dynamic.copy()

    reference_specs = {
        "Uptake_pephigh": cfg.high_file,
        "Uptake_pepmedium": cfg.medium_file,
    }

    for new_col, ref_file in reference_specs.items():
        ref = (
            m0[m0["File"].eq(ref_file)][["Sequence", "z", "Center"]]
            .rename(columns={"Center": f"Center_{new_col}"})
        )

        out = out.merge(ref, on=["Sequence", "z"], how="left")
        out[new_col] = (out["Center"] - out[f"Center_{new_col}"]) * out["z"]
        out.drop(columns=[f"Center_{new_col}"], inplace=True)

    return out


# =============================================================================
# 4. INTRINSIC RATE GENERATION
# =============================================================================

def build_rate_dict(
    sequences: np.ndarray,
    T_C: float,
    x: float,
    pH: float,
    ref: str = "3Ala",
) -> dict[str, np.ndarray]:
    """Calculate intrinsic [kforw, kback] arrays for each sequence."""
    rate_dict: dict[str, np.ndarray] = {}
    T_K = celsius_to_kelvin(T_C)

    for seq in sequences:
        seq = str(seq).strip()
        try:
            rates_df = calculate_kint(seq, T_K, x, pH, ref=ref)
            rate_dict[seq] = rates_df[["kforw", "kback"]].to_numpy(dtype=float)
        except Exception as exc:
            print(f"Warning: calculate_kint failed for {seq}: {exc}")
            rate_dict[seq] = np.zeros((len(seq), 2), dtype=float)

    return rate_dict


# Cache the expensive calculate_kint calls used inside LC profile construction.
# The rounded pH prevents tiny floating-point differences from destroying cache reuse.
@lru_cache(maxsize=200_000)
def cached_lc_base_rate(seq: str, T_C: float, pH_eff_rounded: float) -> tuple[float, ...]:
    rates_df = calculate_kint(seq, celsius_to_kelvin(T_C), 0.0, pH_eff_rounded, ref="3Ala")
    k_base = rates_df["kforw"].to_numpy(dtype=float) + rates_df["kback"].to_numpy(dtype=float)
    return tuple(k_base)


# =============================================================================
# 5. LABELING AND QUENCH STEPS
# =============================================================================

def labeling_vector(
    row: pd.Series,
    rate_dict: dict[str, np.ndarray],
    tau_label_ms: float = 0.0,
) -> np.ndarray | float:
    """
    Calculate residue-level labeling vector for one row.

    tau_label_ms is a global effective labeling-time offset in milliseconds.
    The effective exposure is

        Exposure_eff = max(0, Exposure + tau_label_ms)

    In the current model tau_label_ms is fitted globally and constrained to 0--50 ms.
    """
    seq = str(row["Sequence"]).strip()
    rates = rate_dict.get(seq)
    if rates is None:
        return np.nan

    k_int, K_eq = rates_to_k_and_K(rates)
    x0 = np.zeros(len(k_int), dtype=float)

    # Exposure is stored in ms; rates are assumed s^-1.
    exposure_eff_ms = max(0.0, float(row["Exposure"]) + float(tau_label_ms))
    return flow(k_int, K_eq, x0, exposure_eff_ms / 1000.0)


def quench_vector(
    row: pd.Series,
    quench_rate_dict: dict[str, np.ndarray],
    t_quench_sec: float,
) -> np.ndarray | float:
    """Apply quench exchange to a previously calculated labeling vector."""
    seq = str(row["Sequence"]).strip()
    x0 = row.get("Uptake_Vector")
    rates = quench_rate_dict.get(seq)

    if not isinstance(x0, np.ndarray) or rates is None:
        return np.nan

    k_int, K_eq = rates_to_k_and_K(rates)
    n = min(len(x0), len(k_int))

    return flow(k_int[:n], K_eq[:n], x0[:n], t_quench_sec)


def add_labeling_and_quench_columns(
    dynamic: pd.DataFrame,
    label_rate_dict: dict[str, np.ndarray],
    quench_rate_dict: dict[str, np.ndarray],
    cfg: ExperimentConfig,
    tau_label_ms: float = 0.0,
) -> pd.DataFrame:
    """Add labeling and quench vector/scalar columns."""
    out = dynamic.copy()
    out["tau_label_ms"] = float(tau_label_ms)

    out["Uptake_Vector"] = out.apply(
        labeling_vector,
        axis=1,
        rate_dict=label_rate_dict,
        tau_label_ms=float(tau_label_ms),
    )
    out["Uptake_Labeling"] = out["Uptake_Vector"].map(vector_sum)

    out["Uptake_Vector_Quenched"] = out.apply(
        quench_vector,
        axis=1,
        quench_rate_dict=quench_rate_dict,
        t_quench_sec=cfg.t_quench_sec,
    )
    out["Uptake_Quenched"] = out["Uptake_Vector_Quenched"].map(vector_sum)

    return out


def vector_sum(value: object) -> float:
    """Safe sum for array-valued dataframe columns."""
    if isinstance(value, np.ndarray):
        return float(np.nansum(value))
    return np.nan


# =============================================================================
# 6. LC STEP
# =============================================================================

def lc_gradient(t_min: float, lc: LCParams) -> float:
    """Linear LC gradient from V0 to V_TAU, then hold."""
    if t_min <= lc.TAU_GRADIENT:
        slope = (lc.V_TAU - lc.V0) / lc.TAU_GRADIENT
        return lc.V0 + slope * t_min
    return lc.V_TAU


def lc_rate_vector(
    v: float,
    seq: str,
    lc_chem: LCChemParams,
    T_C: float,
    pH: float,
) -> np.ndarray:
    """
    LC back-exchange rate vector at solvent fraction v.

    Base LC term without the peptide-local intercept:
        k_base,i * (1 - v)^eta * exp(gamma * v)

    The peptide-local gamma_prime_p intercept is exposure-independent and is
    applied in lc_step as exp(gamma_prime_p). This is equivalent to:
        k_LC,i^(p)(v) = k_base,i * (1 - v)^eta * exp(gamma * v + gamma_prime_p)
    """
    pH_rounded = round(float(pH), 6)
    k_base = np.asarray(cached_lc_base_rate(seq, float(T_C), pH_rounded), dtype=float)

    exponent = lc_chem.gamma * float(v)
    scaling = ((1.0 - v) ** lc_chem.eta) * np.exp(np.clip(exponent, -80.0, 80.0))
    return scaling * k_base


def precalculate_lc_integrals(
    sequences: np.ndarray,
    lc: LCParams,
    lc_chem: LCChemParams,
    T_C: float,
    pH: float,
    n_steps: int,
) -> dict[str, Callable[[float], np.ndarray]]:
    """
    Precalculate integral_0^RT k_LC(t) dt for each sequence.

    Returns:
        lc_integrals[sequence](RT) -> vector of integrated LC rates
    """
    t_grid = np.linspace(0.0, lc.TAU_GRADIENT, n_steps)
    v_grid = np.array([lc_gradient(t, lc) for t in t_grid])

    lc_integrals: dict[str, Callable[[float], np.ndarray]] = {}

    for seq in sequences:
        seq = str(seq).strip()

        rate_matrix = np.vstack([
            lc_rate_vector(
                v,
                seq,
                lc_chem,
                T_C,
                pH,
            )
            for v in v_grid
        ])

        cumulative = cumulative_trapezoid(rate_matrix, t_grid, axis=0, initial=0.0)
        lc_integrals[seq] = interp1d(
            t_grid,
            cumulative,
            axis=0,
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=True,
        )

    return lc_integrals


def lc_step(
    row: pd.Series,
    lc_integrals: dict[str, Callable[[float], np.ndarray]],
    gamma_prime_params: dict[str, float] | None = None,
) -> pd.Series:
    """
    Apply LC survival to the quenched vector.
    """
    seq = str(row["Sequence"]).strip()
    D0 = row.get("Uptake_Vector_Quenched")

    if seq not in lc_integrals or not isinstance(D0, np.ndarray):
        return pd.Series([np.nan, np.nan], index=["Uptake_LC", "Uptake_Vector_LC"])

    integral_vector = np.asarray(lc_integrals[seq](row["RT"]), dtype=float)
    if gamma_prime_params is not None:
        gamma_prime_p = float(gamma_prime_params.get(seq, 0.0))
        # Since gamma_prime_p is an exposure-independent LC intercept,
        # multiplying the LC integral by exp(gamma_prime_p) is mathematically
        # equivalent to putting +gamma_prime_p inside the LC rate exponent.
        integral_vector = integral_vector * np.exp(np.clip(gamma_prime_p, -80.0, 80.0))

    n = min(len(D0), len(integral_vector))

    D_lc = D0[:n] * np.exp(np.clip(-integral_vector[:n], -700.0, 0.0))
    return pd.Series([np.nansum(D_lc), D_lc], index=["Uptake_LC", "Uptake_Vector_LC"])


def add_lc_columns(
    dynamic: pd.DataFrame,
    lc_integrals: dict[str, Callable[[float], np.ndarray]],
    gamma_prime_params: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Apply LC step using precomputed LC integrals."""
    out = dynamic.copy()
    out[["Uptake_LC", "Uptake_Vector_LC"]] = out.apply(
        lc_step,
        axis=1,
        lc_integrals=lc_integrals,
        gamma_prime_params=gamma_prime_params,
    )
    return out


# =============================================================================
# 7. ESI STEP
# =============================================================================

# New local charge-density ESI model with accessibility ignored:
# A_s(z) = 1 for every charge-localizing site.
#
# Model used here:
#   P_s(z) = min(1, z * exp(alpha * b_s) / sum_u exp(alpha * b_u))
#   Q_i(z) = sum_s P_s(z) * exp(-abs(i - s) / xi_ESI)
#   f_i(z) = exp(lambda_ESI * Q_i(z))
#   k_ESI,i = k_int,i * f_i(z)

DEFAULT_CHARGE_AFFINITIES: dict[str, float] = {
    "NTERM": 1.0,
    "R": 1.0,
    "K": 0.7,
    "H": 0.3,
}
DEFAULT_XI_ESI: float = 3.0
DEFAULT_LAMBDA_ESI: float = 1.0


def charge_localizing_sites(seq: str) -> list[tuple[int, str]]:
    """
    Return possible charge-localizing sites as (position, site_type).

    Positions use the same zero-based indexing as the uptake/rate vectors.
    The N-terminus is assigned to position 0. Arg/Lys/His are assigned to
    their residue positions in the peptide sequence.
    """
    seq = str(seq).strip()
    sites: list[tuple[int, str]] = [(0, "NTERM")]

    for pos, aa in enumerate(seq):
        if aa in {"R", "K", "H"}:
            sites.append((pos, aa))

    return sites


def charge_site_probabilities(
    z: float,
    seq: str,
    alpha: float = 1.0,
    charge_affinities: dict[str, float] | None = None,
) -> list[tuple[int, str, float]]:
    """
    Calculate capped charge-localization probabilities P_s(z).

    alpha controls how selectively charge is assigned to high-affinity sites.
    Larger alpha concentrates charge more strongly on Arg/N-terminus-like sites.
    Accessibility is intentionally ignored here, equivalent to A_s(z)=1.
    """
    if charge_affinities is None:
        charge_affinities = DEFAULT_CHARGE_AFFINITIES

    sites = charge_localizing_sites(seq)
    if len(sites) == 0:
        return []

    z = max(float(z), 0.0)
    alpha = max(float(alpha), 0.0)

    b = np.array([charge_affinities.get(site_type, 0.0) for _, site_type in sites], dtype=float)

    # Stable softmax: subtract max before exponentiation.
    logits = alpha * b
    logits = logits - np.nanmax(logits)
    weights = np.exp(logits)
    weights_sum = np.nansum(weights)

    if not np.isfinite(weights_sum) or weights_sum <= 0.0:
        probs = np.zeros(len(sites), dtype=float)
    else:
        probs = z * weights / weights_sum
        probs = np.minimum(1.0, probs)

    return [(pos, site_type, float(prob)) for (pos, site_type), prob in zip(sites, probs)]


def local_charge_density_vector(
    seq: str,
    z: float,
    n_amides: int,
    alpha: float = 1.0,
    xi_ESI: float = DEFAULT_XI_ESI,
    charge_affinities: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Calculate Q_i(z) for every amide i with A_s(z)=1.

    Q_i(z) = sum_s P_s(z) * exp(-abs(i - s) / xi_ESI)
    """
    n_amides = int(n_amides)
    if n_amides <= 0:
        return np.array([], dtype=float)

    xi_ESI = max(float(xi_ESI), 1e-12)
    amide_positions = np.arange(n_amides, dtype=float)
    Q = np.zeros(n_amides, dtype=float)

    for site_pos, _, P_s in charge_site_probabilities(z, seq, alpha, charge_affinities):
        Q += P_s * np.exp(-np.abs(amide_positions - float(site_pos)) / xi_ESI)

    return Q


def local_charge_enhancement_vector(
    seq: str,
    z: float,
    n_amides: int,
    alpha: float = 1.0,
    xi_ESI: float = DEFAULT_XI_ESI,
    lambda_ESI: float = DEFAULT_LAMBDA_ESI,
    charge_affinities: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Calculate f_i(z) = exp(lambda_ESI * Q_i(z)).
    """
    Q = local_charge_density_vector(
        seq=seq,
        z=z,
        n_amides=n_amides,
        alpha=alpha,
        xi_ESI=xi_ESI,
        charge_affinities=charge_affinities,
    )
    return np.exp(float(lambda_ESI) * Q)


def esi_rate_vector(
    k_int_vec: np.ndarray,
    z: float,
    seq: str,
    alpha: float = 1.0,
    xi_ESI: float = DEFAULT_XI_ESI,
    lambda_ESI: float = DEFAULT_LAMBDA_ESI,
    charge_affinities: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Site-specific ESI rate vector using the local charge-density model.

    k_ESI,i = k_int,i * exp(lambda_ESI * Q_i(z))
    """
    k_int_vec = np.asarray(k_int_vec, dtype=float)
    f_i = local_charge_enhancement_vector(
        seq=seq,
        z=z,
        n_amides=len(k_int_vec),
        alpha=alpha,
        xi_ESI=xi_ESI,
        lambda_ESI=lambda_ESI,
        charge_affinities=charge_affinities,
    )
    return k_int_vec * f_i


def esi_step(
    row: pd.Series,
    quench_rate_dict: dict[str, np.ndarray],
    t_ESI: float,
    alpha: float = 1.0,
    xi_ESI: float = DEFAULT_XI_ESI,
    lambda_ESI: float = DEFAULT_LAMBDA_ESI,
) -> pd.Series:
    """Apply site-specific ESI decay to the LC vector with A_s(z)=1."""
    seq = str(row["Sequence"]).strip()
    D_lc = row.get("Uptake_Vector_LC")
    rates = quench_rate_dict.get(seq)

    if not isinstance(D_lc, np.ndarray) or rates is None:
        return pd.Series([np.nan, np.nan], index=["Uptake_ESI", "Uptake_Vector_ESI"])

    k_int = rates[:, 0] + rates[:, 1]
    k_esi = esi_rate_vector(
        k_int,
        row["z"],
        seq,
        alpha=alpha,
        xi_ESI=xi_ESI,
        lambda_ESI=lambda_ESI,
    )
    n = min(len(D_lc), len(k_esi))

    D_esi = D_lc[:n] * np.exp(-k_esi[:n] * t_ESI)
    return pd.Series([np.nansum(D_esi), D_esi], index=["Uptake_ESI", "Uptake_Vector_ESI"])


def apply_esi_columns(
    dynamic: pd.DataFrame,
    esi_params: dict[str, tuple[float, float]],
    quench_rate_dict: dict[str, np.ndarray],
    fixed_alpha: float = 1.0,
    fixed_xi_ESI: float = 1.0,
) -> pd.DataFrame:
    """
    Apply optimized ESI parameters to all rows.

    Current final ESI model keeps alpha and xi_ESI fixed and uses only
    peptide-specific optimized (t_ESI, lambda_ESI).
    """
    out = dynamic.copy()

    def _apply(row: pd.Series) -> pd.Series:
        seq = str(row["Sequence"]).strip()
        if seq not in esi_params:
            return pd.Series([np.nan, np.nan], index=["Uptake_ESI", "Uptake_Vector_ESI"])

        t_ESI, lambda_ESI = esi_params[seq]

        return esi_step(
            row,
            quench_rate_dict,
            t_ESI=t_ESI,
            alpha=fixed_alpha,
            lambda_ESI=lambda_ESI,
            xi_ESI=fixed_xi_ESI,
        )

    out[["Uptake_ESI", "Uptake_Vector_ESI"]] = out.apply(_apply, axis=1)
    return out


# =============================================================================
# 8. OPTIMIZATION
# =============================================================================


def prepare_esi_records(
    dynamic_with_lc: pd.DataFrame,
    quench_rate_dict: dict[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    """
    Convert dataframe rows into compact numpy records for fast ESI optimization.
    """
    records: dict[str, dict[str, object]] = {}

    for seq, peptide_df in dynamic_with_lc.groupby("Sequence"):
        seq = str(seq).strip()
        peptide_df = peptide_df.reset_index(drop=True)
        rates = quench_rate_dict.get(seq)

        if rates is None:
            continue

        valid_rows = peptide_df[peptide_df["Uptake_Vector_LC"].map(lambda x: isinstance(x, np.ndarray))].copy()
        if valid_rows.empty:
            continue

        k_int_full = rates[:, 0] + rates[:, 1]
        n_amides = min(
            len(k_int_full),
            min(len(v) for v in valid_rows["Uptake_Vector_LC"]),
        )

        if n_amides <= 0:
            continue

        D_lc_matrix = np.vstack([
            np.asarray(v, dtype=float)[:n_amides]
            for v in valid_rows["Uptake_Vector_LC"]
        ])

        z_values = valid_rows["z"].to_numpy(dtype=float)
        exp_values = valid_rows["Uptake"].to_numpy(dtype=float)
        exposure_values = valid_rows["Exposure"].to_numpy(dtype=float)
        k_int = np.asarray(k_int_full[:n_amides], dtype=float)

        exposure_groups = []
        for exposure in np.unique(exposure_values):
            idx = np.where(exposure_values == exposure)[0]
            if len(idx) >= 2:
                exposure_groups.append(idx)

        records[seq] = {
            "seq": seq,
            "D_lc_matrix": D_lc_matrix,
            "k_int": k_int,
            "z_values": z_values,
            "unique_z": np.unique(z_values),
            "exp_values": exp_values,
            "exposure_groups": exposure_groups,
            "n_amides": n_amides,
        }

    return records


def simulate_esi_values_from_record(
    params,
    record: dict[str, object],
) -> np.ndarray:
    """Fast vectorized simulation of total ESI uptake for one peptide."""
    alpha = float(params[0])
    t_ESI = float(params[1])
    lambda_ESI = float(params[2])
    xi_ESI = float(params[3])

    seq = record["seq"]
    D_lc_matrix = record["D_lc_matrix"]
    k_int = record["k_int"]
    z_values = record["z_values"]
    unique_z = record["unique_z"]
    n_amides = int(record["n_amides"])

    sim_values = np.empty(len(z_values), dtype=float)

    for z in unique_z:
        Q = local_charge_density_vector(
            seq=seq,
            z=float(z),
            n_amides=n_amides,
            alpha=alpha,
            xi_ESI=xi_ESI,
        )
        enhancement = np.exp(np.clip(lambda_ESI * Q, -50.0, 50.0))
        k_esi = k_int * enhancement

        row_mask = z_values == z
        decay = np.exp(np.clip(-k_esi * t_ESI, -700.0, 0.0))
        sim_values[row_mask] = np.nansum(D_lc_matrix[row_mask] * decay, axis=1)

    return sim_values


def esi_objective_fast(
    params,
    record: dict[str, object],
) -> float:
    """Fast ESI objective for one peptide using prebuilt numpy arrays."""
    alpha = float(params[0])
    t_ESI = float(params[1])
    lambda_ESI = float(params[2])
    xi_ESI = float(params[3])

    if not np.all(np.isfinite([alpha, t_ESI, lambda_ESI, xi_ESI])):
        return 1e12

    # lambda_ESI can be negative; xi_ESI must stay positive.
    if alpha < 0.0 or t_ESI < 0.0 or xi_ESI <= 0.0:
        return 1e12

    sim_values = simulate_esi_values_from_record(params, record)
    exp_values = record["exp_values"]
    z_values = record["z_values"]

    mse = np.nanmean((sim_values - exp_values) ** 2)

    # Signed separation penalty: preserves the direction of charge-state ordering.
    signed_sep_penalty = 0.0
    for idx in record["exposure_groups"]:
        z_group = z_values[idx]
        sim_group = sim_values[idx]
        exp_group = exp_values[idx]
        unique_z = np.sort(np.unique(z_group))

        if len(unique_z) < 2:
            continue

        for a in range(len(unique_z)):
            for b in range(a + 1, len(unique_z)):
                z_low = unique_z[a]
                z_high = unique_z[b]

                sim_low = np.nanmean(sim_group[z_group == z_low])
                sim_high = np.nanmean(sim_group[z_group == z_high])
                exp_low = np.nanmean(exp_group[z_group == z_low])
                exp_high = np.nanmean(exp_group[z_group == z_high])

                signed_sep_sim = sim_high - sim_low
                signed_sep_exp = exp_high - exp_low
                signed_sep_penalty += (signed_sep_sim - signed_sep_exp) ** 2

    return float(mse + 5.0 * signed_sep_penalty)


def esi_objective(
    params,
    peptide_df,
    quench_rate_dict=None,
) -> float:
    """Compatibility wrapper for dataframe or prebuilt record input."""
    if isinstance(peptide_df, dict):
        return esi_objective_fast(params, peptide_df)

    if quench_rate_dict is None:
        raise ValueError("quench_rate_dict is required when peptide_df is a dataframe.")

    records = prepare_esi_records(peptide_df, quench_rate_dict)
    if len(records) == 0:
        return 1e12

    record = next(iter(records.values()))
    return esi_objective_fast(params, record)


def optimize_per_peptide_esi(
    dynamic_with_lc: pd.DataFrame,
    quench_rate_dict: dict[str, np.ndarray],
    opt: OptimizationConfig,
    fast: bool = False,
    verbose: bool = True,
    initial_params: dict[str, tuple[float, float]] | None = None,
    fixed_alpha: float | None = None,
    fixed_xi_ESI: float | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Final ESI optimization with alpha and xi_ESI fixed.

    Optimized per peptide:
        t_ESI
        lambda_ESI

    Fixed globally:
        alpha = opt.fixed_alpha_ESI by default
        xi_ESI = opt.fixed_xi_ESI by default
    """
    if fixed_alpha is None:
        fixed_alpha = float(opt.fixed_alpha_ESI)
    if fixed_xi_ESI is None:
        fixed_xi_ESI = float(opt.fixed_xi_ESI)

    records = prepare_esi_records(dynamic_with_lc, quench_rate_dict)
    esi_params: dict[str, tuple[float, float]] = {}

    bounds = [
        opt.t_esi_bounds,        # t_ESI
        opt.lambda_esi_bounds,   # lambda_ESI can be negative or positive
    ]

    maxiter = 80 if fast else 1500

    for seq, record in records.items():
        if initial_params is not None and seq in initial_params:
            base_x0 = list(initial_params[seq])
        else:
            base_x0 = [opt.t_esi_initial, opt.lambda_esi_initial]

        if fast:
            starts = [base_x0]
        else:
            starts = [
                base_x0,
                [opt.t_esi_initial,  2.0],
                [opt.t_esi_initial, -2.0],
                [opt.t_esi_initial, -5.0],
                [500.0,              2.0],
                [500.0,             -2.0],
            ]

        best_result = None
        for x0 in starts:
            x0 = np.asarray(x0, dtype=float)
            x0[0] = np.clip(x0[0], bounds[0][0], bounds[0][1])
            x0[1] = np.clip(x0[1], bounds[1][0], bounds[1][1])

            def objective_t_lambda(x: np.ndarray) -> float:
                t_ESI = float(x[0])
                lambda_ESI = float(x[1])

                params4 = [
                    fixed_alpha,
                    t_ESI,
                    lambda_ESI,
                    fixed_xi_ESI,
                ]
                return esi_objective_fast(params4, record)

            result = minimize(
                objective_t_lambda,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": 1e-9},
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result

        tESI_opt = float(best_result.x[0])
        lambda_opt = float(best_result.x[1])

        esi_params[str(seq)] = (tESI_opt, lambda_opt)

        if verbose:
            print(
                f"{seq} -> alpha_fixed={fixed_alpha:.3f}, "
                f"xi_ESI_fixed={fixed_xi_ESI:.3f}, "
                f"t_ESI={tESI_opt:.2f}, "
                f"lambda_ESI={lambda_opt:.3f}, "
                f"loss={best_result.fun:.4f}"
            )

    return esi_params


def optimize_reduced_esi_for_lc_stage(
    dynamic_with_lc: pd.DataFrame,
    quench_rate_dict: dict[str, np.ndarray],
    opt: OptimizationConfig,
    fixed_alpha: float = 1.0,
    fixed_xi_ESI: float = 1.0,
    fast: bool = True,
) -> dict[str, tuple[float, float, float, float]]:
    """
    Reduced temporary ESI fit used during LC optimization stages.

    This is intentionally consistent with the final reduced ESI model:
        alpha is fixed.
        xi_ESI is fixed.
        t_ESI and lambda_ESI are optimized per peptide.

    This replaces the older LC-stage surrogate where lambda_ESI was fixed to 1.0.
    The goal is to avoid calibrating LC against an unrealistic ESI correction.
    """
    records = prepare_esi_records(dynamic_with_lc, quench_rate_dict)
    esi_params: dict[str, tuple[float, float, float, float]] = {}

    bounds = [
        opt.t_esi_bounds,
        opt.lambda_esi_bounds,
    ]

    if fast:
        starts = [
            [opt.t_esi_initial, opt.lambda_esi_initial],
            [opt.t_esi_initial, 2.0],
            [opt.t_esi_initial, 6.0],
            [10.0, 2.0],
            [10.0, 6.0],
        ]
        maxiter = 120
    else:
        starts = [
            [opt.t_esi_initial, opt.lambda_esi_initial],
            [opt.t_esi_initial, 2.0],
            [opt.t_esi_initial, 6.0],
            [opt.t_esi_initial, -2.0],
            [10.0, 2.0],
            [10.0, 6.0],
            [100.0, 2.0],
            [500.0, 2.0],
        ]
        maxiter = 300

    for seq, record in records.items():

        def objective_t_lambda(x: np.ndarray) -> float:
            t_ESI = float(x[0])
            lambda_ESI = float(x[1])
            params = [fixed_alpha, t_ESI, lambda_ESI, fixed_xi_ESI]
            return esi_objective_fast(params, record)

        best_result = None
        for x0 in starts:
            x0 = np.asarray(x0, dtype=float)
            x0[0] = np.clip(x0[0], bounds[0][0], bounds[0][1])
            x0[1] = np.clip(x0[1], bounds[1][0], bounds[1][1])

            result = minimize(
                objective_t_lambda,
                x0=x0,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": 1e-8},
            )

            if best_result is None or result.fun < best_result.fun:
                best_result = result

        esi_params[str(seq)] = (
            fixed_alpha,
            float(best_result.x[0]),
            float(best_result.x[1]),
            fixed_xi_ESI,
        )

    return esi_params


def make_global_lc_gamma_tau_objective(
    dynamic_base: pd.DataFrame,
    sequences: np.ndarray,
    label_rate_dict: dict[str, np.ndarray],
    quench_rate_dict: dict[str, np.ndarray],
    cfg: ExperimentConfig,
    lc: LCParams,
    lc_chem_template: LCChemParams,
    T_lc_C: float,
    pH_lc: float,
    opt: OptimizationConfig,
    gamma_prime_params: dict[str, float] | None = None,
) -> Callable[[np.ndarray], float]:
    """
    Objective for joint global LC gamma and global effective labeling-time optimization.

    Optimized globally:
        gamma
        tau_label_ms, constrained to 0--50 ms

    During this LC-stage calibration, ESI is reduced in the same way as the final
    reduced ESI model: alpha and xi_ESI are fixed, while t_ESI and lambda_ESI
    are optimized per peptide.
    """
    cache: dict[tuple[float, float], float] = {}

    def objective(x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=float).ravel()
        gamma = float(arr[0])
        tau_label_ms = float(arr[1])

        if not (opt.gamma_bounds[0] <= gamma <= opt.gamma_bounds[1]):
            return 1e12
        if not (opt.tau_label_ms_bounds[0] <= tau_label_ms <= opt.tau_label_ms_bounds[1]):
            return 1e12

        key = (round(gamma, 5), round(tau_label_ms, 3))
        if key in cache:
            return cache[key]

        dynamic_labeled = add_labeling_and_quench_columns(
            dynamic_base,
            label_rate_dict,
            quench_rate_dict,
            cfg,
            tau_label_ms=tau_label_ms,
        )

        lc_chem = LCChemParams(**vars(lc_chem_template))
        lc_chem.gamma = gamma
        lc_integrals = precalculate_lc_integrals(
            sequences,
            lc,
            lc_chem,
            T_lc_C,
            pH_lc,
            n_steps=opt.lc_n_steps,
        )

        dynamic_lc = add_lc_columns(
            dynamic_labeled,
            lc_integrals,
            gamma_prime_params=gamma_prime_params,
        )

        esi_params = optimize_reduced_esi_for_lc_stage(
            dynamic_lc,
            quench_rate_dict,
            opt,
            fixed_alpha=opt.fixed_alpha_ESI,
            fixed_xi_ESI=opt.fixed_xi_ESI,
        )

        records = prepare_esi_records(dynamic_lc, quench_rate_dict)
        losses = []
        for seq, params in esi_params.items():
            if seq in records:
                losses.append(esi_objective_fast(params, records[seq]))

        esi_loss = float(np.nanmean(losses)) if losses else 1e12
        reg = opt.tau_label_ms_reg_weight * (tau_label_ms / 50.0) ** 2
        total_loss = float(esi_loss + reg)
        cache[key] = total_loss

        print(
            f"Gamma={gamma:.3f}, tau_label_ms={tau_label_ms:.2f}, "
            f"reduced-ESI loss={total_loss:.4f}"
        )
        return total_loss

    return objective


def optimize_global_lc_gamma_and_tau_label(
    dynamic_base: pd.DataFrame,
    sequences: np.ndarray,
    label_rate_dict: dict[str, np.ndarray],
    quench_rate_dict: dict[str, np.ndarray],
    cfg: ExperimentConfig,
    lc: LCParams,
    lc_chem_template: LCChemParams,
    T_lc_C: float,
    pH_lc: float,
    opt: OptimizationConfig,
    gamma_prime_params: dict[str, float] | None = None,
) -> tuple[float, float, float]:
    """Optimize global LC gamma and global effective labeling-time offset."""
    global_objective = make_global_lc_gamma_tau_objective(
        dynamic_base=dynamic_base,
        sequences=sequences,
        label_rate_dict=label_rate_dict,
        quench_rate_dict=quench_rate_dict,
        cfg=cfg,
        lc=lc,
        lc_chem_template=lc_chem_template,
        T_lc_C=T_lc_C,
        pH_lc=pH_lc,
        opt=opt,
        gamma_prime_params=gamma_prime_params,
    )

    bounds = [opt.gamma_bounds, opt.tau_label_ms_bounds]
    starts = [
        [opt.initial_gamma, opt.tau_label_ms_initial],
        [opt.initial_gamma, 10.0],
        [opt.initial_gamma, 25.0],
        [opt.initial_gamma, 50.0],
        [0.0, 10.0],
        [10.0, 25.0],
        [20.0, 25.0],
    ]

    best_result = None
    for x0 in starts:
        x0 = np.asarray(x0, dtype=float)
        x0[0] = np.clip(x0[0], bounds[0][0], bounds[0][1])
        x0[1] = np.clip(x0[1], bounds[1][0], bounds[1][1])

        result = minimize(
            global_objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 80, "ftol": 1e-8},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return float(best_result.x[0]), float(best_result.x[1]), float(best_result.fun)


def lc_envelope_loss(peptide_lc: pd.DataFrame) -> float:
    """
    Keep LC as a plausible upper envelope before ESI.

    ESI can only decrease uptake after LC, so LC values below experiment are
    penalized strongly. LC far above experiment is penalized more softly.
    """
    lc_vals = peptide_lc["Uptake_LC"].to_numpy(dtype=float)
    exp_vals = peptide_lc["Uptake"].to_numpy(dtype=float)

    diff = lc_vals - exp_vals
    below_penalty = np.nanmean(np.minimum(diff, 0.0) ** 2)
    above_penalty = np.nanmean(np.maximum(diff, 0.0) ** 2)

    return float(20.0 * below_penalty + above_penalty)


def optimize_lc_gamma_prime_per_peptide(
    dynamic_base: pd.DataFrame,
    lc: LCParams,
    lc_chem_template: LCChemParams,
    T_lc_C: float,
    pH_lc: float,
    quench_rate_dict: dict[str, np.ndarray],
    opt: OptimizationConfig,
) -> dict[str, float]:
    """
    Optimize one local gamma_prime_p value per peptide with global gamma fixed.

    The objective is not perfectly smooth because each gamma_prime_p trial
    includes an inner restricted ESI fit. To avoid false local minima, scan the
    full allowed range first, then refine around the best grid point.
    """
    gamma_prime_params: dict[str, float] = {}
    n_grid = 81

    for seq, peptide_df in dynamic_base.groupby("Sequence"):
        seq = str(seq).strip()
        peptide_df = peptide_df.reset_index(drop=True)
        cache: dict[float, float] = {}

        def objective(gamma_prime_value) -> float:
            gamma_prime_p = float(np.ravel([gamma_prime_value])[0])
            gamma_prime_key = round(gamma_prime_p, 5)

            if gamma_prime_key in cache:
                return cache[gamma_prime_key]

            if not (opt.gamma_prime_p_bounds[0] <= gamma_prime_p <= opt.gamma_prime_p_bounds[1]):
                return 1e12

            lc_chem = LCChemParams(**vars(lc_chem_template))
            lc_integrals = precalculate_lc_integrals(
                np.array([seq]),
                lc,
                lc_chem,
                T_lc_C,
                pH_lc,
                n_steps=opt.lc_n_steps,
            )

            peptide_lc = add_lc_columns(
                peptide_df,
                lc_integrals,
                gamma_prime_params={seq: gamma_prime_p},
            )

            esi_params = optimize_reduced_esi_for_lc_stage(
                peptide_lc,
                quench_rate_dict,
                opt,
                fixed_alpha=opt.fixed_alpha_ESI,
                fixed_xi_ESI=opt.fixed_xi_ESI,
            )

            records = prepare_esi_records(peptide_lc, quench_rate_dict)
            if seq not in records or seq not in esi_params:
                return 1e12

            loss_lc = lc_envelope_loss(peptide_lc)
            loss_esi = esi_objective_fast(esi_params[seq], records[seq])
            reg = opt.gamma_prime_p_reg_weight * gamma_prime_p**2
            total_loss = float(loss_lc + 0.25 * loss_esi + reg)
            cache[gamma_prime_key] = total_loss

            return total_loss

        low, high = opt.gamma_prime_p_bounds
        gamma_grid = np.linspace(low, high, n_grid)
        grid_losses = np.array([objective(value) for value in gamma_grid], dtype=float)

        if np.all(~np.isfinite(grid_losses)):
            gamma_prime_params[seq] = 0.0
            print(f"{seq} -> gamma_prime_p=0.000, local-LC loss=nan")
            continue

        best_grid_idx = int(np.nanargmin(grid_losses))
        best_grid_gamma = float(gamma_grid[best_grid_idx])
        best_grid_loss = float(grid_losses[best_grid_idx])

        left_idx = max(0, best_grid_idx - 1)
        right_idx = min(n_grid - 1, best_grid_idx + 1)
        refine_bounds = (float(gamma_grid[left_idx]), float(gamma_grid[right_idx]))

        result = minimize_scalar(
            objective,
            bounds=refine_bounds,
            method="bounded",
            options={"xatol": 0.01, "maxiter": 80},
        )

        if np.isfinite(result.fun) and float(result.fun) <= best_grid_loss:
            gamma_prime_opt = float(result.x)
            best_loss = float(result.fun)
        else:
            gamma_prime_opt = best_grid_gamma
            best_loss = best_grid_loss

        gamma_prime_params[seq] = gamma_prime_opt
        print(
            f"{seq} -> gamma_prime_p={gamma_prime_opt:.3f}, "
            f"local-LC loss={best_loss:.4f}, "
            f"grid_best={best_grid_gamma:.3f}"
        )

    return gamma_prime_params


# =============================================================================
# 9. RESULT ARCHIVING / RELOAD HELPERS
# =============================================================================
# These functions save quantities that the existing model already computes.
# They do not change any equation, objective function, rate law, or optimizer.

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


def safe_state_slug(state: str) -> str:
    """Filesystem-safe folder name while keeping the original state in saved tables."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(state).strip())
    return value or "unnamed_state"


def array_to_json(value: object) -> str:
    """Serialize numpy vectors for portable CSV inspection."""
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), separators=(",", ":"))
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return json.dumps(value, separators=(",", ":"))


def build_saved_calculation_columns(
    dynamic: pd.DataFrame,
    label_rate_dict: dict[str, np.ndarray],
    quench_rate_dict: dict[str, np.ndarray],
    lc_integrals: dict[str, Callable[[float], np.ndarray]],
    gamma_prime_params: dict[str, float],
    esi_params: dict[str, tuple[float, float]],
    fixed_alpha: float,
    fixed_xi_ESI: float,
    lc: LCParams,
    lc_chem: LCChemParams,
    T_lc_C: float,
    pH_lc: float,
) -> pd.DataFrame:
    """
    Attach diagnostic/calculation vectors to the final dataframe.

    IMPORTANT: this is archival only. It is called after the final model result is
    obtained and therefore cannot alter the optimization or predicted uptake.
    """
    out = dynamic.copy()

    label_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    quench_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for seq in out["Sequence"].dropna().astype(str).str.strip().unique():
        if seq in label_rate_dict:
            label_cache[seq] = rates_to_k_and_K(label_rate_dict[seq])
        if seq in quench_rate_dict:
            quench_cache[seq] = rates_to_k_and_K(quench_rate_dict[seq])

    def _row_vectors(row: pd.Series) -> pd.Series:
        seq = str(row["Sequence"]).strip()

        k_label, K_label = label_cache.get(seq, (np.array([]), np.array([])))
        k_quench, K_quench = quench_cache.get(seq, (np.array([]), np.array([])))

        if seq in lc_integrals:
            gamma_prime_p = float(gamma_prime_params.get(seq, 0.0))
            peptide_scale = np.exp(np.clip(gamma_prime_p, -80.0, 80.0))

            lc_integral = np.asarray(lc_integrals[seq](row["RT"]), dtype=float)
            lc_integral = lc_integral * peptide_scale

            v_at_rt = lc_gradient(float(row["RT"]), lc)
            k_lc_at_rt = lc_rate_vector(
                v_at_rt,
                seq,
                lc_chem,
                T_lc_C,
                pH_lc,
            ) * peptide_scale
        else:
            lc_integral = np.array([], dtype=float)
            k_lc_at_rt = np.array([], dtype=float)

        D_lc = row.get("Uptake_Vector_LC")
        n_amides = len(D_lc) if isinstance(D_lc, np.ndarray) else len(k_quench)

        if seq in esi_params and n_amides > 0:
            t_ESI, lambda_ESI = esi_params[seq]
            del t_ESI  # saved separately as a scalar column; not needed to construct the rate vector
            Q = local_charge_density_vector(
                seq=seq,
                z=row["z"],
                n_amides=min(n_amides, len(k_quench)),
                alpha=fixed_alpha,
                xi_ESI=fixed_xi_ESI,
            )
            k_esi = esi_rate_vector(
                np.asarray(k_quench[:len(Q)], dtype=float),
                row["z"],
                seq,
                alpha=fixed_alpha,
                xi_ESI=fixed_xi_ESI,
                lambda_ESI=lambda_ESI,
            )
        else:
            Q = np.array([], dtype=float)
            k_esi = np.array([], dtype=float)

        return pd.Series(
            {
                "k_int_label_vector": np.asarray(k_label, dtype=float),
                "K_eq_label_vector": np.asarray(K_label, dtype=float),
                "k_int_quench_vector": np.asarray(k_quench, dtype=float),
                "K_eq_quench_vector": np.asarray(K_quench, dtype=float),
                "LC_Integral_Vector": np.asarray(lc_integral, dtype=float),
                "k_LC_at_RT_vector": np.asarray(k_lc_at_rt, dtype=float),
                "Q_ESI_Vector": np.asarray(Q, dtype=float),
                "k_ESI_vector": np.asarray(k_esi, dtype=float),
            }
        )

    archived_vectors = out.apply(_row_vectors, axis=1)
    for col in archived_vectors.columns:
        out[col] = archived_vectors[col]

    return out


def build_rate_table(
    sequences: np.ndarray,
    label_rate_dict: dict[str, np.ndarray],
    quench_rate_dict: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Human-readable peptide-level intrinsic-rate table."""
    rows = []
    for seq in sequences:
        seq = str(seq).strip()
        label_rates = label_rate_dict.get(seq)
        quench_rates = quench_rate_dict.get(seq)

        if label_rates is not None:
            k_label, K_label = rates_to_k_and_K(label_rates)
        else:
            k_label, K_label = np.array([]), np.array([])

        if quench_rates is not None:
            k_quench, K_quench = rates_to_k_and_K(quench_rates)
        else:
            k_quench, K_quench = np.array([]), np.array([])

        rows.append(
            {
                "Sequence": seq,
                "label_kforw_vector": array_to_json(label_rates[:, 0] if label_rates is not None else []),
                "label_kback_vector": array_to_json(label_rates[:, 1] if label_rates is not None else []),
                "label_k_int_vector": array_to_json(k_label),
                "label_K_eq_vector": array_to_json(K_label),
                "quench_kforw_vector": array_to_json(quench_rates[:, 0] if quench_rates is not None else []),
                "quench_kback_vector": array_to_json(quench_rates[:, 1] if quench_rates is not None else []),
                "quench_k_int_vector": array_to_json(k_quench),
                "quench_K_eq_vector": array_to_json(K_quench),
            }
        )
    return pd.DataFrame(rows)


def save_state_results(
    state: str,
    dynamic: pd.DataFrame,
    metrics_df: pd.DataFrame,
    sequences: np.ndarray,
    label_rate_dict: dict[str, np.ndarray],
    quench_rate_dict: dict[str, np.ndarray],
    gamma_opt: float,
    tau_label_ms_opt: float,
    gamma_loss: float,
    gamma_prime_params: dict[str, float],
    esi_params: dict[str, tuple[float, float]],
    cfg: ExperimentConfig,
    lc: LCParams,
    lc_chem: LCChemParams,
    opt: OptimizationConfig,
    T_lc_C: float,
    pH_lc: float,
    output_cfg: OutputConfig,
) -> Path:
    """Save one salt/state run in a self-contained, organized directory."""
    state_dir = Path(output_cfg.root_dir) / safe_state_slug(state)
    tables_dir = state_dir / "tables"
    optimization_dir = state_dir / "optimization"
    exact_dir = state_dir / "exact_objects"
    tables_dir.mkdir(parents=True, exist_ok=True)
    optimization_dir.mkdir(parents=True, exist_ok=True)
    exact_dir.mkdir(parents=True, exist_ok=True)

    vector_cols = [c for c in VECTOR_COLUMNS if c in dynamic.columns]
    scalar_cols = [c for c in dynamic.columns if c not in vector_cols]

    if output_cfg.save_human_readable_csv:
        dynamic[scalar_cols].to_csv(tables_dir / "row_results_scalar.csv", index=False)

        id_candidates = [
            "Protein", "Sequence", "State", "File", "Exposure", "z", "RT",
            "Uptake", "Uptake_Labeling", "Uptake_Quenched", "Uptake_LC", "Uptake_ESI",
        ]
        id_cols = [c for c in id_candidates if c in dynamic.columns]
        vectors_csv = dynamic[id_cols + vector_cols].copy()
        for col in vector_cols:
            vectors_csv[col] = vectors_csv[col].map(array_to_json)
        vectors_csv.to_csv(tables_dir / "row_results_vectors.csv", index=False)

        metrics_df.to_csv(tables_dir / "fit_metrics.csv", index=False)
        build_rate_table(sequences, label_rate_dict, quench_rate_dict).to_csv(
            tables_dir / "intrinsic_rate_vectors.csv", index=False
        )

        peptide_rows = []
        for seq in sequences:
            seq = str(seq).strip()
            t_esi, lambda_esi = esi_params.get(seq, (np.nan, np.nan))
            peptide_rows.append(
                {
                    "State": state,
                    "Sequence": seq,
                    "gamma_prime_p": gamma_prime_params.get(seq, np.nan),
                    "t_ESI": t_esi,
                    "lambda_ESI": lambda_esi,
                    "alpha_ESI_fixed": opt.fixed_alpha_ESI,
                    "xi_ESI_fixed": opt.fixed_xi_ESI,
                }
            )
        peptide_parameters_df = pd.DataFrame(peptide_rows)
        peptide_parameters_df.to_csv(tables_dir / "peptide_parameters.csv", index=False)

        # -------------------------------------------------------------
        # Dedicated optimization outputs
        # -------------------------------------------------------------
        # These are archival summaries only. They do not participate in
        # optimization and therefore do not change the computational model.

        # Global optimized values.
        global_opt_df = pd.DataFrame(
            [{
                "State": state,
                "gamma_lc_opt": float(gamma_opt),
                "tau_label_ms_opt": float(tau_label_ms_opt),
                "global_gamma_tau_objective": float(gamma_loss),
            }]
        )
        global_opt_df.to_csv(
            optimization_dir / "global_optimization.csv",
            index=False,
        )

        # Peptide -> protein lookup for readable optimization tables.
        protein_lookup = (
            dynamic[["Sequence", "Protein"]]
            .dropna(subset=["Sequence"])
            .drop_duplicates(subset=["Sequence"])
            .assign(Sequence=lambda x: x["Sequence"].astype(str).str.strip())
            .set_index("Sequence")["Protein"]
            .to_dict()
        )

        # Peptide-specific LC optimized values.
        lc_opt_rows = []
        for seq in sequences:
            seq = str(seq).strip()
            lc_opt_rows.append(
                {
                    "State": state,
                    "Protein": protein_lookup.get(seq, ""),
                    "Sequence": seq,
                    "gamma_lc_global": float(gamma_opt),
                    "gamma_prime_p_opt": float(gamma_prime_params.get(seq, np.nan)),
                }
            )

        pd.DataFrame(lc_opt_rows).to_csv(
            optimization_dir / "peptide_lc_optimization.csv",
            index=False,
        )

        # Peptide-specific ESI optimized values and final objective value.
        esi_records = prepare_esi_records(dynamic, quench_rate_dict)
        esi_opt_rows = []

        for seq in sequences:
            seq = str(seq).strip()
            t_esi, lambda_esi = esi_params.get(seq, (np.nan, np.nan))

            if seq in esi_records and np.all(np.isfinite([t_esi, lambda_esi])):
                params4 = [
                    float(opt.fixed_alpha_ESI),
                    float(t_esi),
                    float(lambda_esi),
                    float(opt.fixed_xi_ESI),
                ]
                esi_objective_value = float(
                    esi_objective_fast(params4, esi_records[seq])
                )
            else:
                esi_objective_value = np.nan

            esi_opt_rows.append(
                {
                    "State": state,
                    "Protein": protein_lookup.get(seq, ""),
                    "Sequence": seq,
                    "alpha_ESI_fixed": float(opt.fixed_alpha_ESI),
                    "xi_ESI_fixed": float(opt.fixed_xi_ESI),
                    "t_ESI_opt": float(t_esi),
                    "lambda_ESI_opt": float(lambda_esi),
                    "ESI_objective_at_optimum": esi_objective_value,
                }
            )

        esi_opt_df = pd.DataFrame(esi_opt_rows)
        esi_opt_df.to_csv(
            optimization_dir / "peptide_esi_optimization.csv",
            index=False,
        )

        # One convenient per-peptide file containing all final fitted values.
        combined_opt_df = pd.DataFrame(lc_opt_rows).merge(
            esi_opt_df,
            on=["State", "Protein", "Sequence"],
            how="outer",
        )
        combined_opt_df["tau_label_ms_global"] = float(tau_label_ms_opt)
        combined_opt_df["global_gamma_tau_objective"] = float(gamma_loss)
        combined_opt_df.to_csv(
            optimization_dir / "all_optimized_parameters.csv",
            index=False,
        )

        # Save all bounds, initial values, and fixed optimization settings.
        optimization_settings = {
            "initial_gamma": float(opt.initial_gamma),
            "gamma_bounds": list(opt.gamma_bounds),
            "tau_label_ms_bounds": list(opt.tau_label_ms_bounds),
            "tau_label_ms_initial": float(opt.tau_label_ms_initial),
            "tau_label_ms_reg_weight": float(opt.tau_label_ms_reg_weight),
            "fixed_alpha_ESI": float(opt.fixed_alpha_ESI),
            "fixed_xi_ESI": float(opt.fixed_xi_ESI),
            "t_esi_bounds": list(opt.t_esi_bounds),
            "lambda_esi_bounds": list(opt.lambda_esi_bounds),
            "t_esi_initial": float(opt.t_esi_initial),
            "lambda_esi_initial": float(opt.lambda_esi_initial),
            "lc_n_steps": int(opt.lc_n_steps),
            "gamma_prime_p_bounds": list(opt.gamma_prime_p_bounds),
            "gamma_prime_p_reg_weight": float(opt.gamma_prime_p_reg_weight),
        }
        with open(
            optimization_dir / "optimization_settings.json",
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(optimization_settings, fh, indent=2)

    global_parameters = {
        "State": state,
        "gamma_lc": float(gamma_opt),
        "tau_label_ms": float(tau_label_ms_opt),
        "global_gamma_tau_objective": float(gamma_loss),
        "T_lc_C": float(T_lc_C),
        "pH_lc": float(pH_lc),
        "LC_V0": float(lc.V0),
        "LC_V_TAU": float(lc.V_TAU),
        "LC_TAU_GRADIENT": float(lc.TAU_GRADIENT),
        "LC_eta": float(lc_chem.eta),
        "fixed_alpha_ESI": float(opt.fixed_alpha_ESI),
        "fixed_xi_ESI": float(opt.fixed_xi_ESI),
    }
    with open(state_dir / "global_parameters.json", "w", encoding="utf-8") as fh:
        json.dump(global_parameters, fh, indent=2)

    manifest = {
        "state": state,
        "input_csv": cfg.input_csv,
        "n_rows_dynamic": int(len(dynamic)),
        "n_sequences": int(len(sequences)),
        "files": {
            "row_results_scalar": "tables/row_results_scalar.csv",
            "row_results_vectors": "tables/row_results_vectors.csv",
            "fit_metrics": "tables/fit_metrics.csv",
            "intrinsic_rate_vectors": "tables/intrinsic_rate_vectors.csv",
            "peptide_parameters": "tables/peptide_parameters.csv",
            "global_parameters": "global_parameters.json",
            "global_optimization": "optimization/global_optimization.csv",
            "peptide_lc_optimization": "optimization/peptide_lc_optimization.csv",
            "peptide_esi_optimization": "optimization/peptide_esi_optimization.csv",
            "all_optimized_parameters": "optimization/all_optimized_parameters.csv",
            "optimization_settings": "optimization/optimization_settings.json",
            "full_exact_dataframe": "exact_objects/dynamic_results.pkl",
            "model_artifacts": "exact_objects/model_artifacts.pkl",
        },
        "notes": [
            "CSV vector columns are JSON arrays for portability.",
            "Pickle files preserve numpy arrays exactly and are the fastest way to reload the final result.",
            "LC_Integral_Vector is the integrated LC rate actually used in the LC survival calculation, including gamma_prime_p.",
            "k_LC_at_RT_vector is the instantaneous LC rate vector at that row's retention time, including gamma_prime_p.",
            "k_ESI_vector is the final row/charge-specific ESI rate vector used for ESI decay.",
        ],
    }
    with open(state_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    if output_cfg.save_full_pickle:
        dynamic.to_pickle(exact_dir / "dynamic_results.pkl")
        with open(exact_dir / "model_artifacts.pkl", "wb") as fh:
            pickle.dump(
                {
                    "state": state,
                    "label_rate_dict": label_rate_dict,
                    "quench_rate_dict": quench_rate_dict,
                    "gamma_opt": gamma_opt,
                    "tau_label_ms_opt": tau_label_ms_opt,
                    "gamma_loss": gamma_loss,
                    "gamma_prime_params": gamma_prime_params,
                    "esi_params": esi_params,
                    "metrics_df": metrics_df,
                    "experiment_config": cfg,
                    "lc_params": lc,
                    "lc_chem_params": lc_chem,
                    "optimization_config": opt,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    return state_dir


def load_saved_state_results(root_dir: str, state: str) -> pd.DataFrame:
    """Reload the exact final dataframe for a salt/state without running the model."""
    path = Path(root_dir) / safe_state_slug(state) / "exact_objects" / "dynamic_results.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Saved state result not found: {path}")
    return pd.read_pickle(path)


def load_saved_state_artifacts(root_dir: str, state: str) -> dict[str, object]:
    """Reload fitted parameters and intrinsic rate dictionaries without rerunning optimization."""
    path = Path(root_dir) / safe_state_slug(state) / "exact_objects" / "model_artifacts.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Saved state artifacts not found: {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


# =============================================================================
# 10. PLOTTING
# =============================================================================
def calculate_fit_metrics(
    df: pd.DataFrame,
    exp_col: str = "Uptake",
    pred_col: str = "Uptake_ESI",
) -> dict[str, float]:
    """
    Calculate fit metrics between experimental uptake and simulated ESI uptake.

    RMSE_Da is in absolute deuterium uptake units.
    RMSE_frac is normalized by the maximum experimental uptake for that peptide.
    """
    valid = df.dropna(subset=[exp_col, pred_col]).copy()

    if valid.empty:
        return {
            "N": 0,
            "RMSE_Da": np.nan,
            "RMSE_frac": np.nan,
        }

    exp_vals = valid[exp_col].to_numpy(dtype=float)
    pred_vals = valid[pred_col].to_numpy(dtype=float)

    residual = pred_vals - exp_vals

    rmse_da = float(np.sqrt(np.mean(residual**2)))

    denom = float(np.nanmax(np.abs(exp_vals)))
    rmse_frac = float(rmse_da / denom) if denom > 0 else np.nan

    return {
        "N": int(len(valid)),
        "RMSE_Da": rmse_da,
        "RMSE_frac": rmse_frac,
    }



def plot_peptides(dynamic: pd.DataFrame, target_seqs: list[str], state: str = "") -> None:
    base_config = {
        "Uptake_Labeling": ("black", "*", "-.", "1. Labeling"),
        "Uptake_Quenched": ("orange", "^", "-.", "2. After Quench"),
        "Uptake_LC": ("cyan", "p", "--", "3. After LC"),
    }

    for target_seq in target_seqs:
        df_plot = dynamic[dynamic["Sequence"].eq(target_seq)].copy().sort_values("Exposure")
        if df_plot.empty:
            print(f"No rows found for peptide: {target_seq}")
            continue

        fig, ax = plt.subplots(figsize=(12, 7))

        for col, (color, marker, style, label) in base_config.items():
            if col not in df_plot.columns:
                continue

            valid = df_plot.dropna(subset=[col])
            if valid.empty:
                continue

            ax.scatter(valid["Exposure"], valid[col], color=color, marker=marker, s=60, alpha=0.5, label=label)
            mean_vals = valid.groupby("Exposure")[col].mean()
            ax.plot(mean_vals.index, mean_vals.values, color=color, linestyle=style, linewidth=1.5, alpha=0.7)

        unique_z = sorted(df_plot["z"].dropna().unique())
        colors = cm.viridis(np.linspace(0, 0.8, len(unique_z)))

        for idx, z_val in enumerate(unique_z):
            z_group = df_plot[df_plot["z"].eq(z_val)].dropna(subset=["Uptake", "Uptake_ESI"])
            if z_group.empty:
                continue

            current_color = colors[idx]

            ax.plot(
                z_group["Exposure"],
                z_group["Uptake_ESI"],
                color=current_color,
                linestyle="-",
                linewidth=2.5,
                alpha=0.9,
                label=f"After ESI (z={int(z_val)})",
            )
            ax.scatter(
                z_group["Exposure"],
                z_group["Uptake_ESI"],
                color=current_color,
                marker="D",
                s=70,
                alpha=1.0,
                edgecolor="white",
            )
            ax.scatter(
                z_group["Exposure"],
                z_group["Uptake"],
                facecolors="none",
                edgecolors=current_color,
                marker="o",
                s=80,
                linewidth=1.5,
                alpha=0.8,
                label=f"Exp Data (z={int(z_val)})",
            )

            last_row = z_group.iloc[-1]
            ax.text(
                last_row["Exposure"] * 1.15,
                last_row["Uptake_ESI"],
                f"z={int(z_val)}",
                color=current_color,
                fontweight="bold",
                va="center",
                fontsize=10,
            )

        
        protein_names = (
            df_plot["Protein"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        
        protein_name = protein_names[0] if protein_names else "Unknown protein"
        
        ax.set_title(
            f"Peptides: {protein_name} , {target_seq} | State: {state}",
            fontsize=18,
            fontweight="bold",
            pad=20,
        )
        
        metrics = calculate_fit_metrics(df_plot)

        metric_text = (
            f"RMSE = {metrics['RMSE_frac']:.4f}\n"
        )
        
        ax.text(
            0.04,
            0.96,
            metric_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.15),
        )
        
        ax.set_xscale("log")
        ax.set_xlabel("Exposure Time (ms)", fontsize=14)
        ax.set_ylabel("Deuterium Uptake (Da)", fontsize=14)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.set_xlim(df_plot["Exposure"].min() * 0.8, df_plot["Exposure"].max() * 4.0)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower right",
            ncol=2,
            fontsize=9,
            frameon=True,
            framealpha=0.8,
            edgecolor="lightgray",
        )

        plt.tight_layout()
        plt.show()


# =============================================================================
# 11. SINGLE-STATE WORKFLOW
# =============================================================================

def run_single_state(
    base_cfg: ExperimentConfig,
    state: str,
    output_cfg: OutputConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """
    Run the original model for exactly one salt/state condition.

    The mathematical/computational model below is the same sequence as the
    original main(): labeling -> quench -> global LC/tau optimization -> local
    LC optimization -> final ESI optimization -> final ESI application.
    """
    cfg = replace(base_cfg, state=str(state))
    lc = LCParams()
    lc_chem = LCChemParams()
    opt = OptimizationConfig()

    T_lc_C = 0.0
    pH_lc = 2.55

    print("\n" + "=" * 90)
    print(f"RUNNING STATE / SALT CONDITION: {state}")
    print("=" * 90)

    print("Loading and preparing data...")
    _, m0_matrix, dynamic_matrix = load_and_prepare_data(cfg, state=state)
    if dynamic_matrix.empty:
        raise ValueError(f"No dynamic Exposure > 0 rows found for State={state!r}")

    dynamic_matrix = add_reference_uptakes(dynamic_matrix, m0_matrix, cfg)

    sequences = dynamic_matrix["Sequence"].dropna().astype(str).str.strip().unique()
    print(f"Dynamic rows: {len(dynamic_matrix)}")
    print(f"Unique sequences: {len(sequences)}")

    print("Calculating intrinsic labeling and quench rates...")
    label_rate_dict = build_rate_dict(sequences, cfg.T_label_C, cfg.x_label, cfg.pH_label)
    quench_rate_dict = build_rate_dict(sequences, cfg.T_quench_C, cfg.x_quench, cfg.pH_quench)

    print("Starting joint global optimization for effective labeling time and LC gamma with reduced ESI...")
    gamma_opt, tau_label_ms_opt, gamma_loss = optimize_global_lc_gamma_and_tau_label(
        dynamic_base=dynamic_matrix,
        sequences=sequences,
        label_rate_dict=label_rate_dict,
        quench_rate_dict=quench_rate_dict,
        cfg=cfg,
        lc=lc,
        lc_chem_template=lc_chem,
        T_lc_C=T_lc_C,
        pH_lc=pH_lc,
        opt=opt,
    )
    lc_chem.gamma = gamma_opt

    print(f"Optimal global gamma:        {gamma_opt:.6f}")
    print(f"Optimal tau_label_ms:        {tau_label_ms_opt:.3f} ms")
    print(f"Global gamma/tau objective:  {gamma_loss:.6f}")
    print("=" * 40)

    print("Adding labeling and quench columns with optimized effective labeling time...")
    dynamic_matrix = add_labeling_and_quench_columns(
        dynamic_matrix,
        label_rate_dict,
        quench_rate_dict,
        cfg,
        tau_label_ms=tau_label_ms_opt,
    )

    print("\nOptimizing peptide-specific gamma_prime_p with restricted ESI...")
    gamma_prime_params = optimize_lc_gamma_prime_per_peptide(
        dynamic_base=dynamic_matrix,
        lc=lc,
        lc_chem_template=lc_chem,
        T_lc_C=T_lc_C,
        pH_lc=pH_lc,
        quench_rate_dict=quench_rate_dict,
        opt=opt,
    )
    print("Peptide-specific gamma_prime_p optimization finished.")
    print("=" * 40)

    final_lc_integrals = precalculate_lc_integrals(
        sequences,
        lc,
        lc_chem,
        T_lc_C,
        pH_lc,
        n_steps=opt.lc_n_steps,
    )

    print("\nApplying LC with optimized global gamma and peptide-specific gamma_prime_p...")
    dynamic_matrix = add_lc_columns(
        dynamic_matrix,
        final_lc_integrals,
        gamma_prime_params=gamma_prime_params,
    )

    dynamic_matrix["tau_label_ms"] = float(tau_label_ms_opt)
    dynamic_matrix["gamma_lc"] = float(gamma_opt)
    dynamic_matrix["gamma_prime_p"] = dynamic_matrix["Sequence"].astype(str).str.strip().map(gamma_prime_params)

    print("\nOptimizing ESI parameters per peptide using final LC result...")
    esi_params = optimize_per_peptide_esi(
        dynamic_matrix,
        quench_rate_dict,
        opt,
        fast=False,
        verbose=True,
    )

    dynamic_matrix["alpha_ESI_fixed"] = float(opt.fixed_alpha_ESI)
    dynamic_matrix["xi_ESI_fixed"] = float(opt.fixed_xi_ESI)
    dynamic_matrix["t_ESI"] = dynamic_matrix["Sequence"].astype(str).str.strip().map(
        {seq: vals[0] for seq, vals in esi_params.items()}
    )
    dynamic_matrix["lambda_ESI"] = dynamic_matrix["Sequence"].astype(str).str.strip().map(
        {seq: vals[1] for seq, vals in esi_params.items()}
    )

    print("\nApplying optimized ESI model...")
    dynamic_matrix = apply_esi_columns(
        dynamic_matrix,
        esi_params,
        quench_rate_dict,
        fixed_alpha=opt.fixed_alpha_ESI,
        fixed_xi_ESI=opt.fixed_xi_ESI,
    )

    # Archival-only columns: no optimization or prediction is performed here.
    print("Attaching calculation/rate vectors for archival...")
    dynamic_matrix = build_saved_calculation_columns(
        dynamic=dynamic_matrix,
        label_rate_dict=label_rate_dict,
        quench_rate_dict=quench_rate_dict,
        lc_integrals=final_lc_integrals,
        gamma_prime_params=gamma_prime_params,
        esi_params=esi_params,
        fixed_alpha=opt.fixed_alpha_ESI,
        fixed_xi_ESI=opt.fixed_xi_ESI,
        lc=lc,
        lc_chem=lc_chem,
        T_lc_C=T_lc_C,
        pH_lc=pH_lc,
    )

    metric_rows = []
    for seq, df_seq in dynamic_matrix.groupby("Sequence"):
        metrics = calculate_fit_metrics(df_seq)
        protein_names = df_seq["Protein"].dropna().astype(str).unique().tolist()
        protein_name = protein_names[0] if protein_names else "Unknown protein"
        metric_rows.append(
            {
                "Protein": protein_name,
                "Sequence": seq,
                "State": str(state),
                "N": metrics["N"],
                "RMSE_Da": metrics["RMSE_Da"],
                "RMSE_frac": metrics["RMSE_frac"],
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    dynamic_matrix.attrs["fit_metrics"] = metrics_df.to_dict(orient="records")

    state_dir = save_state_results(
        state=state,
        dynamic=dynamic_matrix,
        metrics_df=metrics_df,
        sequences=sequences,
        label_rate_dict=label_rate_dict,
        quench_rate_dict=quench_rate_dict,
        gamma_opt=gamma_opt,
        tau_label_ms_opt=tau_label_ms_opt,
        gamma_loss=gamma_loss,
        gamma_prime_params=gamma_prime_params,
        esi_params=esi_params,
        cfg=cfg,
        lc=lc,
        lc_chem=lc_chem,
        opt=opt,
        T_lc_C=T_lc_C,
        pH_lc=pH_lc,
        output_cfg=output_cfg,
    )
    print(f"Saved organized results to: {state_dir}")

    if output_cfg.show_plots_during_batch:
        target_seqs = dynamic_matrix["Sequence"].dropna().astype(str).str.strip().unique().tolist()
        plot_peptides(dynamic_matrix, target_seqs, state=str(state))

    run_summary = {
        "State": str(state),
        "gamma_lc": float(gamma_opt),
        "tau_label_ms": float(tau_label_ms_opt),
        "global_gamma_tau_objective": float(gamma_loss),
        "n_rows": int(len(dynamic_matrix)),
        "n_peptides": int(len(sequences)),
        "mean_RMSE_Da": float(metrics_df["RMSE_Da"].mean()) if not metrics_df.empty else np.nan,
        "mean_RMSE_frac": float(metrics_df["RMSE_frac"].mean()) if not metrics_df.empty else np.nan,
    }
    return dynamic_matrix, metrics_df, run_summary


# =============================================================================
# 12. BATCH WORKFLOW: ALL SALT / STATE CONDITIONS
# =============================================================================

def main() -> dict[str, pd.DataFrame]:
    base_cfg = ExperimentConfig()
    output_cfg = OutputConfig()

    states = discover_states(base_cfg)
    if not states:
        raise ValueError("No State values were found in the input CSV.")

    root = Path(output_cfg.root_dir)
    root.mkdir(parents=True, exist_ok=True)

    print("States / salt concentrations that will be processed:")
    for state in states:
        print(f"  - {state}")

    all_results: dict[str, pd.DataFrame] = {}
    all_metrics: list[pd.DataFrame] = []
    run_summaries: list[dict[str, float]] = []

    for state in states:
        dynamic_state, metrics_state, run_summary = run_single_state(
            base_cfg=base_cfg,
            state=state,
            output_cfg=output_cfg,
        )
        all_results[str(state)] = dynamic_state
        all_metrics.append(metrics_state)
        run_summaries.append(run_summary)

    if all_metrics:
        all_metrics_df = pd.concat(all_metrics, ignore_index=True)
        all_metrics_df.to_csv(root / "all_states_fit_metrics.csv", index=False)
    else:
        all_metrics_df = pd.DataFrame()

    run_summary_df = pd.DataFrame(run_summaries)
    run_summary_df.to_csv(root / "all_states_run_summary.csv", index=False)

    # Cross-salt table for direct downstream comparison without vector/object columns.
    scalar_frames = []
    peptide_parameter_frames = []
    for state, df_state in all_results.items():
        vector_cols = [c for c in VECTOR_COLUMNS if c in df_state.columns]

        scalar_df = df_state.drop(columns=vector_cols, errors="ignore").copy()
        scalar_df.attrs.clear()
        scalar_frames.append(scalar_df)

        parameter_cols = [
            c for c in [
                "State", "Protein", "Sequence", "gamma_lc", "tau_label_ms",
                "gamma_prime_p", "alpha_ESI_fixed", "xi_ESI_fixed", "t_ESI", "lambda_ESI"
            ]
            if c in df_state.columns
        ]

        parameter_df = df_state[parameter_cols].drop_duplicates().copy()
        parameter_df.attrs.clear()
        peptide_parameter_frames.append(parameter_df)

    if scalar_frames:
        pd.concat(scalar_frames, ignore_index=True).to_csv(
            root / "all_states_row_results_scalar.csv", index=False
        )
    if peptide_parameter_frames:
        pd.concat(peptide_parameter_frames, ignore_index=True).to_csv(
            root / "all_states_peptide_parameters.csv", index=False
        )

    with open(root / "batch_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "input_csv": base_cfg.input_csv,
                "states_processed": [str(s) for s in states],
                "state_folders": [safe_state_slug(s) for s in states],
                "combined_files": [
                    "all_states_fit_metrics.csv",
                    "all_states_run_summary.csv",
                    "all_states_row_results_scalar.csv",
                    "all_states_peptide_parameters.csv",
                ],
                "reload_example": (
                    "df = load_saved_state_results('hdmx_results_all_salt', '0mM')"
                ),
            },
            fh,
            indent=2,
        )

    print("\n" + "=" * 90)
    print("ALL SALT / STATE CONDITIONS FINISHED")
    print(f"Root results directory: {root.resolve()}")
    print("Combined summary files:")
    print(f"  - {root / 'all_states_fit_metrics.csv'}")
    print(f"  - {root / 'all_states_run_summary.csv'}")
    print(f"  - {root / 'all_states_row_results_scalar.csv'}")
    print(f"  - {root / 'all_states_peptide_parameters.csv'}")
    print("=" * 90)

    return all_results


if __name__ == "__main__":
    all_state_results = main()

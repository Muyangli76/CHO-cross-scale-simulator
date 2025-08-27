# app.py
import re
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp

# =========================
# Core ODE Model (includes TAN + Alanine)
# =========================
def bioreactor_odes_with_scale(t, y, scale_params):
    """
    CHO model with density-dependent death & space limitations,
    plus secondary metabolites: total ammonia (TAN) and alanine.
    States (y): [C_X (cells/mL), C_glc, C_lac, C_Ab, V (mL), C_DO, pCO2, C_NH3, C_Ala]
    """
    # Unpack states
    C_X   = max(1e-6, y[0])                 # cells/mL
    C_glc = max(0.1,  y[1])                 # mmol/L
    C_lac = max(0.0,  y[2])                 # mmol/L
    C_Ab  = max(0.0,  y[3])                 # g/L
    V     = max(1000, y[4])                 # mL
    C_DO  = max(1e-6, min(0.3, y[5]))       # mmol/L
    pCO2  = max(20,   min(500, y[6]))       # mmHg
    C_NH3 = max(0.0,  y[7])                 # mmol/L (TAN)
    C_Ala = max(0.0,  y[8])                 # mmol/L

    # ---------- Core constants ----------
    KIlac = 7.1
    mglc = 82.3e-12
    Qp_base = 9.0e-13
    mu_max_base = 5.17e-2
    kd = 2.32e-2
    Yxglc = 1.33e9
    Ylac_glucose_stoich = 1.8
    Kdglc = 1.54
    Kglc = 1.0
    Kdlac = 2.0
    DO_sat = 0.25

    # ---------- Scale & engineering ----------
    K_max_base = 25e6
    volume_factor   = scale_params.get("volume_factor", 1.0)
    kLa_base        = scale_params.get("kLa", 200)
    pCO2_threshold  = scale_params.get("pCO2_threshold", 150)
    mixing_eff      = scale_params.get("mixing_eff", 1.0)
    EDR             = scale_params.get("EDR", 0.05)
    bubble_EDR      = scale_params.get("bubble_EDR", 1e4)
    antifoam_enabled= scale_params.get("antifoam", True)
    antifoam_type   = scale_params.get("antifoam_type", "PEG")

    # Carrying capacity & growth cutoff by scale
    if volume_factor < 0.1:      # 2L
        K_max = K_max_base * 1.05
        growth_cutoff = 0.60
        death_mult = 1.5
    elif volume_factor < 2:      # 50L
        K_max = K_max_base * 1.00
        growth_cutoff = 0.65
        death_mult = 1.2
    else:                        # 2000L
        K_max = K_max_base * 0.80
        growth_cutoff = 0.70
        death_mult = 1.0

    # kLa with antifoam effects
    if antifoam_enabled:
        kLa = kLa_base * 0.9
        if antifoam_type == "silicone":
            kLa *= 0.95
    else:
        kLa = kLa_base

    # ---------- Temperature shift ----------
    temp_shift_time = scale_params.get("temp_shift_time", 72)
    if t > temp_shift_time:
        Qp = Qp_base * 1.35
        mu_max = mu_max_base * 0.65
        kd_temp = kd * 0.9
        lactate_shift_enhanced = True
    else:
        Qp = Qp_base
        mu_max = mu_max_base
        kd_temp = kd
        lactate_shift_enhanced = False

    # ---------- Feed profile ----------
    cin = scale_params.get("cin", 722)  # mmol/L glucose in feed
    profile = scale_params.get("feed_profile", None)  # [(start_h, vvd)]
    if profile:
        profile = sorted(profile, key=lambda p: p[0])
        vvd = 0.0
        for start, rate in profile:
            if t >= start:
                vvd = rate
            else:
                break
    else:
        # default
        if t < 60:
            vvd = 0.0
        elif t < 150:
            vvd = 0.05
        elif t < 220:
            vvd = 0.06
        else:
            vvd = 0.065
    Fin = vvd * V / 24.0  # mL/h

    # ---------- Growth & death control ----------
    density_factor = C_X / K_max
    if density_factor < growth_cutoff:
        f_carry = 1.0
    elif density_factor < 1.0:
        f_carry = (1.0 - density_factor) / (1.0 - growth_cutoff)
    else:
        f_carry = 0.0

    if density_factor > 0.70:
        f_death_density = 1 + death_mult * 10 * (density_factor - 0.70) / 0.30
    elif density_factor > 0.50:
        f_death_density = 1 + death_mult * 3 * (density_factor - 0.50) / 0.20
    else:
        f_death_density = 1.0

    # pCO2 effects
    if pCO2 > 100:
        f_CO2_growth = 0.0
    elif pCO2 > 80:
        f_CO2_growth = (100 - pCO2) / 20
    else:
        f_CO2_growth = 1.0

    if pCO2 > 150:
        f_CO2_qp = 0.6
    elif pCO2 > 120:
        f_CO2_qp = 1.0 - 0.4 * (pCO2 - 120) / 30
    else:
        f_CO2_qp = 1.0

    # DO & shear
    f_DO = C_DO / (C_DO + 0.02)
    f_shear_growth = 0.06 / EDR if EDR > 0.06 else 1.0
    if EDR > 0.4:
        f_shear_qp = 0.7
    elif EDR > 0.06:
        f_shear_qp = 1.0 - 0.3 * (EDR - 0.06) / 0.34
    else:
        f_shear_qp = 1.0

    shear_death_factor = 1 + (bubble_EDR / 1e6 - 1) * 0.5 if bubble_EDR > 1e6 else 1.0
    nutrient_gradient_factor = 0.5 + 0.5 * mixing_eff

    # ---------- Base growth/productivity ----------
    C_glc_safe = max(C_glc, 0.1)
    C_lac_safe = max(C_lac, 1e-6)

    mu_max_eff = max(
        0.001,
        mu_max * f_carry * f_CO2_growth * f_DO * f_shear_growth * nutrient_gradient_factor
    )
    mu = max(0, mu_max_eff * (C_glc_safe / (Kglc + C_glc_safe)) * (KIlac / (KIlac + C_lac_safe)))

    mu_sat = mu / (mu + 0.02)
    qP_shape = 0.40 + 0.60 * mu_sat

    Qp_with_antifoam = Qp * (1.1 if (antifoam_enabled and antifoam_type == "PEG") else 1.0)
    Qp_eff = max(1e-15, Qp_with_antifoam * f_CO2_qp * f_DO * f_shear_qp * qP_shape)

    # ---------- Secondary metabolites (production & penalties) ----------
    y_NH3_glc       = scale_params.get("y_NH3_glc", 0.05)
    y_NH3_growth    = scale_params.get("y_NH3_growth", 0.0)
    k_strip_NH3     = scale_params.get("k_strip_NH3", 0.01)
    K_NH3_g         = scale_params.get("K_NH3_g", 4.0)
    K_NH3_qp        = scale_params.get("K_NH3_qp", 6.0)
    use_ala_penalty = scale_params.get("ala_penalty", False)
    y_Ala_lac       = scale_params.get("y_Ala_lac", 0.05)
    k_Ala_lowDO     = scale_params.get("k_Ala_lowDO", 2e-12)
    q_Ala_cons_max  = scale_params.get("q_Ala_cons_max", 1e-12)
    K_Ala           = scale_params.get("K_Ala", 1.0)
    K_Ala_g         = scale_params.get("K_Ala_g", 20.0)
    K_Ala_qp        = scale_params.get("K_Ala_qp", 20.0)

    f_NH3_growth = K_NH3_g  / (K_NH3_g  + C_NH3)
    f_NH3_qp     = K_NH3_qp / (K_NH3_qp + C_NH3)
    if use_ala_penalty:
        f_Ala_growth = K_Ala_g  / (K_Ala_g  + C_Ala)
        f_Ala_qp     = K_Ala_qp / (K_Ala_qp + C_Ala)
    else:
        f_Ala_growth = 1.0
        f_Ala_qp     = 1.0

    mu_max_eff *= f_NH3_growth * f_Ala_growth
    Qp_eff     *= f_NH3_qp     * f_Ala_qp

    # Recompute mu with updated mu_max_eff
    mu = max(0, mu_max_eff * (C_glc_safe / (Kglc + C_glc_safe)) * (KIlac / (KIlac + C_lac_safe)))

    # Death rate
    mu_d = max(0, kd_temp * shear_death_factor * f_death_density *
               (1 + 0.5 * max(density_factor - 0.6, 0) / 0.4) *
               (C_lac_safe / (Kdlac + C_lac_safe)) * (Kdglc / (Kdglc + C_glc_safe)))

    # Uptake/production fluxes
    glucose_consumption_rate = ((mu - mu_d) / Yxglc + mglc) * C_X
    lactate_production_rate  = Ylac_glucose_stoich * glucose_consumption_rate
    lactate_shift_inhibited  = pCO2 > pCO2_threshold
    lactate_consumption_conditions = (not lactate_shift_inhibited and (t > 96 or lactate_shift_enhanced) and C_lac > 2.0)
    if lactate_consumption_conditions:
        lactate_consumption_rate = 20e-12 * C_X * (C_lac / (0.5 + C_lac))
        if lactate_shift_enhanced:
            lactate_consumption_rate *= 1.8
    else:
        lactate_consumption_rate = 0.0

    # Oxygen/CO2 balances
    qO2 = max(1e-15, 6.0 * ((mu - mu_d) / Yxglc + mglc))
    OTR = max(0, kLa * (DO_sat - C_DO))
    OUR = max(0, qO2 * C_X)
    qCO2 = max(1e-15, 1.2 * ((mu - mu_d) / Yxglc + mglc))
    volume_effect = 1 + volume_factor * (V / 50000)
    CO2_prod = max(0, qCO2 * C_X * volume_effect)
    if volume_factor > 2:
        stripping_efficiency = 0.1
    elif volume_factor > 0.5:
        stripping_efficiency = 0.2
    else:
        stripping_efficiency = 0.25
    CO2_strip = max(0, kLa * stripping_efficiency * max(0, (pCO2 - 40) / 100))

    # ---------- NEW: TAN & Ala dynamics ----------
    overflow_lac = max(lactate_production_rate - lactate_consumption_rate, 0.0)
    r_NH3  = y_NH3_glc   * glucose_consumption_rate + y_NH3_growth * max(mu - mu_d, 0) * C_X
    r_Ala  = y_Ala_lac   * overflow_lac + k_Ala_lowDO * (1 - f_DO) * C_X
    uptake_Ala = q_Ala_cons_max * C_X * C_Ala / (K_Ala + C_Ala)

    # ---------- ODEs ----------
    dC_X_dt   = max(-C_X/10, (mu - mu_d) * C_X - Fin / V * C_X)
    dC_glc_dt = -glucose_consumption_rate + Fin / V * (cin - C_glc)
    dC_lac_dt = lactate_production_rate - lactate_consumption_rate - Fin / V * C_lac
    dC_Ab_dt  = max(0, Qp_eff * C_X * 1000.0 - Fin / V * C_Ab)
    dV_dt     = max(0, Fin)
    dC_DO_dt  = OTR - OUR - Fin / V * C_DO
    dpCO2_dt  = CO2_prod - CO2_strip - Fin / V * pCO2
    dC_NH3_dt = r_NH3 - k_strip_NH3 * C_NH3 - Fin / V * C_NH3
    dC_Ala_dt = r_Ala - uptake_Ala - Fin / V * C_Ala

    return [dC_X_dt, dC_glc_dt, dC_lac_dt, dC_Ab_dt, dV_dt, dC_DO_dt, dpCO2_dt, dC_NH3_dt, dC_Ala_dt]

# =========================
# Helpers: sampling + noise
# =========================
def _attach_sampling_markers(df, t_end_hours, enabled=False, mode="Lab", offline_hours=24, online_minutes=15):
    if not enabled:
        return df
    t = df["time"].values

    def _nearest_indices(step_hours):
        targets = np.arange(0.0, t_end_hours + step_hours, step_hours)
        idx = np.unique([int(np.abs(t - tau).argmin()) for tau in targets])
        return idx

    offline_idx = _nearest_indices(float(offline_hours))
    online_idx  = _nearest_indices(float(online_minutes) / 60.0) if mode == "Lab" else np.arange(len(t))

    offline_vars = ["C_X","C_glc","C_lac","C_Ab","C_NH3","C_Ala"]
    online_vars  = ["C_DO","pCO2"]
    for var in offline_vars + online_vars:
        df[f"{var}_meas"] = np.nan
    for var in offline_vars:
        df.loc[offline_idx, f"{var}_meas"] = df.loc[offline_idx, var]
    for var in online_vars:
        df.loc[online_idx,  f"{var}_meas"] = df.loc[online_idx,  var]

    df["offline_sample"] = False; df.loc[offline_idx, "offline_sample"] = True
    df["online_sample"]  = False; df.loc[online_idx,  "online_sample"]  = True
    df["sampling_mode"] = "Hybrid (Lab + Continuous Online)" if mode == "Hybrid" else "Realistic Lab Sampling"
    df["offline_interval_hrs"] = offline_hours
    df["online_interval_mins"] = online_minutes if mode == "Lab" else 0
    return df

def _apply_measurement_noise(df, enable=False, cv_offline=0.05, cv_online=0.02, seed=42):
    if not enable:
        return df
    rng = np.random.default_rng(seed)
    meas_cols = [c for c in df.columns if c.endswith("_meas")]
    for col in meas_cols:
        vals = df[col].to_numpy()
        mask = ~np.isnan(vals)
        if mask.sum() == 0:
            continue
        is_online = col.startswith("C_DO") or col.startswith("pCO2")
        cv = cv_online if is_online else cv_offline
        noise = rng.normal(loc=0.0, scale=cv, size=mask.sum())
        perturbed = vals[mask] * (1.0 + noise)
        perturbed = np.clip(perturbed, 0.0, 0.30) if col.startswith("C_DO") else np.clip(perturbed, 0.0, None)
        out_col = col.replace("_meas", "_meas_noisy")
        df[out_col] = np.nan
        df.loc[mask, out_col] = perturbed
    return df

# =========================
# Simulation wrapper
# =========================
def run_simulation(batch_name, scale, process_params, advanced_params=None):
    scale_params = {
        "temp_shift_time": process_params.get("temp_shift_time", 72),
        "antifoam": process_params.get("antifoam", True),
        "antifoam_type": process_params.get("antifoam_type", "PEG"),
        "feed_profile": process_params.get("feed_profile", None),
        "cin": process_params.get("feed_glucose", 722),
        "ala_penalty": process_params.get("ala_penalty", False),
    }
    if scale == "2L":
        scale_params.update({"kLa": 280,"pCO2_threshold": 200,"mixing_eff": 1.0,"volume_factor": 0.02,"EDR": 0.02,"bubble_EDR": 1e4})
    elif scale == "50L":
        scale_params.update({"kLa": 90,"pCO2_threshold": 100,"mixing_eff": 0.90,"volume_factor": 1.0,"EDR": 0.05,"bubble_EDR": 2e5})
    else:
        scale_params.update({"kLa": 35,"pCO2_threshold": 60,"mixing_eff": 0.65,"volume_factor": 4.0,"EDR": 0.15,"bubble_EDR": 2e6})
        scale_params["antifoam_type"] = "silicone"
    if advanced_params:
        scale_params.update(advanced_params)

    # Time & initial conditions
    t_start, t_end = 0, 14*24
    time_points = np.linspace(t_start, t_end, 200)
    inoc_cells_per_ml = process_params.get("inoculation_density", 0.8e6)
    init_V_mL = {"2L":2000, "50L":50000, "2000L":2000000}[scale]
    initial_conditions = [inoc_cells_per_ml, 41, 0, 0, init_V_mL, 0.20, 40, 0, 0]  # + NH3, Ala

    try:
        sol = solve_ivp(
            lambda t, y: bioreactor_odes_with_scale(t, y, scale_params),
            [t_start, t_end], initial_conditions, t_eval=time_points,
            method='LSODA', rtol=1e-4, atol=1e-8, max_step=2.0
        )
        if not sol.success:
            st.error(f"Simulation failed: {sol.message}")
            return None

        cols = ["C_X","C_glc","C_lac","C_Ab","V","C_DO","pCO2","C_NH3","C_Ala"]
        df = pd.DataFrame(np.vstack(sol.y).T, columns=cols)
        df["time"] = sol.t

        df["C_X_raw_cells_per_ml"] = df["C_X"].copy()
        df["C_X"] /= 1e6  # display as 10^6 cells/mL

        # Realistic sampling markers & noise
        if process_params.get("sampling_enabled", False):
            df = _attach_sampling_markers(
                df, t_end_hours=t_end, enabled=True,
                mode=("Hybrid" if process_params.get("sampling_mode") == "Hybrid" else "Lab"),
                offline_hours=int(process_params.get("offline_interval", 24)),
                online_minutes=int(process_params.get("online_interval", 15)),
            )
            df = _apply_measurement_noise(
                df,
                enable=process_params.get("noise_enabled", False),
                cv_offline=float(process_params.get("noise_cv_offline", 0.05)),
                cv_online=float(process_params.get("noise_cv_online", 0.02)),
                seed=int(process_params.get("noise_seed", 42)),
            )

        # Sanity checks
        assert df["C_Ab"].iloc[-1] < 8.0, f"Titer exploded: {df['C_Ab'].iloc[-1]:.2f} g/L (should be < 8)"
        assert df["C_X"].max() < 50, f"Peak VCD too high: {df['C_X'].max():.1f} ×10⁶/mL (should be < 50)"

        # Metadata
        df["batch_name"] = batch_name
        df["scale"] = scale
        for k, v in process_params.items():
            if k in ("feed_profile",):  # stringify separately
                continue
            if isinstance(v, bool):
                df[k] = "Yes" if v else "No"
            elif np.isscalar(v) or isinstance(v, str):
                df[k] = v
            else:
                df[k] = str(v)
        fp = process_params.get("feed_profile")
        df["feed_profile"] = "; ".join(f"{int(s)}h@{r:.3f}/d" for s, r in sorted(fp, key=lambda x: x[0])) if fp else \
                              "default: 0h@0.000, 60h@0.050, 150h@0.060, 220h@0.065"
        return df

    except Exception as e:
        st.error(f"Simulation error: {str(e)}")
        return None

# =========================
# Metrics
# =========================
def calculate_summary_metrics(df, tan_thresh=4.0):
    if df is None or df.empty:
        return {}
    days = df["time"].iloc[-1] / 24.0
    if "C_X_raw_cells_per_ml" in df.columns:
        ivcd_cells_per_L_day = np.trapz(df["C_X_raw_cells_per_ml"] * 1000.0, df["time"] / 24.0)
        qP_pg_per_cell_day = (df["C_Ab"].iloc[-1]) / ivcd_cells_per_L_day * 1e12 if ivcd_cells_per_L_day > 0 else 0.0
    else:
        qP_pg_per_cell_day = (df["C_Ab"].iloc[-1] * 1000) / (df["C_X"].mean() * days + 1e-9)

    max_TAN = df["C_NH3"].max() if "C_NH3" in df.columns else 0.0
    time_TAN_high = 0.0
    if "C_NH3" in df.columns:
        t = df["time"].values
        mask = df["C_NH3"].values > tan_thresh
        if mask.any():
            time_TAN_high = np.trapz(mask.astype(float), t)  # ~hours above threshold

    return {
        "final_titer": df["C_Ab"].iloc[-1],
        "peak_VCD": df["C_X"].max(),
        "final_VCD": df["C_X"].iloc[-1],
        "max_pCO2": df["pCO2"].max(),
        "min_DO": df["C_DO"].min(),
        "lactate_shift_success": df["C_lac"].iloc[-1] < df["C_lac"].max() - 3,
        "growth_arrest": df["pCO2"].max() > 100,
        "cell_specific_productivity": qP_pg_per_cell_day,
        "max_TAN_mM": max_TAN,
        "hours_TAN_above_thresh": time_TAN_high,
        "final_Ala_mM": df["C_Ala"].iloc[-1] if "C_Ala" in df.columns else 0.0
    }

# =========================
# CSV parsing helpers
# =========================
def _safe_float(x, default=None):
    try:
        if pd.isna(x): return default
        return float(x)
    except Exception:
        return default

def infer_kla_from_do(scale, do_pct):
    """Only used if 'kLa' not provided. Lightweight, adjustable mapping."""
    if do_pct is None:
        return None
    def pick(mapping, v):
        key = min(mapping.keys(), key=lambda k: abs(k - v))
        return mapping[key]
    if str(scale) == "2L":
        return pick({40: 225, 60: 280, 80: 335}, do_pct)
    elif str(scale) == "50L":
        return pick({40: 70, 60: 90, 80: 110}, do_pct)
    else:  # 2000L
        return pick({40: 25, 60: 35, 80: 45}, do_pct)

def build_feed_profile_from_row(row):
    """
    Supports:
      A) explicit schedule:
         num_feed_phases, phase1_start_h, phase1_vvd_per_day, phase2_start_h, ...
      B) simple one-phase:
         feed_start_day, (optional) phase2_* columns
      C) default -> None (uses model default)
      Optional: feed_VVD_relative multiplier applies to all phases.
    """
    rel = _safe_float(row.get("feed_VVD_relative"), 1.0)
    # A) explicit schedule
    n = int(_safe_float(row.get("num_feed_phases"), 0) or 0)
    phases = []
    if n > 0:
        for i in range(1, n + 1):
            sh = _safe_float(row.get(f"phase{i}_start_h"))
            vv = _safe_float(row.get(f"phase{i}_vvd_per_day"))
            if sh is not None and vv is not None:
                phases.append((int(sh), float(vv) * rel))
        if phases:
            return sorted(phases, key=lambda x: x[0])
    # B) simple one/two-phase
    start_day = _safe_float(row.get("feed_start_day"))
    if start_day is not None:
        start_h = int(start_day * 24)
        phases = [(start_h, 0.050 * rel)]
        sh2 = _safe_float(row.get("phase2_start_h"))
        vv2 = _safe_float(row.get("phase2_vvd_per_day"))
        if sh2 is not None and vv2 is not None:
            phases.append((int(sh2), float(vv2) * rel))
        return sorted(phases, key=lambda x: x[0])
    # C) default
    return None

def parse_row_to_params(row):
    """Convert a CSV row (Series) into (batch_name, scale, process_params, advanced_params)."""
    # identifiers
    batch_name = str(row.get("BatchName", row.get("Global_Run_Order", "Run")))

    scale = str(row.get("Scale", "2L")).strip()
    if scale not in {"2L", "50L", "2000L"}:
        if "2000" in scale: scale = "2000L"
        elif "50" in scale: scale = "50L"
        else: scale = "2L"

    inoc_million = _safe_float(row.get("inoc_density_1e6_per_mL",
                                       row.get("inoc_dens", 0.35)), 0.35)
    inoculation_density = inoc_million * 1e6

    temp_shift_time = int(_safe_float(row.get("temp_shift_time_h",
                                             row.get("temp_shif", 72)), 72))
    pH_setpoint     = _safe_float(row.get("pH_setpoint"), 7.0)
    agitation_rate  = _safe_float(row.get("agitation_rate_rpm"), 100.0)
    pO2_setpoint    = _safe_float(row.get("DO_setpoint_pct"), 60.0)

    antifoam_type = str(row.get("antifoam_type", "PEG")).strip()
    antifoam = (antifoam_type.lower() != "none")

    feed_glucose = _safe_float(row.get("cin_mM", row.get("feed_glucose_mM", row.get("cin", 722))), 722)

    feed_profile = build_feed_profile_from_row(row)

    process_params = {
        "inoculation_density": inoculation_density,
        "temp_shift_time": temp_shift_time,
        "pH_setpoint": pH_setpoint,
        "agitation_rate": agitation_rate,
        "pO2_setpoint": pO2_setpoint,
        "antifoam": antifoam,
        "antifoam_type": antifoam_type if antifoam else "PEG",
        "feed_profile": feed_profile,
        "feed_glucose": feed_glucose,
        "ala_penalty": False,
        "sampling_enabled": True,
        "sampling_mode": "Lab",
        "offline_interval": 24,
        "online_interval": 15,
        "noise_enabled": False,
        "noise_cv_offline": 0.05,
        "noise_cv_online": 0.02,
        "noise_seed": 42,
    }

    # advanced overrides
    advanced_params = {}
    kLa_csv = _safe_float(row.get("kLa"))
    do_pct  = _safe_float(row.get("DO_setpoint_pct"))
    if kLa_csv is not None:
        advanced_params["kLa"] = float(kLa_csv)
    else:
        kla_guess = infer_kla_from_do(scale, do_pct)
        if kla_guess is not None:
            advanced_params["kLa"] = float(kla_guess)

    return str(batch_name), scale, process_params, advanced_params

# =========================
# NEW: Validation + Batch runner + DoE generator
# =========================
SCALE_PRESETS = {
    "2L": {
        "inoc_density_1e6_per_mL": (0.30, 1.20),
        "temp_shift_time_h": (48, 120),
        "pH_setpoint": (6.80, 7.40),
        "DO_setpoint_pct": (20, 60),
        "kLa": (20, 400),
        "phase1_start_h": (24, 96),
        "phase1_vvd_per_day": (0.010, 0.040),
        "phase2_start_h": (84, 180),
        "phase2_vvd_per_day": (0.015, 0.060),
        "cin_mM": (200, 3000),
        "delta_phase_h_min": 24
    },
    "50L": {
        "inoc_density_1e6_per_mL": (0.40, 1.00),
        "temp_shift_time_h": (48, 96),
        "pH_setpoint": (6.90, 7.10),
        "DO_setpoint_pct": (40, 60),
        "kLa": (5, 15),
        "phase1_start_h": (36, 60),
        "phase1_vvd_per_day": (0.015, 0.035),
        "phase2_start_h": (96, 144),
        "phase2_vvd_per_day": (0.020, 0.045),
        "cin_mM": (1000, 3000),
        "delta_phase_h_min": 24
    },
    "2000L": {
        "inoc_density_1e6_per_mL": (0.40, 1.00),
        "temp_shift_time_h": (48, 96),
        "pH_setpoint": (6.90, 7.10),
        "DO_setpoint_pct": (40, 60),
        "kLa": (2, 8),
        "phase1_start_h": (36, 60),
        "phase1_vvd_per_day": (0.010, 0.030),
        "phase2_start_h": (96, 144),
        "phase2_vvd_per_day": (0.015, 0.035),
        "cin_mM": (1000, 3000),
        "delta_phase_h_min": 24
    }
}

def validate_row(row, scale):
    """Return (ok, warnings, errors)."""
    s = SCALE_PRESETS.get(scale, SCALE_PRESETS["2L"])
    warnings, errors = [], []
    def _chk(name, val):
        if val is None: return
        lo, hi = s[name]
        if not (lo <= val <= hi):
            warnings.append(f"{name}={val} outside {lo}-{hi}")

    # scalars
    _chk("pH_setpoint", _safe_float(row.get("pH_setpoint")))
    _chk("DO_setpoint_pct", _safe_float(row.get("DO_setpoint_pct")))
    _chk("kLa", _safe_float(row.get("kLa")))
    _chk("inoc_density_1e6_per_mL", _safe_float(row.get("inoc_density_1e6_per_mL", row.get("inoc_dens"))))
    _chk("temp_shift_time_h", _safe_float(row.get("temp_shift_time_h", row.get("temp_shif"))))
    # feed
    p1s = _safe_float(row.get("phase1_start_h"));     _chk("phase1_start_h", p1s)
    p1v = _safe_float(row.get("phase1_vvd_per_day")); _chk("phase1_vvd_per_day", p1v)
    p2s = _safe_float(row.get("phase2_start_h"));     _chk("phase2_start_h", p2s)
    p2v = _safe_float(row.get("phase2_vvd_per_day")); _chk("phase2_vvd_per_day", p2v)
    if p1s is not None and p2s is not None:
        if p2s < p1s + s["delta_phase_h_min"]:
            errors.append(f"phase2_start_h must be ≥ phase1_start_h + {s['delta_phase_h_min']}h")
    return (len(errors) == 0), warnings, errors

def lhs(n_samples, n_dims, rng):
    H = np.zeros((n_samples, n_dims))
    for j in range(n_dims):
        cut = np.linspace(0, 1, n_samples + 1)
        u = rng.uniform(size=n_samples)
        pts = cut[:-1] + (cut[1:] - cut[:-1]) * u
        rng.shuffle(pts)
        H[:, j] = pts
    return H

def generate_doe(scale, n_runs, seed, ranges, feed_rel_policy="Tertiles", antifoam_mode="Alternate", cin_mM=2220):
    """Return a DataFrame with your template headers, ready to download."""
    rng = np.random.default_rng(seed)
    # factor keys in order
    cont_keys = ["pH_setpoint","DO_setpoint_pct","kLa",
                 "inoc_density_1e6_per_mL","temp_shift_time_h",
                 "phase1_start_h","phase1_vvd_per_day",
                 "phase2_start_h","phase2_vvd_per_day"]
    low = np.array([ranges[k][0] for k in cont_keys])
    high= np.array([ranges[k][1] for k in cont_keys])
    X = lhs(n_runs, len(cont_keys), rng)
    M = low + (high - low) * X
    df = pd.DataFrame(M, columns=cont_keys)

    # practical rounding
    df["pH_setpoint"] = df["pH_setpoint"].round(2)
    df["DO_setpoint_pct"] = df["DO_setpoint_pct"].round(0).astype(int)
    df["kLa"] = df["kLa"].round(1)
    df["inoc_density_1e6_per_mL"] = df["inoc_density_1e6_per_mL"].round(2)
    df["temp_shift_time_h"] = df["temp_shift_time_h"].round(0).astype(int)
    df["phase1_start_h"] = df["phase1_start_h"].round(0).astype(int)
    df["phase2_start_h"] = np.maximum(df["phase2_start_h"], df["phase1_start_h"] + ranges.get("delta_phase_h_min",24))
    df["phase2_start_h"] = df["phase2_start_h"].clip(ranges["phase2_start_h"][0], ranges["phase2_start_h"][1]).round(0).astype(int)
    df["phase1_vvd_per_day"] = df["phase1_vvd_per_day"].round(3)
    df["phase2_vvd_per_day"] = df["phase2_vvd_per_day"].round(3)

    # categorical / constants
    df["Scale"] = scale
    df["num_feed_phases"] = 2
    df["Design"] = f"LHS-{n_runs}"
    df["Phase"] = "fed-batch"
    df["cin_mM"] = int(cin_mM)

    # antifoam
    if antifoam_mode == "PEG only":
        df["antifoam_type"] = "PEG"
    elif antifoam_mode == "silicone only":
        df["antifoam_type"] = "silicone"
    else:
        df["antifoam_type"] = ["PEG" if i % 2 == 0 else "silicone" for i in range(n_runs)]

    # feed_VVD_relative
    if feed_rel_policy == "Fixed 1.0":
        rel = np.full(n_runs, 1.0)
    elif feed_rel_policy == "Cycle 0.8/1.0/1.2":
        rel = np.array([0.8,1.0,1.2] * (n_runs // 3 + 1))[:n_runs]
    else:
        mean_vvd = (df["phase1_vvd_per_day"] + df["phase2_vvd_per_day"]) / 2.0
        q1, q2 = np.quantile(mean_vvd, [1/3, 2/3])
        rel = mean_vvd.apply(lambda x: 0.8 if x <= q1 else (1.0 if x <= q2 else 1.2)).values
    df["feed_VVD_relative"] = rel

    # template headers + IDs
    df.insert(0, "Global_Run_Order", np.arange(1, n_runs + 1))
    df.insert(1, "BatchName", [f"R{scale.replace('L','')}_{i:03d}" for i in range(1, n_runs + 1)])
    # Map inoc_dens alias to keep legacy compatibility
    df["inoc_dens"] = df["inoc_density_1e6_per_mL"]

    # final column order (template-friendly)
    cols = [
        "Global_Run_Order","BatchName","Scale",
        "pH_setpoint","DO_setpoint_pct","kLa",
        "inoc_dens","temp_shift_time_h",
        "num_feed_phases",
        "phase1_start_h","phase1_vvd_per_day",
        "phase2_start_h","phase2_vvd_per_day",
        "feed_VVD_relative","cin_mM",
        "Design","Phase","antifoam_type"
    ]
    return df[cols]

def run_batch(doe_df, apply_overrides=None):
    """Run all valid rows of a DoE dataframe. Returns (results_list, as_run_rows_df, problems_df)."""
    results, rows_run, problems = [], [], []
    for idx, row in doe_df.iterrows():
        batch_name, scale, pp, adv = parse_row_to_params(row)
        ok, warns, errs = validate_row(row, scale)
        if not ok:
            problems.append({"row_index": idx, "BatchName": batch_name, "Scale": scale,
                             "errors": "; ".join(errs), "warnings": "; ".join(warns)})
            continue
        # apply simple overrides to all rows if provided
        if apply_overrides:
            pp.update(apply_overrides)
        df = run_simulation(f"{batch_name}_{scale}", scale, pp, adv if len(adv) else None)
        if df is not None:
            results.append(df)
            rows_run.append(row)
    as_run = pd.DataFrame(rows_run) if rows_run else pd.DataFrame(columns=doe_df.columns)
    prob_df = pd.DataFrame(problems) if problems else pd.DataFrame(columns=["row_index","BatchName","Scale","errors","warnings"])
    return results, as_run, prob_df

# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config(page_title="CHO Cross-scale Fed-Batch Bioprocess DoE Simulator", page_icon="🧬", layout="wide")
    st.title("CHO Cross-scale Fed-Batch Bioprocess DoE Simulator")
    st.markdown("**Death-focused VCD control with preserved productivity** + TAN & Alanine")

    # Remember last selected mode if we programmatically switch
    default_mode = st.session_state.get("default_mode", "Single Run Simulation")
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Single Run Simulation", "DoE CSV Runner", "DoE Generator"],
        index=["Single Run Simulation", "DoE CSV Runner", "DoE Generator"].index(default_mode)
    )

    # =========================
    # Mode 1: Manual single-run UI
    # =========================
    if mode == "Single Run Simulation":
        st.header("CHO Bioprocess Simulation")
        left, right = st.columns([1, 2])

        with left:
            batch_name = st.text_input("Batch Name", value="CHO_Run")

            st.subheader("Scale Selection")
            scales_to_run = st.multiselect("Select Scales to Run", ["2L","50L","2000L"], default=["2L","50L","2000L"])
            if not scales_to_run:
                st.warning("Please select at least one scale")

            st.subheader("Process Parameters")
            inoculation_density = st.slider("Inoculation Density (×10⁶ cells/mL)", 0.3, 1.2, 0.8)
            temp_shift_time = st.slider("Temperature Shift Time (hours)", 48, 120, 72)
            pH_setpoint = st.slider("pH Setpoint", 6.8, 7.4, 7.0)
            agitation_rate = st.slider("Agitation Rate (rpm)", 50, 200, 100)
            pO2_setpoint = st.slider("pO₂ Setpoint (%)", 20, 80, 60)

            antifoam = st.checkbox("Use Antifoam", value=True)
            antifoam_type = st.selectbox("Antifoam Type", ["PEG", "silicone"]) if antifoam else "PEG"

            with st.expander("Feed Profile (VVD schedule)"):
                use_custom_feed = st.checkbox("Customize feed schedule", value=True)
                feed_glucose = st.number_input("Feed glucose concentration, cin (mmol/L)", 100, 3000, 722, step=10)
                if use_custom_feed:
                    n_phases = st.number_input("Number of feed phases", 1, 2, 2, step=1)
                    default_starts = [60, 150]
                    default_vvds   = [0.05, 0.065]
                    feed_profile = []
                    last_start = 0
                    for i in range(n_phases):
                        start = st.number_input(f"Phase {i+1} start hour", 0, 336, default_starts[i] if i < len(default_starts) else last_start+60, step=6)
                        vvd   = st.number_input(f"Phase {i+1} VVD (1/day)", 0.0, 0.2, float(default_vvds[i] if i < len(default_vvds) else default_vvds[-1]), step=0.005, format="%.3f")
                        feed_profile.append((int(start), float(vvd)))
                        last_start = int(start)
                    feed_profile = sorted(feed_profile, key=lambda p: p[0])
                else:
                    feed_profile = None

            with st.expander("Secondary metabolites (TAN & Alanine)"):
                ala_penalty = st.checkbox("Apply alanine penalty to growth/productivity", value=False)

            with st.expander("Realistic Measurement Frequency"):
                sampling_enabled = st.checkbox("Enable measurement markers", value=True)
                if sampling_enabled:
                    sampling_mode = st.radio("Mode", ["Lab (offline+online at intervals)", "Hybrid (offline sampled, online continuous)"], index=0)
                    sampling_mode_token = "Lab" if sampling_mode.startswith("Lab") else "Hybrid"
                    offline_interval = st.number_input("Offline lab sampling interval (hours)", 12, 48, 24, step=6)
                    online_interval  = st.number_input("Online sensor sampling interval (minutes)", 5, 60, 15, step=5)
                else:
                    sampling_mode_token, offline_interval, online_interval = "Lab", 24, 15

            with st.expander("Measurement Noise (applied to measured points only)"):
                noise_enabled = st.checkbox("Add measurement noise", value=False)
                noise_cv_offline = st.slider("Offline CV (%, VCD/titer/metabolites)", 0.0, 20.0, 5.0, step=0.5) / 100.0
                noise_cv_online  = st.slider("Online CV (%, DO/pCO₂)", 0.0, 10.0, 2.0, step=0.5) / 100.0
                noise_seed       = st.number_input("Noise seed", 0, 9999, 42, step=1)

            with st.expander("Advanced Engineering Parameters"):
                st.caption("⚠️ Engineering parameters - modify only for model validation")
                enable_advanced = st.checkbox("Enable Advanced Parameter Override")
                advanced_params = {}
                if enable_advanced:
                    advanced_params = {
                        "kLa": st.slider("kLa Override (1/h)", 20, 400, 200),
                        "EDR": st.slider("EDR Override (W/kg)", 0.01, 0.5, 0.05),
                        "mixing_eff": st.slider("Mixing Efficiency", 0.4, 1.0, 0.85),
                        "volume_factor": st.slider("Volume Factor", 0.01, 5.0, 1.0)
                    }

            if st.button("Run Simulation", type="primary"):
                if scales_to_run:
                    process_params = {
                        "inoculation_density": inoculation_density * 1e6,
                        "temp_shift_time": temp_shift_time,
                        "pH_setpoint": pH_setpoint,
                        "agitation_rate": agitation_rate,
                        "pO2_setpoint": pO2_setpoint,
                        "antifoam": antifoam,
                        "antifoam_type": antifoam_type,
                        "feed_profile": feed_profile,
                        "feed_glucose": feed_glucose,
                        "ala_penalty": ala_penalty,
                        "sampling_enabled": sampling_enabled,
                        "sampling_mode": sampling_mode_token,
                        "offline_interval": int(offline_interval),
                        "online_interval": int(online_interval),
                        "noise_enabled": noise_enabled,
                        "noise_cv_offline": float(noise_cv_offline),
                        "noise_cv_online": float(noise_cv_online),
                        "noise_seed": int(noise_seed),
                    }
                    results = []
                    progress = st.progress(0)
                    for i, scale in enumerate(scales_to_run):
                        df = run_simulation(f"{batch_name}_{scale}", scale, process_params, advanced_params if enable_advanced else None)
                        if df is not None:
                            results.append(df)
                        progress.progress((i + 1) / len(scales_to_run))
                    if results:
                        st.session_state["results"] = results
                        st.session_state["batch_name"] = batch_name
                        st.success("Simulation completed!")

        _render_results_panel()

    # =========================
    # Mode 2: DoE CSV Runner (single row OR entire file)
    # =========================
    elif mode == "DoE CSV Runner":
        st.header("Run from DoE .csv")
        up = st.file_uploader("Upload DoE CSV", type=["csv"])
        if up:
            doe = pd.read_csv(up)
            st.write("Detected columns:", list(doe.columns))

            # Execution mode
            exec_mode = st.radio("Execution Mode", ["Single row", "Entire file"], horizontal=True)

            # Common: pick ID col
            id_col = None
            for c in ["Global_Run_Order", "Run", "ID", "BatchName"]:
                if c in doe.columns:
                    id_col = c; break
            if id_col is None:
                id_col = doe.columns[0]

            if exec_mode == "Single row":
                selected = st.selectbox("Select a row to run", options=doe[id_col].tolist())
                row = doe.set_index(id_col).loc[selected]
                _, scale_preview, pp_preview, adv_preview = parse_row_to_params(row)
                st.caption(f"Scale: {scale_preview} | Feed profile: "
                           f"{'; '.join([f'{s}h@{v:.3f}/d' for s, v in (pp_preview.get('feed_profile') or [])]) or 'default'} | "
                           f"kLa override: {adv_preview.get('kLa','(auto/default)')}")

                with st.expander("Sampling & Noise for this run"):
                    pp_preview["sampling_enabled"] = st.checkbox("Enable measurement markers", value=True, key="single_sampling")
                    if pp_preview["sampling_enabled"]:
                        pp_preview["sampling_mode"] = "Lab" if st.radio("Mode", ["Lab", "Hybrid"], index=0, key="single_mode") == "Lab" else "Hybrid"
                        pp_preview["offline_interval"] = st.number_input("Offline sampling (hours)", 12, 48, int(pp_preview["offline_interval"]), step=6, key="single_off")
                        pp_preview["online_interval"]  = st.number_input("Online sampling (minutes)", 5, 60, int(pp_preview["online_interval"]), step=5, key="single_on")
                    pp_preview["noise_enabled"]    = st.checkbox("Add measurement noise", value=False, key="single_noise")
                    pp_preview["noise_cv_offline"] = st.slider("Offline CV (%)", 0.0, 20.0, 5.0, step=0.5, key="single_offcv") / 100.0
                    pp_preview["noise_cv_online"]  = st.slider("Online CV (%)", 0.0, 10.0, 2.0, step=0.5, key="single_oncv") / 100.0
                    pp_preview["noise_seed"]       = st.number_input("Noise seed", 0, 9999, 42, step=1, key="single_seed")

                if st.button("Run this row", type="primary"):
                    batch_name, scale, _, adv = parse_row_to_params(row)
                    process_params = pp_preview
                    df = run_simulation(f"{batch_name}_{scale}", scale, process_params, adv if len(adv) else None)
                    if df is not None:
                        st.session_state["results"] = [df]
                        st.session_state["batch_name"] = batch_name
                        st.success(f"Completed: {batch_name} @ {scale}")

            else:  # Entire file
                st.info("Optional: apply the same sampling/noise overrides to every row.")
                overrides = {}
                with st.expander("Batch Overrides"):
                    use_over = st.checkbox("Apply overrides to all rows", value=True)
                    if use_over:
                        overrides["sampling_enabled"] = st.checkbox("Enable measurement markers", value=True, key="all_sampling")
                        if overrides["sampling_enabled"]:
                            mode_token = "Lab" if st.radio("Mode", ["Lab", "Hybrid"], index=0, key="all_mode") == "Lab" else "Hybrid"
                            overrides["sampling_mode"] = mode_token
                            overrides["offline_interval"] = int(st.number_input("Offline sampling (hours)", 12, 48, 24, step=6, key="all_off"))
                            overrides["online_interval"]  = int(st.number_input("Online sampling (minutes)", 5, 60, 15, step=5, key="all_on"))
                        overrides["noise_enabled"]    = st.checkbox("Add measurement noise", value=False, key="all_noise")
                        overrides["noise_cv_offline"] = float(st.slider("Offline CV (%)", 0.0, 20.0, 5.0, step=0.5, key="all_offcv") / 100.0)
                        overrides["noise_cv_online"]  = float(st.slider("Online CV (%)", 0.0, 10.0, 2.0, step=0.5, key="all_oncv") / 100.0)
                        overrides["noise_seed"]       = int(st.number_input("Noise seed", 0, 9999, 42, step=1, key="all_seed"))
                    else:
                        overrides = None

                if st.button("Run entire CSV", type="primary"):
                    with st.spinner("Running all valid rows..."):
                        results, as_run_df, problems_df = run_batch(doe, apply_overrides=overrides)
                        if results:
                            st.session_state["results"] = results
                            st.session_state["batch_name"] = "DoE_Batch"
                            st.success(f"Completed {len(results)} runs.")
                        if not as_run_df.empty:
                            st.download_button("⬇️ Download As-Run Input CSV", data=as_run_df.to_csv(index=False),
                                               file_name="as_run_input.csv", mime="text/csv")
                        if not problems_df.empty:
                            st.warning("Some rows were skipped due to validation errors.")
                            st.dataframe(problems_df, use_container_width=True)

        _render_results_panel()

    # =========================
    # Mode 3: DoE Generator (simple & template-perfect)
    # =========================
    else:
        st.header("DoE Generator")
        colA, colB, colC = st.columns(3)
        with colA:
            scale = st.selectbox("Scale", ["2L","50L","2000L"])
            n_runs = st.number_input("# of runs", 5, 200, 35, step=5)
            seed = st.number_input("Random seed", 0, 999999, 123, step=1)
        with colB:
            feed_rel_policy = st.selectbox("feed_VVD_relative policy", ["Tertiles", "Fixed 1.0", "Cycle 0.8/1.0/1.2"])
            antifoam_mode = st.selectbox("Antifoam assignment", ["Alternate", "PEG only", "silicone only"])
            cin_mM = st.number_input("Feed glucose (mM)", 100, 3000, 2220, step=10)
        with colC:
            delta_min = st.number_input("Min gap: phase2_start − phase1_start (h)", 12, 72, 24, step=6)

        st.subheader("Factor ranges")
        p = SCALE_PRESETS[scale].copy()
        p["delta_phase_h_min"] = delta_min

        def rng(label, lo, hi, default_lo=None, default_hi=None, step=None):
            a = st.number_input(f"{label} min", value=float(default_lo if default_lo is not None else lo), step=step or 0.01)
            b = st.number_input(f"{label} max", value=float(default_hi if default_hi is not None else hi), step=step or 0.01)
            return (min(a,b), max(a,b))

        c1, c2 = st.columns(2)
        with c1:
            p["inoc_density_1e6_per_mL"] = rng("Inoc density (×10⁶/mL)", *p["inoc_density_1e6_per_mL"])
            p["pH_setpoint"]             = rng("pH setpoint", *p["pH_setpoint"])
            p["DO_setpoint_pct"]         = rng("DO setpoint (%)", *p["DO_setpoint_pct"], step=1.0)
            p["kLa"]                     = rng("kLa (1/h)", *p["kLa"])
            p["cin_mM"]                  = rng("cin (mM)", *p["cin_mM"], step=10.0)
        with c2:
            p["temp_shift_time_h"]       = rng("Temp shift time (h)", *p["temp_shift_time_h"], step=1.0)
            p["phase1_start_h"]          = rng("Phase 1 start (h)", *p["phase1_start_h"], step=1.0)
            p["phase1_vvd_per_day"]      = rng("Phase 1 VVD (/d)", *p["phase1_vvd_per_day"])
            p["phase2_start_h"]          = rng("Phase 2 start (h)", *p["phase2_start_h"], step=1.0)
            p["phase2_vvd_per_day"]      = rng("Phase 2 VVD (/d)", *p["phase2_vvd_per_day"])

        if st.button("Generate DoE CSV", type="primary"):
            df = generate_doe(scale, int(n_runs), int(seed), p, feed_rel_policy, antifoam_mode, cin_mM=int(cin_mM[0] if isinstance(cin_mM, tuple) else int(cin_mM)))
            st.session_state["generated_doe"] = df
            st.success("Generated!")
            st.dataframe(df.head(10), use_container_width=True)
            st.download_button("⬇️ Download CSV", data=df.to_csv(index=False), file_name=f"DoE_{scale}_{n_runs}_runs.csv", mime="text/csv")

        if "generated_doe" in st.session_state and st.button("Send to CSV Runner"):
            st.session_state["uploaded_doe_df"] = st.session_state["generated_doe"].copy()
            st.session_state["default_mode"] = "DoE CSV Runner"
            st.experimental_rerun()

# =========================
# Shared rendering panel (metrics, plots, downloads)
# =========================
def _render_results_panel():
    right = st.container()
    with right:
        if 'results' in st.session_state:
            results = st.session_state['results']
            batch_name = st.session_state.get('batch_name', 'CHO_Run')

            st.subheader("📊 Summary Metrics")
            metrics_rows = []
            for df in results:
                m = calculate_summary_metrics(df)
                metrics_rows.append({
                    "Scale": df["scale"].iloc[0],
                    "Final Titer (g/L)": f"{m['final_titer']:.2f}",
                    "Peak VCD (10⁶/mL)": f"{m['peak_VCD']:.1f}",
                    "Max pCO₂ (mmHg)": f"{m['max_pCO2']:.1f}",
                    "Min DO (mmol/L)": f"{m['min_DO']:.3f}",
                    "Max TAN (mM)": f"{m['max_TAN_mM']:.2f}",
                    "Hours TAN > 4 mM": f"{m['hours_TAN_above_thresh']:.1f}",
                    "Final Alanine (mM)": f"{m['final_Ala_mM']:.2f}",
                    "Lactate Shift": "Yes" if m['lactate_shift_success'] else "No",
                    "Growth Arrest": "Warning" if m['growth_arrest'] else "No",
                })
            metrics_df = pd.DataFrame(metrics_rows)
            st.dataframe(metrics_df, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            final_titers = [calculate_summary_metrics(df)['final_titer'] for df in results]
            peak_vcds    = [calculate_summary_metrics(df)['peak_VCD'] for df in results]
            max_pco2s    = [calculate_summary_metrics(df)['max_pCO2'] for df in results]
            growths      = sum([calculate_summary_metrics(df)['growth_arrest'] for df in results])
            c1.metric("Best Titer", f"{max(final_titers):.2f} g/L")
            c2.metric("Max VCD", f"{max(peak_vcds):.1f} ×10⁶/mL")
            c3.metric("Max pCO₂", f"{max(max_pco2s):.0f} mmHg")
            c4.metric("Growth Arrests", f"{growths}/{len(results)}")

            st.subheader("📈 Time Series Analysis")
            fig = make_subplots(
                rows=4, cols=3,
                subplot_titles=[
                    "Viable Cells (10⁶/mL)", "Antibody Titer (g/L)", "Glucose (mmol/L)",
                    "Lactate (mmol/L)", "Dissolved O₂ (mmol/L)", "pCO₂ (mmHg)",
                    "Volume (mL)", "Cell-Specific Productivity", "Glucose Consumption Rate",
                    "TAN (mmol/L)", "Alanine (mmol/L)", ""
                ],
                specs=[
                    [{"secondary_y": False}]*3,
                    [{"secondary_y": False}]*3,
                    [{"secondary_y": False}]*3,
                    [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]
                ]
            )
            colors = {"2L":"#1f77b4","50L":"#2ca02c","2000L":"#d62728"}

            for df in results:
                scale = df["scale"].iloc[0]; color = colors.get(scale, "#444")
                df['cell_specific_productivity'] = df['C_Ab'] * 1000 / (df['C_X'] * df['time'] / 24 + 1e-6)
                df['glucose_consumption_rate'] = -df['C_glc'].diff() / df['time'].diff()
                variables = [
                    ('C_X',1,1),('C_Ab',1,2),('C_glc',1,3),
                    ('C_lac',2,1),('C_DO',2,2),('pCO2',2,3),
                    ('V',3,1),('cell_specific_productivity',3,2),('glucose_consumption_rate',3,3),
                    ('C_NH3',4,1),('C_Ala',4,2)
                ]
                for var,row,col in variables:
                    if var in df.columns:
                        fig.add_trace(go.Scatter(x=df["time"], y=df[var], name=f"{scale}",
                                                 line=dict(color=color), showlegend=(row==1 and col==1)),
                                      row=row, col=col)

            show_markers = any(c.endswith("_meas") for c in results[0].columns)
            if show_markers:
                for df in results:
                    measured_map = [
                        ("C_X",1,1),("C_Ab",1,2),("C_glc",1,3),
                        ("C_lac",2,1),("C_DO",2,2),("pCO2",2,3),
                        ("C_NH3",4,1),("C_Ala",4,2)
                    ]
                    for base,row,col in measured_map:
                        noisy = f"{base}_meas_noisy"
                        clean = f"{base}_meas"
                        use_col = noisy if noisy in df.columns and df[noisy].notna().any() else clean
                        if use_col in df.columns and df[use_col].notna().any():
                            fig.add_trace(
                                go.Scatter(
                                    x=df.loc[df[use_col].notna(),"time"],
                                    y=df.loc[df[use_col].notna(),use_col],
                                    name=f"{df['scale'].iloc[0]} (meas)",
                                    mode="markers",
                                    marker=dict(size=6),
                                    showlegend=False
                                ), row=row, col=col
                            )

            for r in range(1,5):
                for c in range(1,4):
                    fig.add_vline(x=72, line_dash="dash", line_color="black", opacity=0.5, row=r, col=c)
                    fig.add_vline(x=96, line_dash="dot",  line_color="purple", opacity=0.5, row=r, col=c)

            fig.update_layout(height=1200, showlegend=True,
                              title_text="Comprehensive CHO Bioprocess Analysis (with TAN & Alanine)")
            fig.update_xaxes(title_text="Time (h)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("💾 Download Results")
            colA, colB = st.columns(2)
            combined_df = pd.concat(results, ignore_index=True)
            safe_name = re.sub(r"[^\w\-.]+", "_", batch_name)
            combined_df[batch_name] = batch_name
            metrics_out = pd.DataFrame(metrics_rows); metrics_out[batch_name] = batch_name

            with colA:
                st.download_button("📄 Download Complete Dataset",
                    data=combined_df.to_csv(index=False), file_name=f"{safe_name}.csv", mime="text/csv")
            with colB:
                st.download_button("📊 Download Summary Metrics",
                    data=metrics_out.to_csv(index=False), file_name=f"{safe_name}_summary.csv", mime="text/csv")

if __name__ == "__main__":
    main()

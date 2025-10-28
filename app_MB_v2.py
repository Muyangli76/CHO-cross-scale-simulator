# app.py
import re
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp

# --- numpy integration compatibility (np.trapezoid on new, np.trapz on old) ---
if hasattr(np, "trapezoid"):
    _trapezoid = np.trapezoid
else:
    _trapezoid = np.trapz

# =========================
# Small helpers / globals
# =========================
DO_SAT_GLOBAL = 0.21  # mmol/L ~100% air saturation @ 37°C (model constant)
ML_PER_L = 1000.0     # cells/mL -> cells/L
B_CO2_EFF_DEFAULT = 1.2  # mmol/L per mmHg (effective carbonate buffer capacity)

def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)

def _safe_float(x, default=None):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def _gauss_window(t: float, center: float, width: float) -> float:
    x = (t - center) / max(1e-6, width)
    return np.exp(-0.5 * x * x)

# =========================
# NEW: pH model (Tier-A) & overlay→CO2 strip mapper
# =========================
PH_REF = 7.00
ALPHA_PCO2 = 0.0025   # ΔpH per mmHg above 40 (tune if needed)
ALPHA_LAC  = 0.0100   # ΔpH per mmol/L lactate above LAC_REF_MM
LAC_REF_MM = 2.0

def ph_multiplier(pH, center=7.00, width=0.20, floor=0.70):
    x = (pH - center) / max(1e-6, width)
    return float(max(floor, np.exp(-0.5 * x * x)))

def estimate_pH_from_CO2_Lac(pCO2_mmHg, C_lac_mM, pH_set=7.00):
    dpco2 = max(0.0, float(pCO2_mmHg) - 40.0)
    dlac  = max(0.0, float(C_lac_mM) - LAC_REF_MM)
    return float(pH_set - ALPHA_PCO2 * dpco2 - ALPHA_LAC * dlac)

def kla_co2_from_overlay(kLa_effective, overlay_vvm, base_strip_CO2, kLa_CO2_factor):
    """Map overlay airflow to an effective CO2 stripping coefficient (mmHg/h)."""
    gain = 1.0 + 2.0 * (overlay_vvm / (0.03 + overlay_vvm))  # saturates ~×3 near 0.1 vvm
    return float(kLa_effective * base_strip_CO2 * kLa_CO2_factor * gain)

# =========================
# Scale-aware lactate params
# =========================
def _lactate_params(volume_factor: float):
    if volume_factor < 0.1:  # ~2 L
        return dict(G_th=10.0, mu_th=0.024, prod_cap=2.6e-11,
                    age_tau=32.0, biom_K=3.5e6, cons_base=2.1e-12,
                    cons_start=72.0, cons_ramp=24.0)
    elif volume_factor < 2.0:  # ~50 L
        return dict(G_th=11.5, mu_th=0.028, prod_cap=2.4e-11,
                    age_tau=38.0, biom_K=4.5e6, cons_base=2.3e-12,
                    cons_start=78.0, cons_ramp=32.0)
    else:  # ~2000 L
        return dict(G_th=9.0, mu_th=0.022, prod_cap=3.6e-11,
                    age_tau=44.0, biom_K=1.8e6, cons_base=1.6e-12,
                    cons_start=96.0, cons_ramp=56.0)

# =========================
# Core ODE Model (with TAN + Alanine + NEW pH effects)
# =========================
def bioreactor_odes_with_scale(t, y, scale_params):
    C_X   = max(1e-6, y[0])  # cells/mL
    C_glc = max(0.05, y[1])  # mmol/L
    C_lac = max(0.0, y[2])   # mmol/L
    C_Ab  = max(0.0, y[3])   # g/L
    V     = max(1000, y[4])  # mL
    C_DO  = max(1e-6, y[5])  # mmol/L
    pCO2  = max(20.0, y[6])  # mmHg
    C_NH3 = max(0.0, y[7])   # mmol/L
    C_Ala = max(0.0, y[8])   # mmol/L

    KIlac = 7.1
    mglc = 1.5e-11
    Qp_base = 9.0e-13
    mu_max_base = 5.17e-2
    kd = 2.32e-2
    Yxglc = 1.6e8
    Kdglc = 1.54
    Kglc = 1.0
    Kdlac = 2.0
    DO_sat = scale_params.get("DO_sat", DO_SAT_GLOBAL)

    K_max_base = 25e6
    volume_factor = scale_params.get("volume_factor", 1.0)
    kLa_base = scale_params.get("kLa", 200.0)  # 1/h
    mixing_eff = scale_params.get("mixing_eff", 1.0)
    EDR = scale_params.get("EDR", 0.05)
    bubble_EDR = scale_params.get("bubble_EDR", 1e4)
    antifoam_enabled = scale_params.get("antifoam", True)
    antifoam_type = scale_params.get("antifoam_type", "PEG")

    if volume_factor < 0.1:  # 2L
        K_max = K_max_base * 1.05
        growth_cutoff = 0.60
        death_mult = 1.5
    elif volume_factor < 2:  # 50L
        K_max = K_max_base * 1.00
        growth_cutoff = 0.65
        death_mult = 1.2
    else:  # 2000L
        K_max = K_max_base * 0.80
        growth_cutoff = 0.70
        death_mult = 1.0

    if antifoam_enabled:
        kLa = kLa_base * 0.90
        if antifoam_type == "silicone":
            kLa *= 0.95
    else:
        kLa = kLa_base

    if volume_factor < 0.1:
        tau_probe_h = 0.03
        mix_ramp_tau_h = 2.0
    elif volume_factor < 2.0:
        tau_probe_h = 0.06
        mix_ramp_tau_h = 5.0
    else:
        tau_probe_h = 0.12
        mix_ramp_tau_h = 12.0

    temp_shift_time = scale_params.get("temp_shift_time", 72.0)
    if t > temp_shift_time:
        Qp = Qp_base * 1.35
        mu_max = mu_max_base * 0.65
        kd_temp = kd * 0.90
        lactate_shift_enhanced = True
    else:
        Qp = Qp_base
        mu_max = mu_max_base
        kd_temp = kd
        lactate_shift_enhanced = False

    cin = scale_params.get("cin", 722.0)
    profile = scale_params.get("feed_profile", None)

    if profile:
        profile = sorted(profile, key=lambda p: p[0])
        vvd = 0.0
        for start, rate in profile:
            if t >= start:
                vvd = rate
            else:
                break
    else:
        if t < 60:
            vvd = 0.0
        elif t < 150:
            vvd = 0.05
        elif t < 220:
            vvd = 0.06
        else:
            vvd = 0.065

    glc_target = scale_params.get("glc_target_mM", 20.0)
    if C_glc > glc_target:
        ratio = max(1.0, C_glc / glc_target)
        k_guard = scale_params.get("feed_guard_k", 1.0)
        vvd *= max(0.15, ratio**(-k_guard))

    glc_floor = scale_params.get("glc_floor_mM", 18.0)
    if C_glc < glc_floor:
        vvd = min(0.08, vvd * (1.0 + 0.3 * (glc_floor - C_glc) / glc_floor))

    Fin = vvd * V / 24.0  # mL/h

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

    if pCO2 <= 80:
        f_CO2_growth = 1.0
    elif pCO2 >= 120:
        f_CO2_growth = 0.6
    else:
        f_CO2_growth = 1.0 - 0.4 * (pCO2 - 80.0) / 40.0

    if pCO2 <= 100:
        f_CO2_qp = 1.0
    elif pCO2 >= 150:
        f_CO2_qp = 0.6
    else:
        f_CO2_qp = 1.0 - 0.4 * (pCO2 - 100.0) / 50.0

    f_DO = _clamp01(C_DO / (C_DO + 0.02))
    f_shear_growth = _clamp01(1.0 - 0.8 * max(0.0, EDR - 0.06) / 0.44)
    f_shear_qp = _clamp01(1.0 - 0.3 * max(0.0, EDR - 0.06) / 0.34)
    shear_death_factor = 1 + (bubble_EDR / 1e6 - 1) * 0.5 if bubble_EDR > 1e6 else 1.0
    nutrient_gradient_factor = 0.5 + 0.5 * mixing_eff

    C_glc_safe = max(C_glc, 0.05)
    C_lac_safe = max(C_lac, 1e-6)

    # --- NEW: pH estimate & multipliers (algebraic, not a state) ---
    pH_set = scale_params.get("pH_setpoint", 7.0)
    pH_est = estimate_pH_from_CO2_Lac(pCO2, C_lac, pH_set=pH_set)
    f_pH_growth = ph_multiplier(pH_est, center=7.00, width=0.18, floor=0.75)
    f_pH_qp     = ph_multiplier(pH_est, center=7.05, width=0.20, floor=0.80)
    f_pH_lac    = ph_multiplier(pH_est, center=6.95, width=0.22, floor=0.85)

    mu_max_eff = max(0.001, mu_max * f_carry * f_CO2_growth * f_DO * f_shear_growth * nutrient_gradient_factor)
    mu_max_eff *= f_pH_growth  # NEW pH effect on growth potential
    mu = max(0.0, mu_max_eff * (C_glc_safe / (Kglc + C_glc_safe)) * (KIlac / (KIlac + C_lac_safe)))
    mu_sat = mu / (mu + 0.02)
    qP_shape = 0.40 + 0.60 * mu_sat
    Qp_with_antifoam = Qp * (1.1 if (antifoam_enabled and antifoam_type == "PEG") else 1.0)
    Qp_eff = max(1e-15, Qp_with_antifoam * f_CO2_qp * f_DO * f_shear_qp * qP_shape)
    Qp_eff *= f_pH_qp  # NEW pH effect on productivity

    y_NH3_glc = scale_params.get("y_NH3_glc", 0.05)
    y_NH3_growth = scale_params.get("y_NH3_growth", 0.0)
    k_strip_NH3 = scale_params.get("k_strip_NH3", 0.01)
    K_NH3_g = scale_params.get("K_NH3_g", 4.0)
    K_NH3_qp = scale_params.get("K_NH3_qp", 6.0)

    use_ala_penalty = scale_params.get("ala_penalty", False)
    y_Ala_lac = scale_params.get("y_Ala_lac", 0.05)
    k_Ala_lowDO = scale_params.get("k_Ala_lowDO", 2e-12)
    q_Ala_cons_max = scale_params.get("q_Ala_cons_max", 1e-12)
    K_Ala = scale_params.get("K_Ala", 1.0)
    K_Ala_g = scale_params.get("K_Ala_g", 20.0)
    K_Ala_qp = scale_params.get("K_Ala_qp", 20.0)

    f_NH3_growth = K_NH3_g / (K_NH3_g + C_NH3)
    f_NH3_qp = K_NH3_qp / (K_NH3_qp + C_NH3)
    if use_ala_penalty:
        f_Ala_growth = K_Ala_g / (K_Ala_g + C_Ala)
        f_Ala_qp = K_Ala_qp / (K_Ala_qp + C_Ala)
    else:
        f_Ala_growth = 1.0
        f_Ala_qp = 1.0

    mu_max_eff *= f_NH3_growth * f_Ala_growth
    Qp_eff *= f_NH3_qp * f_Ala_qp
    mu = max(0.0, mu_max_eff * (C_glc_safe / (Kglc + C_glc_safe)) * (KIlac / (KIlac + C_lac_safe)))

    mu_d = max(0.0, kd_temp *
               shear_death_factor *
               f_death_density *
               (1 + 0.5 * max(density_factor - 0.6, 0) / 0.4) *
               (C_lac_safe / (Kdlac + C_lac_safe)) *
               (Kdglc / (Kdglc + C_glc_safe)))

    q_glc = ((mu - mu_d) / Yxglc + mglc)
    glucose_consumption_rate = max(0.0, q_glc * C_X * ML_PER_L)

    lacp = _lactate_params(volume_factor)
    G_th, mu_th = lacp["G_th"], lacp["mu_th"]
    f_age = 1.0 - np.exp(- (t / max(1.0, lacp["age_tau"]))**2)
    f_biom = C_X / (C_X + lacp["biom_K"])
    overflow_gate = (_clamp01((C_glc - G_th)/max(G_th,1e-6)) *
                     _clamp01((mu - mu_th)/max(mu_th,1e-6)) *
                     f_age * f_biom)

    pco2_term = _clamp01((pCO2 - 40.0) / 80.0)
    lowDO_term = (1.0 - _clamp01(C_DO / (C_DO + 0.02)))
    qLac_prod = 0.45 * ((mu - mu_d) / Yxglc + mglc) * overflow_gate
    qLac_prod *= (1.0 + 0.35*pco2_term + 0.20*lowDO_term)
    qLac_prod *= (1.0 + 0.10 * (1.0 - f_pH_lac))  # NEW: low pH slightly boosts overflow ≤ +10%
    qLac_prod = min(qLac_prod, lacp["prod_cap"])
    lactate_production_rate = max(0.0, qLac_prod * C_X * ML_PER_L)

    cons_open = _clamp01((t - lacp["cons_start"]) / max(1.0, lacp["cons_ramp"]))
    qLac_cons_base = lacp["cons_base"] if not lactate_shift_enhanced else 1.8*lacp["cons_base"]
    mix_cons_pen = max(0.25, mixing_eff**1.5)
    scale_pen = _clamp01((volume_factor - 0.5) / 3.5)
    inhib = (1.0 - 0.50*scale_pen*pco2_term) * (1.0 - 0.30*scale_pen*lowDO_term)
    inhib = max(0.3, inhib)
    qLac_cons = qLac_cons_base * mix_cons_pen * inhib * (C_lac / (0.5 + C_lac)) * cons_open
    lactate_consumption_rate = max(0.0, qLac_cons * C_X * ML_PER_L)

    VCD = C_X
    if volume_factor < 0.1:
        kLa_base_eff = kLa * 1.0
        do_peak_time = 30.0
        do_peak_boost = 0.04
        final_decline = 0.7
    elif volume_factor < 2.0:
        kLa_base_eff = kLa * 0.9
        do_peak_time = 25.0
        do_peak_boost = 0.02
        final_decline = 0.5
    else:
        kLa_base_eff = kLa * 0.75
        do_peak_time = 20.0
        do_peak_boost = 0.01
        final_decline = 0.3

    peak_factor = do_peak_boost * np.exp(-((t - do_peak_time)**2) / (2 * 15**2))
    if t > do_peak_time:
        time_decline = final_decline * (1.0 - np.exp(-(t - do_peak_time) / 80.0))
    else:
        time_decline = 0.0
    vcd_decline = 0.3 * min(1.0, VCD / 20e6)
    kLa_effective = kLa_base_eff * (1.0 + peak_factor - time_decline - vcd_decline)
    kLa_effective = max(20.0, kLa_effective)

    qO2 = 6.0 * q_glc + 1e-12
    qO2 *= (1.0 - 0.08 * max(0.0, (pCO2 - 50.0) / 80.0))
    OUR = max(0.0, qO2 * C_X * ML_PER_L)
    OTR = kLa_effective * (DO_sat - C_DO)
    DO_transfer_net = OTR - OUR

    qCO2 = max(1e-15, (1.2 * q_glc + 2.0e-12) * (1.0 + 0.6 * (1.0 - f_DO)))
    holdup = 1.0 + 0.80 * (1.0 - mixing_eff) * _clamp01(VCD / 12e6)
    CO2_prod = max(0.0, qCO2 * C_X * ML_PER_L) * holdup
    B_CO2 = scale_params.get("B_CO2_eff", B_CO2_EFF_DEFAULT)
    CO2_prod_mmHg = CO2_prod / B_CO2

    base_strip = scale_params.get("base_strip_CO2", 0.10)
    kLa_CO2_factor = scale_params.get("kLa_CO2_factor", 0.90)
    if t < 36.0:
        strip_mult = 1.25
    elif t < 72.0:
        strip_mult = 1.00
    else:
        strip_mult = 0.75

    # NEW: overlay airflow → CO2 strip coefficient
    overlay_vvm = float(scale_params.get("overlay_vvm", 0.03))
    kLa_CO2_mmHg = kla_co2_from_overlay(kLa_effective, overlay_vvm, base_strip, kLa_CO2_factor) * strip_mult
    co2_grad_pen = max(0.15, 1.0 - 0.85 * (1.0 - mixing_eff) * _clamp01(VCD / 15e6))
    kLa_CO2_mmHg *= co2_grad_pen
    CO2_strip = max(0.0, kLa_CO2_mmHg * max(0.0, (pCO2 - 40.0)))
    dpCO2_dt = CO2_prod_mmHg - CO2_strip - Fin / V * pCO2

    r_NH3 = scale_params.get("y_NH3_glc", 0.05) * glucose_consumption_rate + \
            scale_params.get("y_NH3_growth", 0.0) * max(mu - mu_d, 0) * C_X * ML_PER_L
    overflow_lac = max(lactate_production_rate - lactate_consumption_rate, 0.0)
    r_Ala = scale_params.get("y_Ala_lac", 0.05) * overflow_lac + scale_params.get("k_Ala_lowDO", 2e-12) * (1 - f_DO) * C_X * ML_PER_L
    uptake_Ala = scale_params.get("q_Ala_cons_max", 1e-12) * C_X * ML_PER_L * C_Ala / (scale_params.get("K_Ala", 1.0) + C_Ala)

    dC_X_dt   = max(-C_X/10.0, (mu - mu_d) * C_X - Fin / V * C_X)
    dC_glc_dt = -glucose_consumption_rate + Fin / V * (cin - C_glc)
    dC_lac_dt = lactate_production_rate - lactate_consumption_rate - Fin / V * C_lac
    dC_Ab_dt  = Qp_eff * C_X * 1000.0 - Fin / V * C_Ab
    dV_dt     = max(0.0, Fin)
    dC_DO_dt  = DO_transfer_net - Fin / V * C_DO
    dC_NH3_dt = r_NH3 - scale_params.get("k_strip_NH3", 0.01) * C_NH3 - Fin / V * C_NH3
    dC_Ala_dt = r_Ala - uptake_Ala - Fin / V * C_Ala

    return [dC_X_dt, dC_glc_dt, dC_lac_dt, dC_Ab_dt, dV_dt, dC_DO_dt, dpCO2_dt, dC_NH3_dt, dC_Ala_dt]

# =========================
# Specific rates helper (no solver change)
# =========================
def _compute_specific_rates_only(t, y, scale_params):
    C_X, C_glc, C_lac, C_Ab, V, C_DO, pCO2, C_NH3, C_Ala = y
    C_X   = max(1e-6, C_X); C_glc = max(0.05, C_glc); C_lac = max(0.0, C_lac); C_DO = max(1e-6, C_DO); pCO2 = max(20.0, pCO2)

    KIlac = 7.1; mglc = 1.5e-11; Qp_base = 9.0e-13; mu_max_base = 5.17e-2
    kd = 2.32e-2; Yxglc = 1.6e8; Kdglc = 1.54; Kglc = 1.0; Kdlac = 2.0

    DO_sat = scale_params.get("DO_sat", DO_SAT_GLOBAL)
    volume_factor = scale_params.get("volume_factor", 1.0)
    kLa_base = scale_params.get("kLa", 200.0)
    mixing_eff = scale_params.get("mixing_eff", 1.0)
    EDR = scale_params.get("EDR", 0.05)
    bubble_EDR = scale_params.get("bubble_EDR", 1e4)
    antifoam_enabled = scale_params.get("antifoam", True)
    antifoam_type = scale_params.get("antifoam_type", "PEG")

    K_max_base = 25e6
    if volume_factor < 0.1:   K_max = K_max_base * 1.05; growth_cutoff = 0.60; death_mult = 1.5
    elif volume_factor < 2.0: K_max = K_max_base * 1.00; growth_cutoff = 0.65; death_mult = 1.2
    else:                     K_max = K_max_base * 0.80; growth_cutoff = 0.70; death_mult = 1.0

    kLa = kLa_base * (0.90 if antifoam_enabled else 1.0)
    if antifoam_enabled and antifoam_type == "silicone":
        kLa *= 0.95

    temp_shift_time = scale_params.get("temp_shift_time", 72.0)
    if t > temp_shift_time:
        Qp = Qp_base * 1.35; mu_max = mu_max_base * 0.65; kd_temp = kd * 0.90; lactate_shift_enhanced = True
    else:
        Qp = Qp_base; mu_max = mu_max_base; kd_temp = kd; lactate_shift_enhanced = False

    density_factor = C_X / K_max
    if density_factor < growth_cutoff: f_carry = 1.0
    elif density_factor < 1.0:         f_carry = (1.0 - density_factor)/(1.0 - growth_cutoff)
    else:                               f_carry = 0.0

    if density_factor > 0.70: f_death_density = 1 + death_mult * 10 * (density_factor - 0.70) / 0.30
    elif density_factor > 0.50: f_death_density = 1 + death_mult * 3 * (density_factor - 0.50) / 0.20
    else: f_death_density = 1.0

    if pCO2 <= 80:   f_CO2_growth = 1.0
    elif pCO2 >=120: f_CO2_growth = 0.6
    else:            f_CO2_growth = 1.0 - 0.4*(pCO2 - 80.0)/40.0

    if pCO2 <=100:   f_CO2_qp = 1.0
    elif pCO2 >=150: f_CO2_qp = 0.6
    else:            f_CO2_qp = 1.0 - 0.4*(pCO2 - 100.0)/50.0

    f_DO = _clamp01(C_DO / (C_DO + 0.02))
    f_shear_growth = _clamp01(1.0 - 0.8 * max(0.0, EDR - 0.06) / 0.44)
    f_shear_qp     = _clamp01(1.0 - 0.3 * max(0.0, EDR - 0.06) / 0.34)
    shear_death_factor = 1 + (bubble_EDR / 1e6 - 1) * 0.5 if bubble_EDR > 1e6 else 1.0
    nutrient_gradient_factor = 0.5 + 0.5 * mixing_eff

    K_NH3_g  = scale_params.get("K_NH3_g", 4.0)
    K_NH3_qp = scale_params.get("K_NH3_qp", 6.0)
    use_ala_penalty = scale_params.get("ala_penalty", False)
    K_Ala_g = scale_params.get("K_Ala_g", 20.0)
    K_Ala_qp = scale_params.get("K_Ala_qp", 20.0)
    f_NH3_growth = K_NH3_g  / (K_NH3_g  + C_NH3)
    f_NH3_qp     = K_NH3_qp / (K_NH3_qp + C_NH3)
    if use_ala_penalty:
        f_Ala_growth = K_Ala_g  / (K_Ala_g  + C_Ala)
        f_Ala_qp     = K_Ala_qp / (K_Ala_qp + C_Ala)
    else:
        f_Ala_growth = 1.0; f_Ala_qp = 1.0

    C_glc_safe = max(C_glc, 0.05); C_lac_safe = max(C_lac, 1e-6)
    mu_max_eff = max(0.001, mu_max * f_carry * f_CO2_growth * f_DO * f_shear_growth *
                     nutrient_gradient_factor * f_NH3_growth * f_Ala_growth)

    mu = max(0.0, mu_max_eff *
             (C_glc_safe / (Kglc + C_glc_safe)) *
             (KIlac / (KIlac + C_lac_safe)))

    mu_d = max(0.0, kd_temp * shear_death_factor * f_death_density *
               (1 + 0.5 * max(density_factor - 0.6, 0) / 0.4) *
               (C_lac_safe / (Kdlac + C_lac_safe)) *
               (Kdglc / (Kdglc + C_glc_safe)))

    q_glc = (mu - mu_d) / 1.6e8 + 1.5e-11  # Yxglc + mglc

    # === Lactate production ===
    lacp = _lactate_params(volume_factor)
    G_th, mu_th = lacp["G_th"], lacp["mu_th"]
    f_age = 1.0 - np.exp(- (t / max(1.0, lacp["age_tau"]))**2)
    f_biom = C_X / (C_X + lacp["biom_K"])
    overflow_gate = (_clamp01((C_glc - G_th)/max(G_th,1e-6)) *
                     _clamp01((mu - mu_th)/max(mu_th,1e-6)) *
                     f_age * f_biom)
    pco2_term = _clamp01((pCO2 - 40.0) / 80.0)
    lowDO_term = 1.0 - _clamp01(C_DO / (C_DO + 0.02))
    qLac_prod = 0.45 * ((mu - mu_d) / 1.6e8 + 1.5e-11) * overflow_gate
    qLac_prod *= (1.0 + 0.35*pco2_term + 0.20*lowDO_term)

    # === NEW: Lactate consumption (mirror of ODE logic) ===
    cons_open = _clamp01((t - lacp["cons_start"]) / max(1.0, lacp["cons_ramp"]))
    qLac_cons_base = lacp["cons_base"] if t <= scale_params.get("temp_shift_time", 72.0) else 1.8 * lacp["cons_base"]
    mix_cons_pen = max(0.25, mixing_eff ** 1.5)
    pco2_term = _clamp01((pCO2 - 40.0) / 80.0)
    lowDO_term = 1.0 - _clamp01(C_DO / (C_DO + 0.02))
    scale_pen = _clamp01((volume_factor - 0.5) / 3.5)
    inhib = (1.0 - 0.50 * scale_pen * pco2_term) * (1.0 - 0.30 * scale_pen * lowDO_term)
    inhib = max(0.3, inhib)
    qLac_cons = qLac_cons_base * mix_cons_pen * inhib * (C_lac_safe / (0.5 + C_lac_safe)) * cons_open

    # === Remaining rates ===
    mu_sat = mu / (mu + 0.02)
    qP_shape = 0.40 + 0.60 * mu_sat
    Qp_with_antifoam = Qp * (1.1 if (antifoam_enabled and antifoam_type == "PEG") else 1.0)
    Qp_eff = max(1e-15, Qp_with_antifoam * f_CO2_qp * f_DO * f_shear_qp * qP_shape * f_NH3_qp * f_Ala_qp)

    qO2 = (6.0 * q_glc + 1e-12) * (1.0 - 0.08 * max(0.0, (pCO2 - 50.0) / 80.0))
    qCO2 = max(1e-15, (1.2 * q_glc + 2.0e-12) * (1.0 + 0.6 * (1.0 - (C_DO / (C_DO + 0.02)))))

    return {
        "mu": mu, "mu_d": mu_d, "q_glc": q_glc,
        "qLac_prod": qLac_prod, "qLac_cons": qLac_cons,
        "Qp_eff": Qp_eff, "qO2": qO2, "qCO2": qCO2
    }

# =========================
# Sampling + noise helpers
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
    online_idx = _nearest_indices(float(online_minutes) / 60.0) if mode == "Lab" else np.arange(len(t))

    offline_vars = ["C_X","C_glc","C_lac","C_Ab","C_NH3","C_Ala"]
    online_vars = ["C_DO","pCO2"]

    for var in offline_vars + online_vars:
        df[f"{var}_meas"] = np.nan

    for var in offline_vars:
        df.loc[offline_idx, f"{var}_meas"] = df.loc[offline_idx, var]
    for var in online_vars:
        df.loc[online_idx, f"{var}_meas"] = df.loc[online_idx, var]

    df["offline_sample"] = False; df.loc[offline_idx, "offline_sample"] = True
    df["online_sample"] = False;  df.loc[online_idx, "online_sample"] = True

    df["sampling_mode"] = "Hybrid (Lab + Continuous Online)" if mode == "Hybrid" else "Realistic Lab Sampling"
    df["offline_interval_hrs"] = offline_hours
    df["online_interval_mins"] = online_minutes if mode == "Lab" else 0
    return df

def _apply_measurement_noise(df, enable=False, cv_offline=0.05, cv_online=0.02, seed=42, drift_online=True):
    if not enable:
        return df

    rng = np.random.default_rng(int(seed))

    # ensure output columns exist
    meas_cols = [c for c in df.columns if c.endswith("_meas")]
    for col in meas_cols:
        out_col = col.replace("_meas", "_meas_noisy")
        df[out_col] = np.nan

    DO_MIN, DO_MAX = 0.0, DO_SAT_GLOBAL
    PCO2_MIN, PCO2_MAX = 30.0, 160.0

    # offline (log-normal)
    for var in ["C_X","C_glc","C_lac","C_Ab","C_NH3","C_Ala"]:
        col = f"{var}_meas"
        if col not in df.columns:
            continue
        vals = df[col].to_numpy()
        mask = ~np.isnan(vals)
        if mask.any():
            df.loc[mask, col.replace("_meas","_meas_noisy")] = add_noise_offline_log_normal(vals[mask], cv_offline, rng)

    # online (Gaussian + drift)
    if "C_DO_meas" in df.columns:
        vals = df["C_DO_meas"].to_numpy()
        mask = ~np.isnan(vals)
        if mask.any():
            df.loc[mask, "C_DO_meas_noisy"] = add_noise_online_gaussian(vals[mask], cv_online, rng,
                                                                        y_min=DO_MIN, y_max=DO_MAX, add_drift=drift_online)
    if "pCO2_meas" in df.columns:
        vals = df["pCO2_meas"].to_numpy()
        mask = ~np.isnan(vals)
        if mask.any():
            df.loc[mask, "pCO2_meas_noisy"] = add_noise_online_gaussian(vals[mask], cv_online, rng,
                                                                         y_min=PCO2_MIN, y_max=PCO2_MAX, add_drift=drift_online)
    return df

# =========================
# Noise models (used by _apply_measurement_noise)
# =========================
def add_noise_offline_log_normal(values, cv=0.05, rng=None):
    """
    Multiplicative log-normal noise for offline assays.
    cv is coefficient of variation (e.g., 0.05 = 5%).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    values = np.asarray(values, dtype=float)
    # log-normal params: sigma = sqrt(log(1+cv^2))
    sigma = float(np.sqrt(np.log(1.0 + (cv ** 2))))
    mu = -0.5 * sigma ** 2  # so that E[exp(N(mu, sigma^2))] = 1
    mult = rng.lognormal(mean=mu, sigma=sigma, size=values.shape)
    noisy = values * mult
    # keep non-negative & preserve NaNs
    noisy[~np.isfinite(noisy)] = np.nan
    noisy = np.maximum(noisy, 0.0)
    return noisy

def add_noise_online_gaussian(values, cv=0.02, rng=None, y_min=None, y_max=None, add_drift=True):
    """
    Additive Gaussian noise for online sensors (DO, pCO2).
    cv is relative to current value (heteroscedastic).
    If add_drift=True, include a small random walk drift.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    values = np.asarray(values, dtype=float)
    base_sigma = np.maximum(np.abs(values) * cv, 1e-12)
    gauss = rng.normal(loc=0.0, scale=base_sigma, size=values.shape)
    if add_drift:
        # gentle random walk (10–20% of base noise)
        drift_step = rng.normal(loc=0.0, scale=base_sigma * 0.15, size=values.shape)
        drift = np.cumsum(drift_step)
    else:
        drift = 0.0
    noisy = values + gauss + drift
    if y_min is not None:
        noisy = np.maximum(noisy, y_min)
    if y_max is not None:
        noisy = np.minimum(noisy, y_max)
    noisy[~np.isfinite(noisy)] = np.nan
    return noisy

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
        "feed_guard_k": process_params.get("feed_guard_k", 1.0),
        "glc_floor_mM": process_params.get("glc_floor_mM", 18.0),
        "pH_setpoint": process_params.get("pH_setpoint", 7.0),
        # NEW: overlay airflow (vvm)
        "overlay_vvm": process_params.get("overlay_vvm", 0.03),
    }

    if scale == "2L":
        scale_params.update({
            "kLa": 280, "mixing_eff": 1.0, "volume_factor": 0.02,
            "EDR": 0.02, "bubble_EDR": 1e4,
            "B_CO2_eff": 1.80, "base_strip_CO2": 0.25, "kLa_CO2_factor": 1.05,
        })
    elif scale == "50L":
        scale_params.update({
            "kLa": 90, "mixing_eff": 0.90, "volume_factor": 1.0,
            "EDR": 0.05, "bubble_EDR": 2e5,
            "B_CO2_eff": 1.00, "base_strip_CO2": 0.08, "kLa_CO2_factor": 0.75,
        })
    else:  # 2000 L
        scale_params.update({
            "kLa": 35, "mixing_eff": 0.55, "volume_factor": 4.0,
            "EDR": 0.15, "bubble_EDR": 2e6, "antifoam_type": "silicone",
            "B_CO2_eff": 0.70, "base_strip_CO2": 0.012, "kLa_CO2_factor": 0.60,
        })

    if advanced_params:
        scale_params.update(advanced_params)

    t_start, t_end = 0.0, 14*24.0
    time_points = np.linspace(t_start, t_end, 200)
    inoc_cells_per_ml = process_params.get("inoculation_density", 0.8e6)
    init_V_mL = {"2L": 2000, "50L": 50000, "2000L": 2000000}[scale]

    initial_conditions = [inoc_cells_per_ml, 41.0, 0.0, 0.0, init_V_mL, 0.20, 40.0, 0.0, 0.0]
    try:
        sol = solve_ivp(
            lambda tt, yy: bioreactor_odes_with_scale(tt, yy, scale_params),
            [t_start, t_end],
            initial_conditions,
            t_eval=time_points,
            method='LSODA',
            rtol=1e-4,
            atol=1e-8,
            max_step=2.0
        )
        if not sol.success:
            st.error(f"Simulation failed: {sol.message}")
            return None

        cols = ["C_X","C_glc","C_lac","C_Ab","V","C_DO","pCO2","C_NH3","C_Ala"]
        df = pd.DataFrame(np.vstack(sol.y).T, columns=cols)
        df["time"] = sol.t
        df["C_X_raw_cells_per_ml"] = df["C_X"].copy()
        df["C_X"] /= 1e6  # display as 10^6 cells/mL

        # === Compute specific rates at each timepoint ===
        rates_list = []
        for i in range(len(sol.t)):
            y_i = [sol.y[j][i] for j in range(9)]
            rates = _compute_specific_rates_only(sol.t[i], y_i, scale_params)
            rates_list.append(rates)
        if rates_list:
            for key in rates_list[0].keys():
                df[key] = [r[key] for r in rates_list]

        # === NEW: pH estimate column ===
        df["pH_est"] = [estimate_pH_from_CO2_Lac(p, l, pH_set=float(process_params.get("pH_setpoint", 7.0)))
                        for p, l in zip(df["pCO2"].values, df["C_lac"].values)]

        # === NEW: CTR (mmol-C/L/h) ===
        B_eff = scale_params.get("B_CO2_eff", B_CO2_EFF_DEFAULT)
        df["DIC_mmolC_L"] = B_eff * df["pCO2"].astype(float)

        # use derivative, not plain difference
        t = df["time"].astype(float).values
        DIC = df["DIC_mmolC_L"].astype(float).values
        dDIC_dt = np.gradient(DIC, t)  # mmol-C/L per hour

        # reconstruct Fin and vvd
        V = df["V"].astype(float).values
        dt = np.diff(t)
        dt_med = float(np.median(dt)) if dt.size else 1.0
        Fin = np.diff(V, prepend=V[0]) / max(dt_med, 1e-9)  # mL/h
        vvd_per_day = (Fin / np.maximum(V, 1e-9)) * 24.0
        df["vvd_per_day"] = vvd_per_day
        df["vvd_h"] = vvd_per_day  # (kept for compatibility; same values)

        cells_per_mL = df["C_X_raw_cells_per_ml"].astype(float).values
        qCO2 = df.get("qCO2", pd.Series(np.nan, index=df.index)).astype(float).values
        ProdCO2 = qCO2 * cells_per_mL * 1000.0  # mmol-C/L/h

        dilution = (df["vvd_per_day"] / 24.0) * df["DIC_mmolC_L"]  # mmol-C/L/h
        df["CTR_mmolC_L_h"] = ProdCO2 - dDIC_dt - dilution.values


        # sampling markers & noise
        if process_params.get("sampling_enabled", False):
            df = _attach_sampling_markers(
                df,
                t_end_hours=t_end,
                enabled=True,
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
            drift_online=True
        )

        # sanity checks
        assert df["C_Ab"].iloc[-1] < 8.5, f"Titer exploded: {df['C_Ab'].iloc[-1]:.2f} g/L (should be < 8.5)"
        assert df["C_X"].max() < 55.0, f"Peak VCD too high: {df['C_X'].max():.1f} ×10⁶/mL (should be < 55)"

        # metadata (+ NEW engineering levers you want exported)
        df["batch_name"] = batch_name
        df["scale"] = scale
        df["overlay_vvm"] = float(scale_params.get("overlay_vvm", 0.03))
        df["kLa_nominal"] = float(scale_params.get("kLa", np.nan))
        df["B_CO2_eff"] = float(scale_params.get("B_CO2_eff", B_CO2_EFF_DEFAULT))

        for k, v in process_params.items():
            if k in ("feed_profile",):
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
# ========================
def calculate_summary_metrics(df, tan_thresh=4.0, pco2_thresh=120.0):
    if df is None or df.empty:
        return {}

    days = df["time"].iloc[-1] / 24.0

    if "C_X_raw_cells_per_ml" in df.columns:
        ivcd_cells_per_L_day = _trapezoid(df["C_X_raw_cells_per_ml"] * 1000.0,
                                          df["time"] / 24.0)
        qP_pg_per_cell_day = (df["C_Ab"].iloc[-1]) / ivcd_cells_per_L_day * 1e12 if ivcd_cells_per_L_day > 0 else 0.0
    else:
        qP_pg_per_cell_day = (df["C_Ab"].iloc[-1] * 1000) / (df["C_X"].mean() * days + 1e-9)

    max_TAN = df["C_NH3"].max() if "C_NH3" in df.columns else 0.0

    t = df["time"].values
    time_TAN_high = 0.0
    if "C_NH3" in df.columns:
        mask = (df["C_NH3"].values > tan_thresh).astype(float)
        if mask.any():
            time_TAN_high = float(_trapezoid(mask, t))

    pco2 = df["pCO2"].values
    hours_pco2_high = float(_trapezoid((pco2 > pco2_thresh).astype(float), t))
    idx_peak = int(np.nanargmax(pco2))
    t_peak = float(df["time"].iloc[idx_peak])

    return {
        "final_titer": df["C_Ab"].iloc[-1],
        "peak_VCD": df["C_X"].max(),
        "final_VCD": df["C_X"].iloc[-1],
        "max_pCO2": float(np.nanmax(pco2)),
        "time_peak_pCO2_h": t_peak,
        "hours_pCO2_above_120": hours_pco2_high,
        "min_DO": df["C_DO"].min(),
        "lactate_shift_success": df["C_lac"].iloc[-1] < df["C_lac"].max() - 3,
        "growth_arrest": float(np.nanmax(pco2)) > pco2_thresh,
        "cell_specific_productivity": qP_pg_per_cell_day,
        "max_TAN_mM": max_TAN,
        "hours_TAN_above_thresh": time_TAN_high,
        "final_Ala_mM": df["C_Ala"].iloc[-1] if "C_Ala" in df.columns else 0.0,
    }

# =========================
# CSV helpers
# =========================
def infer_kla_from_do(scale, do_pct):
    if do_pct is None:
        return None
    def pick(mapping, v):
        key = min(mapping.keys(), key=lambda k: abs(k - v))
        return mapping[key]
    if str(scale) == "2L":
        return pick({40: 225, 60: 280, 80: 335}, do_pct)
    elif str(scale) == "50L":
        return pick({40: 70, 60: 90, 80: 110}, do_pct)
    else:
        return pick({40: 25, 60: 35, 80: 45}, do_pct)

def build_feed_profile_from_row(row):
    rel = _safe_float(row.get("feed_VVD_relative"), 1.0)
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
    start_day = _safe_float(row.get("feed_start_day"))
    if start_day is not None:
        start_h = int(start_day * 24)
        phases = [(start_h, 0.050 * rel)]
        sh2 = _safe_float(row.get("phase2_start_h"))
        vv2 = _safe_float(row.get("phase2_vvd_per_day"))
        if sh2 is not None and vv2 is not None:
            phases.append((int(sh2), float(vv2) * rel))
        return sorted(phases, key=lambda x: x[0])
    return None

def parse_row_to_params(row):
    batch_name = str(row.get("BatchName", row.get("Global_Run_Order", "Run")))
    scale = str(row.get("Scale", "2L")).strip()
    if scale not in {"2L", "50L", "2000L"}:
        if "2000" in scale:
            scale = "2000L"
        elif "50" in scale:
            scale = "50L"
        else:
            scale = "2L"

    inoc_million = _safe_float(row.get("inoc_density_1e6_per_mL", row.get("inoc_dens", 0.35)), 0.35)
    inoculation_density = inoc_million * 1e6
    temp_shift_time = int(_safe_float(row.get("temp_shift_time_h", row.get("temp_shif", 72)), 72))
    pH_setpoint = _safe_float(row.get("pH_setpoint"), 7.0)
    agitation_rate = _safe_float(row.get("agitation_rate_rpm"), 100.0)
    pO2_setpoint = _safe_float(row.get("DO_setpoint_pct"), 60.0)
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
        "overlay_vvm": _safe_float(row.get("overlay_vvm"), 0.03),  # NEW: allow in CSV
    }

    advanced_params = {}
    kLa_csv = _safe_float(row.get("kLa"))
    do_pct = _safe_float(row.get("DO_setpoint_pct"))
    if kLa_csv is not None:
        advanced_params["kLa"] = float(kLa_csv)
    else:
        kla_guess = infer_kla_from_do(scale, do_pct)
        if kla_guess is not None:
            advanced_params["kLa"] = float(kla_guess)

    return str(batch_name), scale, process_params, advanced_params

# =========================
# Validation + DoE generator + batch
# =========================
SCALE_PRESETS = {
    "2L": {
        "inoc_density_1e6_per_mL": (0.30, 1.20),
        "temp_shift_time_h": (48, 120),
        "pH_setpoint": (6.80, 7.40),
        "DO_setpoint_pct": (20, 80),
        "kLa": (200, 360),
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
        "kLa": (70, 110),
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
        "kLa": (25, 45),
        "phase1_start_h": (36, 60),
        "phase1_vvd_per_day": (0.010, 0.030),
        "phase2_start_h": (96, 144),
        "phase2_vvd_per_day": (0.015, 0.035),
        "cin_mM": (1000, 3000),
        "delta_phase_h_min": 24
    }
}

def validate_row(row, scale):
    s = SCALE_PRESETS.get(scale, SCALE_PRESETS["2L"])
    warnings, errors = [], []

    def _chk(name, val):
        if val is None:
            return
        lo, hi = s[name]
        if not (lo <= val <= hi):
            warnings.append(f"{name}={val} outside {lo}-{hi}")

    _chk("pH_setpoint", _safe_float(row.get("pH_setpoint")))
    _chk("DO_setpoint_pct", _safe_float(row.get("DO_setpoint_pct")))
    _chk("kLa", _safe_float(row.get("kLa")))
    _chk("inoc_density_1e6_per_mL", _safe_float(row.get("inoc_density_1e6_per_mL", row.get("inoc_dens"))))
    _chk("temp_shift_time_h", _safe_float(row.get("temp_shift_time_h", row.get("temp_shif"))))
    p1s = _safe_float(row.get("phase1_start_h")); _chk("phase1_start_h", p1s)
    p1v = _safe_float(row.get("phase1_vvd_per_day")); _chk("phase1_vvd_per_day", p1v)
    p2s = _safe_float(row.get("phase2_start_h")); _chk("phase2_start_h", p2s)
    p2v = _safe_float(row.get("phase2_vvd_per_day")); _chk("phase2_vvd_per_day", p2v)

    if p1s is not None and p2s is not None and p2s < p1s + s["delta_phase_h_min"]:
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

def generate_doe(scale, n_runs, seed, ranges,
                 feed_rel_policy="Tertiles",
                 antifoam_mode="Alternate",
                 cin_mM=2220):
    rng = np.random.default_rng(seed)
    cont_keys = ["pH_setpoint","DO_setpoint_pct","kLa",
                 "inoc_density_1e6_per_mL","temp_shift_time_h",
                 "phase1_start_h","phase1_vvd_per_day",
                 "phase2_start_h","phase2_vvd_per_day"]
    low  = np.array([ranges[k][0] for k in cont_keys])
    high = np.array([ranges[k][1] for k in cont_keys])

    X = lhs(n_runs, len(cont_keys), rng)
    M = low + (high - low) * X
    df = pd.DataFrame(M, columns=cont_keys)

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

    df["Scale"] = scale
    df["num_feed_phases"] = 2
    df["Design"] = f"LHS-{n_runs}"
    df["Phase"] = "fed-batch"
    df["cin_mM"] = int(cin_mM)

    if antifoam_mode == "PEG only":
        df["antifoam_type"] = "PEG"
    elif antifoam_mode == "silicone only":
        df["antifoam_type"] = "silicone"
    else:
        df["antifoam_type"] = ["PEG" if i % 2 == 0 else "silicone" for i in range(n_runs)]

    if feed_rel_policy == "Fixed 1.0":
        rel = np.full(n_runs, 1.0)
    elif feed_rel_policy == "Cycle 0.8/1.0/1.2":
        rel = np.array([0.8,1.0,1.2] * (n_runs // 3 + 1))[:n_runs]
    else:
        mean_vvd = (df["phase1_vvd_per_day"] + df["phase2_vvd_per_day"]) / 2.0
        q1, q2 = np.quantile(mean_vvd, [1/3, 2/3])
        rel = mean_vvd.apply(lambda x: 0.8 if x <= q1 else (1.0 if x <= q2 else 1.2)).values
    df["feed_VVD_relative"] = rel

    df.insert(0, "Global_Run_Order", np.arange(1, n_runs + 1))
    df.insert(1, "BatchName", [f"R{scale.replace('L','')}_{i:03d}" for i in range(1, n_runs + 1)])
    df["inoc_dens"] = df["inoc_density_1e6_per_mL"]

    cols = ["Global_Run_Order","BatchName","Scale",
            "pH_setpoint","DO_setpoint_pct","kLa",
            "inoc_dens","temp_shift_time_h",
            "num_feed_phases",
            "phase1_start_h","phase1_vvd_per_day",
            "phase2_start_h","phase2_vvd_per_day",
            "feed_VVD_relative","cin_mM",
            "Design","Phase","antifoam_type"]
    return df[cols]

def run_batch(doe_df, apply_overrides=None):
    results, rows_run, problems = [], [], []
    for idx, row in doe_df.iterrows():
        batch_name, scale, pp, adv = parse_row_to_params(row)
        ok, warns, errs = validate_row(row, scale)
        if not ok:
            problems.append({
                "row_index": idx,
                "BatchName": batch_name,
                "Scale": scale,
                "errors": "; ".join(errs),
                "warnings": "; ".join(warns)
            })
            continue
        if apply_overrides:
            pp.update(apply_overrides)
        df = run_simulation(f"{batch_name}_{scale}", scale, pp, adv if len(adv) else None)
        if df is not None:
            results.append(df); rows_run.append(row)

    as_run = pd.DataFrame(rows_run) if rows_run else pd.DataFrame(columns=doe_df.columns)
    prob_df = pd.DataFrame(problems) if problems else pd.DataFrame(columns=["row_index","BatchName","Scale","errors","warnings"])
    return results, as_run, prob_df

# =========================
# Streamlit App
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
                    "Time to peak pCO₂ (h)": f"{m['time_peak_pCO2_h']:.0f}",
                    "Hours pCO₂ > 120 (h)": f"{m['hours_pCO2_above_120']:.1f}",
                    "Min DO (mmol/L)": f"{m['min_DO']:.3f}",
                    "Max TAN (mM)": f"{m['max_TAN_mM']:.2f}",
                    "Hours TAN > 4 mM": f"{m['hours_TAN_above_thresh']:.1f}h",
                    "Final Alanine (mM)": f"{m['final_Ala_mM']:.2f}",
                    "Lactate Shift": "Yes" if m['lactate_shift_success'] else "No",
                    "Growth Arrest": "Warning" if m['growth_arrest'] else "No",
                })
            metrics_df = pd.DataFrame(metrics_rows)
            st.dataframe(metrics_df, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            final_titers = [calculate_summary_metrics(df)['final_titer'] for df in results]
            peak_vcds = [calculate_summary_metrics(df)['peak_VCD'] for df in results]
            max_pco2s = [calculate_summary_metrics(df)['max_pCO2'] for df in results]
            growths = sum([calculate_summary_metrics(df)['growth_arrest'] for df in results])
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
            colors = {"2L":"#1f77b4", "50L":"#2ca02c", "2000L":"#d62728"}
            first_legend_for_scale = set()

            for df in results:
                scale = df["scale"].iloc[0]
                color = colors.get(scale, "#444")
                showlegend = scale not in first_legend_for_scale
                if showlegend:
                    first_legend_for_scale.add(scale)

                with np.errstate(divide='ignore', invalid='ignore'):
                    df['cell_specific_productivity'] = df['C_Ab'] * 1000 / (df['C_X'] * df['time'] / 24 + 1e-6)
                    df['glucose_consumption_rate'] = -df['C_glc'].diff() / df['time'].diff()

                variables = [
                    ('C_X',1,1),('C_Ab',1,2),('C_glc',1,3),
                    ('C_lac',2,1),('C_DO',2,2),('pCO2',2,3),
                    ('V',3,1),('cell_specific_productivity',3,2),('glucose_consumption_rate',3,3),
                    ('C_NH3',4,1),('C_Ala',4,2)
                ]
                for var, row, col in variables:
                    if var not in df.columns:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=df["time"], y=df[var],
                            name=scale, legendgroup=scale,
                            line=dict(color=color),
                            showlegend=showlegend
                        ),
                        row=row, col=col
                    )

                    noisy_col = f"{var}_meas_noisy"
                    meas_col  = f"{var}_meas"
                    if noisy_col in df.columns:
                        tt = df["time"].to_numpy()
                        yy = df[noisy_col].to_numpy()
                        mask = ~np.isnan(yy)
                        if mask.any():
                            fig.add_trace(
                                go.Scatter(
                                    x=tt[mask], y=yy[mask],
                                    mode="markers",
                                    name=f"{scale} noisy {var}",
                                    legendgroup=scale,
                                    showlegend=False,
                                    marker=dict(size=6, symbol="x", opacity=0.9),
                                ),
                                row=row, col=col
                            )
                    elif meas_col in df.columns:
                        tt = df["time"].to_numpy()
                        yy = df[meas_col].to_numpy()
                        mask = ~np.isnan(yy)
                        if mask.any():
                            fig.add_trace(
                                go.Scatter(
                                    x=tt[mask], y=yy[mask],
                                    mode="markers",
                                    name=f"{scale} samples {var}",
                                    legendgroup=scale,
                                    showlegend=False,
                                    marker=dict(size=4, opacity=0.5),
                                ),
                                row=row, col=col
                            )

            for r in range(1,5):
                for c in range(1,4):
                    fig.add_vline(x=72, line_dash="dash", line_color="black", opacity=0.5, row=r, col=c)
                    fig.add_vline(x=96, line_dash="dot", line_color="purple", opacity=0.5, row=r, col=c)

            pvals = []
            for df in results:
                if "pCO2" in df.columns:
                    arr = df["pCO2"].to_numpy()
                    arr = arr[np.isfinite(arr)]
                    if arr.size:
                        pvals.append(arr)
            if pvals:
                pv = np.concatenate(pvals)
                ymin = float(np.nanmin(pv))
                ymax = float(np.nanmax(pv))
                margin = max(1.0, 0.2 * (ymax - ymin))
                lo = max(30.0, ymin - margin)
                hi = min(160.0, ymax + margin)
                if hi <= lo:
                    lo, hi = 38.0, 43.0
                tick0 = float(np.floor(lo))
                fig.update_yaxes(title_text="pCO₂ (mmHg)", range=[lo, hi], tick0=tick0, dtick=1.0, row=2, col=3)
                if lo < 40.0 < hi:
                    fig.add_hline(y=40.0, line_dash="dot", line_color="gray", opacity=0.35, row=2, col=3)
                band_lo, band_hi = 55.0, 150.0
                if hi > band_lo and lo < band_hi:
                    fig.add_hrect(y0=max(band_lo, lo), y1=min(band_hi, hi),
                                  fillcolor="red", opacity=0.08, line_width=0, row=2, col=3)

            fig.update_layout(height=1200, showlegend=True,
                              title_text="Comprehensive CHO Bioprocess Analysis (with TAN & Alanine)")
            fig.update_xaxes(title_text="Time (h)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Specific Rates (first 5 rows per run)")
            rate_cols = ["time","mu","mu_d","q_glc","qLac_prod","qLac_cons","Qp_eff","qO2","qCO2"]
            for df in results:
                st.caption(f"Run: {df['batch_name'].iloc[0]} @ {df['scale'].iloc[0]}")
                present = [c for c in rate_cols if c in df.columns]
                st.dataframe(df[present].head(), use_container_width=True)

            st.subheader("💾 Download Results")
            colA, colB = st.columns(2)
            combined_df = pd.concat(results, ignore_index=True)
            batch_safe = st.session_state.get('batch_name', 'CHO_Run')
            safe_name = re.sub(r"[^\w\-.]+", "_", batch_safe)
            combined_df[batch_safe] = batch_safe
            metrics_out = pd.DataFrame(metrics_rows); metrics_out[batch_safe] = batch_safe
            with colA:
                st.download_button("📄 Download Complete Dataset",
                                   data=combined_df.to_csv(index=False),
                                   file_name=f"{safe_name}.csv",
                                   mime="text/csv")
            with colB:
                st.download_button("📊 Download Summary Metrics",
                                   data=metrics_out.to_csv(index=False),
                                   file_name=f"{safe_name}_summary.csv",
                                   mime="text/csv")

def main():
    st.set_page_config(page_title="CHO Cross-scale Fed-Batch Bioprocess DoE Simulator",
                       page_icon="🧬", layout="wide")
    st.title("CHO Cross-scale Fed-Batch Bioprocess DoE Simulator")
    st.markdown("**Death-focused VCD control with preserved productivity** + TAN & Alanine (improved DO/CO₂ & lactate dynamics)")

    default_mode = st.session_state.get("default_mode", "Single Run Simulation")
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Single Run Simulation", "DoE CSV Runner", "DoE Generator"],
        index=["Single Run Simulation", "DoE CSV Runner", "DoE Generator"].index(default_mode)
    )

    if mode == "Single Run Simulation":
        st.header("CHO Bioprocess Simulation")
        left, right = st.columns([1, 2])
        with left:
            batch_name = st.text_input("Batch Name", value="CHO_Run")

            st.subheader("Scale Selection")
            scales_to_run = st.multiselect("Select Scales to Run", ["2L","50L","2000L"],
                                           default=["2L","50L","2000L"])
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
                feed_glucose = st.number_input("Feed glucose concentration, cin (mmol/L)",
                                               100, 3000, 722, step=10)
                if use_custom_feed:
                    n_phases = st.number_input("Number of feed phases", 1, 2, 2, step=1)
                    default_starts = [60, 150]
                    default_vvds = [0.05, 0.065]
                    feed_profile = []
                    last_start = 0
                    for i in range(n_phases):
                        start = st.number_input(f"Phase {i+1} start hour", 0, 336,
                                                default_starts[i] if i < len(default_starts) else last_start+60,
                                                step=6)
                        vvd = st.number_input(f"Phase {i+1} VVD (1/day)", 0.0, 0.2,
                                              float(default_vvds[i] if i < len(default_vvds) else default_vvds[-1]),
                                              step=0.005, format="%.3f")
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
                    sampling_mode = st.radio("Mode",
                                             ["Lab (offline+online at intervals)", "Hybrid (offline sampled, online continuous)"],
                                             index=0)
                    sampling_mode_token = "Lab" if sampling_mode.startswith("Lab") else "Hybrid"
                    offline_interval = st.number_input("Offline lab sampling interval (hours)", 12, 48, 24, step=6)
                    online_interval = st.number_input("Online sensor sampling interval (minutes)", 5, 60, 15, step=5)
                else:
                    sampling_mode_token, offline_interval, online_interval = "Lab", 24, 15

            with st.expander("Measurement Noise (applied to measured points only)"):
                noise_enabled = st.checkbox("Add measurement noise", value=False)
                noise_cv_offline = st.slider("Offline CV (%, VCD/titer/metabolites)", 0.0, 20.0, 5.0, step=0.5) / 100.0
                noise_cv_online = st.slider("Online CV (%, DO/pCO₂)", 0.0, 10.0, 2.0, step=0.5) / 100.0
                noise_seed = st.number_input("Noise seed", 0, 9999, 42, step=1)

            with st.expander("Advanced Engineering Parameters"):
                st.caption("⚠️ Engineering parameters - modify only for model validation")
                enable_advanced = st.checkbox("Enable Advanced Parameter Override")
                advanced_params = {}
                if enable_advanced:
                    advanced_params = {
                        "kLa": st.slider("kLa Override (1/h)", 20, 400, 200),
                        "EDR": st.slider("EDR Override (W/kg)", 0.01, 0.5, 0.05),
                        "mixing_eff": st.slider("Mixing Efficiency", 0.4, 1.0, 0.85),
                        "volume_factor": st.slider("Volume Factor", 0.01, 5.0, 1.0),
                        "B_CO2_eff": st.slider("Effective carbonate buffer capacity (mmol/L/mmHg)", 0.4, 2.5, 1.2),
                        "base_strip_CO2": st.slider("Base CO₂ strip coeff (fraction of kLa)", 0.005, 0.3, 0.10),
                        "kLa_CO2_factor": st.slider("CO₂/kLa coupling factor", 0.4, 1.5, 0.90),
                    }

            with st.expander("Gas / Overlay settings"):
                overlay_vvm = st.slider("Overlay airflow (vvm)", 0.0, 0.20, 0.03, step=0.005)

            # Run button
            run_btn = st.button("🚀 Run Simulation")

        # Right panel will render plots/results after we compute
        with right:
            if run_btn:
                st.session_state['batch_name'] = batch_name
                results = []
                for scale in scales_to_run:
                    process_params = {
                        "inoculation_density": float(inoculation_density * 1e6),
                        "temp_shift_time": int(temp_shift_time),
                        "pH_setpoint": float(pH_setpoint),
                        "agitation_rate": float(agitation_rate),
                        "pO2_setpoint": float(pO2_setpoint),
                        "antifoam": bool(antifoam),
                        "antifoam_type": str(antifoam_type) if antifoam else "PEG",
                        "feed_profile": feed_profile,
                        "feed_glucose": int(feed_glucose),
                        "ala_penalty": bool(ala_penalty),
                        "sampling_enabled": bool(sampling_enabled),
                        "sampling_mode": sampling_mode_token,
                        "offline_interval": int(offline_interval),
                        "online_interval": int(online_interval),
                        "noise_enabled": bool(noise_enabled),
                        "noise_cv_offline": float(noise_cv_offline),
                        "noise_cv_online": float(noise_cv_online),
                        "noise_seed": int(noise_seed),
                        "overlay_vvm": float(overlay_vvm),
                    }
                    df = run_simulation(f"{batch_name}_{scale}", scale, process_params, advanced_params if enable_advanced else None)
                    if df is not None:
                        results.append(df)

                if results:
                    st.session_state['results'] = results

        _render_results_panel()

    elif mode == "DoE CSV Runner":
        st.header("📂 DoE CSV Runner")
        st.markdown("Upload a CSV with columns like in the **DoE Generator** output (e.g., `BatchName, Scale, pH_setpoint, DO_setpoint_pct, kLa, ...`).")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        apply_overrides = {}
        with st.expander("Optional global overrides (applied to all rows)"):
            if st.checkbox("Apply overrides to all runs", value=False):
                apply_overrides = {
                    "sampling_enabled": st.checkbox("Enable measurement markers", value=True),
                    "sampling_mode": "Lab",
                    "offline_interval": st.number_input("Offline lab sampling interval (hours)", 6, 72, 24, step=6),
                    "online_interval": st.number_input("Online sensor sampling interval (minutes)", 1, 60, 15, step=2),
                    "noise_enabled": st.checkbox("Add measurement noise", value=False),
                    "noise_cv_offline": st.slider("Offline CV (%)", 0.0, 20.0, 5.0, step=0.5)/100.0,
                    "noise_cv_online": st.slider("Online CV (%)", 0.0, 10.0, 2.0, step=0.5)/100.0,
                    "noise_seed": st.number_input("Noise seed", 0, 9999, 42, step=1),
                    "overlay_vvm": st.slider("Overlay airflow (vvm)", 0.0, 0.20, 0.03, step=0.005),
                }
        go_btn = st.button("▶️ Run DoE")
        if go_btn:
            if uploaded is None:
                st.error("Please upload a CSV first.")
            else:
                doe_df = pd.read_csv(uploaded)
                results, as_run, problems = run_batch(doe_df, apply_overrides if apply_overrides else None)
                if len(problems):
                    st.warning("Some rows were skipped due to validation errors.")
                    st.dataframe(problems, use_container_width=True)

                if results:
                    st.session_state['results'] = results
                    st.session_state['batch_name'] = uploaded.name.replace(".csv","")
                    _render_results_panel()
                    st.subheader("As-Run Parameter Table")
                    st.dataframe(as_run, use_container_width=True)
                    # Downloads
                    combined_df = pd.concat(results, ignore_index=True)
                    safe_name = re.sub(r"[^\w\-.]+", "_", st.session_state['batch_name'])
                    st.download_button("📄 Download All Results (CSV)",
                                       data=combined_df.to_csv(index=False),
                                       file_name=f"{safe_name}.csv",
                                       mime="text/csv")

    else:  # DoE Generator
        st.header("🧪 DoE Generator")
        scale = st.selectbox("Scale", ["2L", "50L", "2000L"], index=1)
        preset = SCALE_PRESETS[scale]
        colA, colB, colC = st.columns(3)
        with colA:
            n_runs = st.number_input("Number of runs", 2, 200, 24, step=1)
            seed = st.number_input("Random seed", 0, 10_000, 123, step=1)
        with colB:
            feed_rel_policy = st.selectbox("Feed relative policy", ["Tertiles", "Fixed 1.0", "Cycle 0.8/1.0/1.2"])
            antifoam_mode = st.selectbox("Antifoam pattern", ["Alternate", "PEG only", "silicone only"])
        with colC:
            cin_mM = st.number_input("Feed glucose, cin (mM)", 100, 4000, 2220, step=20)

        st.markdown("**Ranges**")
        r = {}
        for k in ["pH_setpoint","DO_setpoint_pct","kLa","inoc_density_1e6_per_mL","temp_shift_time_h",
                  "phase1_start_h","phase1_vvd_per_day","phase2_start_h","phase2_vvd_per_day"]:
            lo, hi = preset[k]
            c1, c2 = st.columns(2)
            with c1:
                lo_user = st.number_input(f"{k} min", value=float(lo), key=f"{k}_lo")
            with c2:
                hi_user = st.number_input(f"{k} max", value=float(hi), key=f"{k}_hi")
            r[k] = (float(lo_user), float(hi_user))
        r["delta_phase_h_min"] = preset["delta_phase_h_min"]

        gen_btn = st.button("🎲 Generate DoE table")
        if gen_btn:
            doe = generate_doe(scale, int(n_runs), int(seed), r,
                               feed_rel_policy=feed_rel_policy,
                               antifoam_mode=antifoam_mode,
                               cin_mM=int(cin_mM))
            st.dataframe(doe, use_container_width=True)
            st.download_button("💾 Download DoE CSV",
                               data=doe.to_csv(index=False),
                               file_name=f"DoE_{scale}_{n_runs}runs.csv",
                               mime="text/csv")

            st.markdown("---")
            st.subheader("Run the generated DoE now?")
            if st.button("▶️ Simulate Generated DoE"):
                results, as_run, problems = run_batch(doe)
                if len(problems):
                    st.warning("Some rows were skipped due to validation errors.")
                    st.dataframe(problems, use_container_width=True)
                if results:
                    st.session_state['results'] = results
                    st.session_state['batch_name'] = f"DoE_{scale}"
                    _render_results_panel()

if __name__ == "__main__":
    main()

import re
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp

# =========================
# Core ODE Model (now includes TAN + Alanine)
# =========================
def bioreactor_odes_with_scale(t, y, scale_params):
    """
    CHO model with density-dependent death and space limitations,
    plus secondary metabolites: total ammonia (TAN) and alanine.
    """
    # Unpack states (all concentrations are per L unless noted)
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
    # Params (light defaults; can be overridden via advanced UI)
    y_NH3_glc       = scale_params.get("y_NH3_glc", 0.05)      # mmol NH3 per mmol glc-equivalent
    y_NH3_growth    = scale_params.get("y_NH3_growth", 0.0)     # extra NH3 per net growth
    k_strip_NH3     = scale_params.get("k_strip_NH3", 0.01)     # 1/h
    K_NH3_g         = scale_params.get("K_NH3_g", 4.0)          # mM
    K_NH3_qp        = scale_params.get("K_NH3_qp", 6.0)         # mM
    use_ala_penalty = scale_params.get("ala_penalty", False)
    y_Ala_lac       = scale_params.get("y_Ala_lac", 0.05)       # mmol Ala per mmol lactate overflow
    k_Ala_lowDO     = scale_params.get("k_Ala_lowDO", 2e-12)    # mmol/(cell·h) at 0% DO
    q_Ala_cons_max  = scale_params.get("q_Ala_cons_max", 1e-12) # mmol/(cell·h)
    K_Ala           = scale_params.get("K_Ala", 1.0)
    K_Ala_g         = scale_params.get("K_Ala_g", 20.0)
    K_Ala_qp        = scale_params.get("K_Ala_qp", 20.0)

    # Growth/productivity penalties
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

    # Uptake/production fluxes already used elsewhere
    glucose_consumption_rate = ((mu - mu_d) / Yxglc + mglc) * C_X            # mmol/L/h
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
    dC_Ab_dt  = max(0, Qp_eff * C_X * 1000.0 - Fin / V * C_Ab)  # g/L/h
    dV_dt     = max(0, Fin)
    dC_DO_dt  = OTR - OUR - Fin / V * C_DO
    dpCO2_dt  = CO2_prod - CO2_strip - Fin / V * pCO2
    dC_NH3_dt = r_NH3 - k_strip_NH3 * C_NH3 - Fin / V * C_NH3
    dC_Ala_dt = r_Ala - uptake_Ala - Fin / V * C_Ala

    return [dC_X_dt, dC_glc_dt, dC_lac_dt, dC_Ab_dt, dV_dt, dC_DO_dt, dpCO2_dt, dC_NH3_dt, dC_Ala_dt]


# =========================
# Sampling helper (adds *_meas columns without altering continuous series)
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


# =========================
# Measurement noise helper (creates *_meas_noisy; leaves others untouched)
# =========================
def _apply_measurement_noise(df, enable=False, cv_offline=0.05, cv_online=0.02, seed=42):
    if not enable:
        return df
    rng = np.random.default_rng(seed)
    # Identify measured columns
    meas_cols = [c for c in df.columns if c.endswith("_meas")]
    for col in meas_cols:
        vals = df[col].to_numpy()
        mask = ~np.isnan(vals)
        if mask.sum() == 0:  # nothing to do
            continue
        # classify as offline vs online
        is_online = col.startswith("C_DO") or col.startswith("pCO2")
        cv = cv_online if is_online else cv_offline
        noise = rng.normal(loc=0.0, scale=cv, size=mask.sum())
        perturbed = vals[mask] * (1.0 + noise)
        # clip to non-negative (and DO <= saturation)
        if col.startswith("C_DO"):
            perturbed = np.clip(perturbed, 0.0, 0.30)
        else:
            perturbed = np.clip(perturbed, 0.0, None)
        df[col.replace("_meas", "_meas_noisy")] = np.nan
        df.loc[mask, col.replace("_meas", "_meas_noisy")] = perturbed
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
        # secondary metabolite params / toggles
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

        # Keep raw VCD for IVCD/qP
        df["C_X_raw_cells_per_ml"] = df["C_X"].copy()
        df["C_X"] /= 1e6  # display as 10^6 cells/mL

        # Realistic sampling markers
        if process_params.get("sampling_enabled", False):
            df = _attach_sampling_markers(
                df, t_end_hours=t_end, enabled=True,
                mode=("Hybrid" if process_params.get("sampling_mode") == "Hybrid" else "Lab"),
                offline_hours=int(process_params.get("offline_interval", 24)),
                online_minutes=int(process_params.get("online_interval", 15)),
            )
            # Optional: apply measurement noise to *_meas
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

        # Metadata (safe broadcasting)
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

    # TAN metrics
    max_TAN = df["C_NH3"].max() if "C_NH3" in df.columns else 0.0
    time_TAN_high = 0.0
    if "C_NH3" in df.columns:
        t = df["time"].values
        mask = df["C_NH3"].values > tan_thresh
        if mask.any():
            time_TAN_high = np.trapz(mask.astype(float), t)  # hours above threshold (approx)

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
# Streamlit App
# =========================
def main():
    st.set_page_config(page_title="CHO Bioprocess DoE Platform", page_icon="🧬", layout="wide")
    st.title("CHO Bioprocess Design of Experiments Platform")
    st.markdown("**Death-focused VCD control with preserved productivity** + TAN & Alanine")

    st.sidebar.title("Navigation")
    mode = st.sidebar.selectbox("Select Mode", ["Single Run Simulation", "DoE Generator"])

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
            pO2_setpoint = st.slider("pO₂ Setpoint (%)", 20, 60, 40)

            antifoam = st.checkbox("Use Antifoam", value=True)
            antifoam_type = st.selectbox("Antifoam Type", ["PEG", "silicone"]) if antifoam else "PEG"

            # Feed profile UI
            with st.expander("Feed Profile (VVD schedule)"):
                use_custom_feed = st.checkbox("Customize feed schedule", value=True)
                feed_glucose = st.number_input("Feed glucose concentration, cin (mmol/L)", 100, 1000, 722, step=10)
                if use_custom_feed:
                    n_phases = st.number_input("Number of feed phases", 1, 6, 4, step=1)
                    default_starts = [0, 60, 150, 220, 260, 300]
                    default_vvds   = [0.0, 0.05, 0.06, 0.065, 0.07, 0.07]
                    feed_profile = []
                    last_start = 0
                    for i in range(n_phases):
                        start = st.number_input(f"Phase {i+1} start hour", 0, 336, default_starts[i] if i < len(default_starts) else last_start+24, step=6)
                        vvd   = st.number_input(f"Phase {i+1} VVD (1/day)", 0.0, 0.2, float(default_vvds[i] if i < len(default_vvds) else default_vvds[-1]), step=0.005, format="%.3f")
                        feed_profile.append((int(start), float(vvd)))
                        last_start = int(start)
                    feed_profile = sorted(feed_profile, key=lambda p: p[0])
                else:
                    feed_profile = None

            # Secondary metabolites UI
            with st.expander("Secondary metabolites (TAN & Alanine)"):
                ala_penalty = st.checkbox("Apply alanine penalty to growth/productivity", value=False)
                st.caption("Defaults for TAN/Ala kinetics are reasonable; tweak later if needed.")

            # Realistic measurement frequency
            with st.expander("Realistic Measurement Frequency"):
                sampling_enabled = st.checkbox("Enable measurement markers", value=True)
                if sampling_enabled:
                    sampling_mode = st.radio(
                        "Mode",
                        ["Lab (offline+online at intervals)", "Hybrid (offline sampled, online continuous)"],
                        index=0
                    )
                    sampling_mode_token = "Lab" if sampling_mode.startswith("Lab") else "Hybrid"
                    offline_interval = st.number_input("Offline lab sampling interval (hours)", 12, 48, 24, step=6)
                    online_interval  = st.number_input("Online sensor sampling interval (minutes)", 5, 60, 15, step=5)
                else:
                    sampling_mode_token, offline_interval, online_interval = "Lab", 24, 15

            # Measurement noise toggle
            with st.expander("Measurement Noise (applied to measured points only)"):
                noise_enabled = st.checkbox("Add measurement noise", value=False)
                noise_cv_offline = st.slider("Offline CV (%, VCD/titer/metabolites)", 0.0, 20.0, 5.0, step=0.5) / 100.0
                noise_cv_online  = st.slider("Online CV (%, DO/pCO₂)", 0.0, 10.0, 2.0, step=0.5) / 100.0
                noise_seed       = st.number_input("Noise seed", 0, 9999, 42, step=1)

            # Advanced engineering params
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

                # Cards
                c1, c2, c3, c4 = st.columns(4)
                final_titers = [calculate_summary_metrics(df)['final_titer'] for df in results]
                peak_vcds    = [calculate_summary_metrics(df)['peak_VCD'] for df in results]
                max_pco2s    = [calculate_summary_metrics(df)['max_pCO2'] for df in results]
                growths      = sum([calculate_summary_metrics(df)['growth_arrest'] for df in results])
                c1.metric("Best Titer", f"{max(final_titers):.2f} g/L")
                c2.metric("Max VCD", f"{max(peak_vcds):.1f} ×10⁶/mL")
                c3.metric("Max pCO₂", f"{max(max_pco2s):.0f} mmHg")
                c4.metric("Growth Arrests", f"{growths}/{len(results)}")

                # Plots
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
                    scale = df["scale"].iloc[0]; color = colors[scale]
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

                # Overlay measurement markers (use noisy if present)
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

                # Event lines
                for r in range(1,5):
                    for c in range(1,4):
                        fig.add_vline(x=72, line_dash="dash", line_color="black", opacity=0.5, row=r, col=c)
                        fig.add_vline(x=96, line_dash="dot",  line_color="purple", opacity=0.5, row=r, col=c)

                fig.update_layout(height=1200, showlegend=True,
                                  title_text="Comprehensive CHO Bioprocess Analysis (with TAN & Alanine)")
                fig.update_xaxes(title_text="Time (h)")
                st.plotly_chart(fig, use_container_width=True)

                # Downloads
                st.subheader("💾 Download Results")
                colA, colB = st.columns(2)
                combined_df = pd.concat(results, ignore_index=True)
                combined_df[batch_name] = batch_name
                metrics_df[batch_name] = batch_name
                safe_name = re.sub(r"[^\w\-.]+", "_", batch_name)
                with colA:
                    st.download_button("📄 Download Complete Dataset",
                        data=combined_df.to_csv(index=False), file_name=f"{safe_name}.csv", mime="text/csv")
                with colB:
                    st.download_button("📊 Download Summary Metrics",
                        data=metrics_df.to_csv(index=False), file_name=f"{safe_name}_summary.csv", mime="text/csv")

if __name__ == "__main__":
    main()

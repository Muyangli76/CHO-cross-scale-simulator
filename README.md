# CHO Cross-Scale Simulator

Monod-based kinetic CHO **fed‑batch** simulator with scale-dependent mass transfer, mixing and shear effects.  
Models **VCD**, **glucose**, **lactate**, **product**, **DO**, **pCO₂**, **TAN (NH₃/NH₄⁺)**, **alanine**, and **volume** across **2 L / 50 L / 2000 L**.

> Purpose: generate realistic synthetic trajectories for ML pre‑training, transfer learning, and DoE prototyping.

## ✨ Highlights
- Mechanistic core (growth, death, qP–μ coupling) + empirical scale effects (kLa, CO₂ stripping, EDR, mixing).
- Secondary metabolites: **TAN** inhibition and **alanine** overflow under low DO.
- Realistic feeding: 1–2 phase VVD schedule, or **CSV‑driven** per‑run overrides.
- Measurement realism: optional sampling cadence and noise (lab vs. online).

## Quickstart
```bash
# clone
git clone https://github.com/Muyangli76/CHO-cross-scale-simulator.git
cd CHO-cross-scale-simulator

# (optional) create a venv, then install
pip install -r requirements.txt

# run
streamlit run app.py
```

## Run modes
- **Single Run Simulation** – interactive sliders for scale & process setpoints.
- **Run from DoE CSV** – upload a plan and batch‑simulate.

### DoE CSV (explicit 2‑phase feed schedule)
Required columns (typical):
```
BatchName, Scale, num_feed_phases,
phase1_start_h, phase1_vvd_per_day,
phase2_start_h, phase2_vvd_per_day,
feed_VVD_relative,
pH_setpoint, DO_setpoint_pct, kLa,
inoc_density_1e6_per_mL, temp_shift_time_h, antifoam_type
```
> Example file: `examples/DoE_2L_35run_plan_with_schedule.csv`

Notes:
- `kLa` may be supplied directly; otherwise map from `DO_setpoint_pct` in the app.
- `feed_VVD_relative` scales both phases together (e.g., 0.8 / 1.0 / 1.2).
- UI “cin” is the **feed glucose concentration**; optionally add a glucose target controller later.

## Scale presets (defaults)
- **2 L:** kLa≈280 h⁻¹, mixing_eff=1.0, EDR≈0.02–0.05 W/kg, volume_factor≈0.02  
- **50 L:** kLa≈90 h⁻¹,  mixing_eff=0.90, EDR≈0.05 W/kg,   volume_factor=1.0  
- **2000 L:** kLa≈35 h⁻¹, mixing_eff=0.65, EDR≈0.15 W/kg,  volume_factor=4.0

## References
See [`docs/references.md`](docs/references.md) (IEEE style).

## License
MIT

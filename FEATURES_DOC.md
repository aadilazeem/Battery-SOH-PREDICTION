# Feature Documentation (Physics-Informed)

This document describes each per-cycle feature produced by the pipeline (`features_primary.csv`).

---

- **Feature name:** `n_points`
  - **Formula / computation:** number of time samples in the cycle.
  - **Units:** count
  - **Physical meaning:** measurement density; not an intrinsic battery property but affects reliability of derived features.
  - **Expected trend with aging:** typically constant (instrument dependent).
  - **Confounders:** sampling frequency changes, truncated cycles.

- **Feature name:** `duration_s`
  - **Formula:** Time[-1] - Time[0]
  - **Units:** seconds
  - **Physical meaning:** how long the cycle lasted; relates to charge/discharge C-rate and total throughput.
  - **Expected trend:** may increase if currents drop; higher duration at same capacity implies lower C-rate.
  - **Confounders:** pauses in logging, missing points, different test protocols.

- **Feature name:** `voltage_start`, `voltage_end`, `voltage_delta`
  - **Formula:** V[0], V[-1], V[-1] - V[0]
  - **Units:** Volts (V)
  - **Physical meaning:** simple summary of voltage shift during the cycle; delta reflects polarization and SOC change during cycle.
  - **Expected trend:** end-start for discharge may reduce as capacity fades; larger polarization can indicate internal resistance rise (LAM/SEI).
  - **Confounders:** SOC window differences, measurement offset.

- **Feature name:** `voltage_slope_V_per_s` (global) and `discharge_voltage_slope`
  - **Formula:** linear regression slope of V vs t (global or on discharge segment)
  - **Units:** V / s
  - **Physical meaning:** slope captures how quickly voltage changes during the cycle. On discharge, a steeper drop (more negative slope) often indicates increased resistance or loss of active material affecting the usable voltage window.
  - **Expected trend:** as aging proceeds, discharge slope magnitude can increase (faster drop) due to LLI/SEI and increased internal resistance.
  - **Confounders:** different current profiles/C-rate, segment mixing (hence discharge-only slope preferred).

- **Feature name:** `voltage_std`
  - **Formula:** std(V) across cycle
  - **Units:** V
  - **Physical meaning:** variability of the voltage trace; high variability may indicate noisy measurements or dynamic processes (e.g., relaxation steps, plateaus)
  - **Expected trend:** depends — could increase with noise or if cycling protocol introduces more transient events.
  - **Confounders:** measurement noise, logging artifacts.

- **Feature name:** `current_mean`, `current_std`
  - **Formula:** mean(I), std(I)
  - **Units:** Amps (A)
  - **Physical meaning:** average current and its variability across the cycle; relates to C-rate and load conditions.
  - **Expected trend:** protocol-dependent; higher mean current generally accelerates aging (higher stress) — useful as covariate.
  - **Confounders:** mixed charge/discharge within cycle, switching protocols.

- **Feature name:** `temp_start`, `temp_end`, `temp_change`
  - **Formula:** T[0], T[-1], T[-1] - T[0]
  - **Units:** degrees Celsius (°C)
  - **Physical meaning:** thermal conditions and intra-cycle heating — thermal excursions accelerate degradation and can indicate internal resistance heating.
  - **Expected trend:** might slowly increase across cycles if thermal management is poor; spikes can indicate local heating events.
  - **Confounders:** sensor placement, ambient temperature, logging delays.

- **Feature name:** `energy_proxy_VAs`
  - **Formula:** sum_over_steps( V[t] * I[t] * dt ) (approximate integral of V*I over cycle)
  - **Units:** V * A * s (equivalent to Joules) — used as an energy throughput proxy
  - **Physical meaning:** estimate of electrical work processed during the cycle; higher throughput contributes to capacity fade (LLI) and calendar aging acceleration.
  - **Expected trend:** cumulative energy per cycle may remain similar under same protocol; strong correlation with capacity loss across cycles if usage increases.
  - **Confounders:** missing samples, sign convention (I negative for discharge), need to interpret charge vs discharge separately.

- **Feature name:** `capacity` (if available) and `capacity_ratio`
  - **Formula:** reported/estimated cycle capacity; `capacity_ratio` = capacity_current / capacity_initial
  - **Units:** Ampere-hours (Ah) or relative unit (ratio)
  - **Physical meaning:** direct measure of usable charge — primary SOH metric. Decline is LLI/active loss.
  - **Expected trend:** monotonic decrease with cycle count (general capacity fade). Knee points may indicate onset of rapid failure.
  - **Confounders:** measurement SOC window, temperature dependence, test protocol.

- **Feature name:** `discharge_frac`
  - **Formula:** fraction of samples where I < 0 (discharge mask mean)
  - **Units:** fraction (0–1)
  - **Physical meaning:** how much of the cycle is discharge vs charge; useful to detect cycles dominated by discharge (protocol) or partial cycles.
  - **Expected trend:** protocol dependent; useful to normalize features per discharge portion.
  - **Confounders:** sign convention for current, mixed segments.

- **Feature name:** `discharge_curvature_coeff`
  - **Formula:** coefficient `a` from quadratic fit V = a*t^2 + b*t + c on discharge segment
  - **Units:** V / s^2
  - **Physical meaning:** captures curvature of the discharge curve (nonlinear drop). Increased curvature can reflect diffusion limitations, concentration polarization, or evolving internal resistance profiles.
  - **Expected trend:** curvature magnitude may increase as diffusion-related effects (SEI growth, LLI) worsen.
  - **Confounders:** insufficient points for stable quadratic fit, non-uniform sampling, changes in current profile.

- **Feature name:** `voltage_plateau_fraction`
  - **Formula:** fraction of discharge points where |dV/dt| < threshold (small absolute slope)
  - **Units:** fraction (0–1)
  - **Physical meaning:** quantifies how much of the discharge has nearly-flat voltage (plateau). Plateaus are linked to phase transitions and capacity segments.
  - **Expected trend:** plateau fraction can shrink as active material is lost or as the usable voltage window narrows.
  - **Confounders:** threshold sensitivity, noisy derivative estimates.

---

**Where this table is saved / how to use it**
- The pipeline saves the features matrix at `4_Results_and_Graphs/features_primary.csv`.
- Use this table as the ML input (one row per cycle). For modeling SOH/EOL prefer training on cycles with valid `capacity_ratio` values, and include `current_mean` or `duration_s` as covariates for C-rate normalization.

**Suggested short captions for figures (use in report):**
- Capacity_ratio vs Cycle: "Capacity ratio declines over cycles; a knee-point indicates onset of accelerated LLI—investigate cycles where ratio drop accelerates." 
- Voltage_slope vs Cycle: "Increasing magnitude of discharge voltage slope indicates rising polarization and likely resistance growth (SEI/LAM)."

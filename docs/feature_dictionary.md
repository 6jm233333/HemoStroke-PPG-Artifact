# Hemodynamic Feature Dictionary

This document provides the human-readable definition of the 17 photoplethysmography-derived hemodynamic features used in the study.

These features are derived from the raw photoplethysmography (PPG) signal, the velocity plethysmogram (VPG), and the acceleration plethysmogram (APG). They are organized into four groups: time-domain and morphological features, derivative and normalized features, short-term variability features, and relative deviation features.

## I. Time-Domain and Morphological Features

| Symbol | Definition | Physiological significance |
|---|---|---|
| T<sub>sp</sub> | Time to systolic peak. Time elapsed from pulse onset to the systolic peak. | Reflects the velocity of blood ejection and wave transmission. It is related to arterial compliance and may shorten with arterial stiffening. |
| SI | Stiffness index. A stiffness-related index derived from pulse morphology. | Represents systolic upstroke characteristics and arterial stiffness. A higher value may indicate increased arterial stiffness. |
| A<sub>off</sub> | Offset amplitude. Signal amplitude at the pulse offset point. | Reflects vascular recoil at end-diastole and may relate to vessel elasticity and venous return. |
| T<sub>sys</sub>/T<sub>dia</sub> | Systolic-diastolic ratio. Ratio of systolic duration to diastolic duration. | Describes the temporal balance of the cardiac cycle. Abnormal balance may reflect autonomic dysfunction or altered vascular resistance. |

## II. Derivative and Normalized Features

| Symbol | Definition | Physiological significance |
|---|---|---|
| T<sub>u(Tpi)</sub> | Normalized time to VPG peak. Time to the VPG u point normalized by pulse interval T<sub>pi</sub>. | Reflects the relative duration of the rapid ejection phase and may relate to vascular impedance and flow inertia. |
| T<sub>b(Tpi)</sub> | Normalized time to APG b-wave. Time to the APG b wave normalized by pulse interval T<sub>pi</sub>. | Sensitive to vascular aging and early peripheral wave reflection changes. |
| T<sub>v</sub> | Time to VPG valley. Time to the post-systolic VPG inflection point. | Correlates with the timing of reflected wave return and central-peripheral hemodynamic coupling. |
| T<sub>u(Ta,Tpi)</sub> | VPG-APG timing relation. Timing relation between the VPG u point, APG timing T<sub>a</sub>, and pulse interval T<sub>pi</sub>. | Describes the coupling between velocity and acceleration components of the PPG waveform and may reflect neurovascular mismatch. |

## III. Short-Term Variability Features

| Symbol | Definition | Physiological significance |
|---|---|---|
| CV<sub>T,pi</sub> | Pulse interval variability. Coefficient of variation of pulse interval T<sub>pi</sub>. | Serves as an ultra-short-term heart-rate-variability surrogate. Reduced variability may indicate autonomic imbalance. |
| CV<sub>PA</sub> | Pulse amplitude variability. Coefficient of variation of pulse amplitudes. | Reflects beat-to-beat variability in peripheral pulse amplitude and the capacity of sympathetic regulation over peripheral vascular tone. |

## IV. Relative Deviation Features

| Symbol | Definition | Physiological significance |
|---|---|---|
| T<sub>sp,Rel</sub> | Relative time to systolic peak. Deviation of current T<sub>sp</sub> from the individual baseline. | Captures progressive drift in wave transmission time and reduces inter-subject baseline variability. |
| A<sub>sp,Rel</sub> | Relative systolic amplitude. Deviation of systolic peak amplitude A<sub>sp</sub> from the individual baseline. | Sensitive to acute changes in vascular compliance and stroke-volume-related pulse morphology. |
| SI<sub>Rel</sub> | Relative stiffness index. Deviation of SI from the individual baseline. | Tracks patient-specific change in arterial stiffness or compliance degradation. |
| DSI<sub>Rel</sub> | Relative dynamic stability index. Deviation of dynamic stability index from the individual baseline. | Quantifies shifts in waveform morphology and may reflect loss of hemodynamic homeostasis. |
| T<sub>c,Rel</sub> | Relative APG c-wave timing. Shift in late-systolic c-wave timing from baseline. | Relates to late-systolic recoil and may indicate altered arterial wall elasticity. |
| A<sub>off,Rel</sub> | Relative offset amplitude. Deviation in end-diastolic amplitude from baseline. | Associated with changes in venous return and diastolic regulation. |
| A<sub>on,Rel</sub> | Relative onset amplitude. Deviation in pulse-onset nadir from baseline. | Sensitive to peripheral perfusion and microcirculatory abnormalities. |

## Relative Deviation Formula

For a feature x(t), the relative deviation feature is computed with respect to a subject-specific baseline:

`x_Rel(t) = [x(t) - mu_base] / |mu_base|`

where `mu_base` denotes the mean value of the corresponding feature during the subject-specific baseline window.

## Notes

The relative deviation features are computed with respect to a subject-specific baseline to reduce inter-patient heterogeneity and capture patient-level hemodynamic trajectories.

This document is provided as an online feature dictionary for reproducibility. It is not a substitute for the paper, and restricted clinical data are not redistributed in this repository.

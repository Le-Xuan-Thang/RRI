# Project Plan: Reliability-Robustness Index (RRI) for Cultural Heritage Buildings

## Overview
This project implements a 7-step workflow to assess and predict the Reliability-Robustness Index (RRI) of cultural heritage buildings using a Digital Twin (DT), advanced FE simulation, and deep learning. The plan is structured for clarity, reproducibility, and scientific rigor.

---

## 1. Digital Twin (DT) Setup & Calibration
- **Objective:** Build and calibrate a finite element (FE) model using STABIL (KULeuven).
- **Tasks:**
  - Create FE model of the structure.
  - Define random parameters: E (Log-Normal), ρ (Normal), RH/T (GEV), qw (Gumbel), δdeg (Gamma).
  - (Optional) Calibrate E, ρ using measured frequencies/mode shapes (model updating) to <5% error.

## 2. Define Limit States & State Functions
- **Objective:** Establish failure criteria for reliability analysis.
- **Tasks:**
  - Implement limit state functions: g1 = R - S (strength), g2 = θmax - θallow (rotation).
  - Support selective MAX for multiple state functions.

## 3. Compute Reliability Index (β)
- **Objective:** Quantify structural reliability under uncertainty.
- **Tasks:**
  - For each random parameter vector, run FE simulation and compute limit states.
  - Estimate failure probability Pf using FORM or Subset Simulation.
  - Calculate β = -norminv(Pf).

## 4. Compute Robustness Index (RI) & RRI
- **Objective:** Assess robustness to local damage and combine with reliability.
- **Tasks:**
  - Generate local damage scenarios (e.g., remove column, reduce stiffness).
  - Recompute βd for each scenario.
  - Compute RI = 1 - (Pf,d / Pf,0).
  - Combine: RRI = w1*β + w2*RI (w1=0.6, w2=0.4 recommended).

## 5. Generate Dataset for Deep Learning
- **Objective:** Create a comprehensive dataset for training and evaluation.
- **Tasks:**
  - Simulate 10,000 samples: save FE response, parameter vector, and RRI.
  - Structure: Input (response), Metadata (params), Output (RRI).

## 6. Train Rapid-RRI-Net
- **Objective:** Develop a deep learning model for rapid RRI prediction.
- **Tasks:**
  - Architecture: 1D-CNN → LSTM → ResNet → Dense → Output.
  - Pre-train on simulated data, fine-tune on real data.
  - Use MSE loss, Adam optimizer.

## 7. Evaluation & Inference
- **Objective:** Validate model accuracy and speed for real-time application.
- **Tasks:**
  - Assess MAE (<0.02), inference speed, and domain-gap robustness.

---

## File Structure (Recommended)
- `DT_setup.m` – Digital Twin setup and calibration
- `limit_states.m` – Limit state function definitions
- `compute_reliability.m` – Reliability index calculation
- `compute_robustness.m` – Robustness and RRI calculation
- `generate_dataset.m` – Dataset generation for deep learning
- `export_for_dl.m` – Data export for deep learning
- `evaluate_model.m` – Model evaluation and inference
- `plan.md` – This project plan

---

## Notes
- All data and formulas must be scientifically justified and reproducible.
- Document all formulas and reference sources clearly.
- Follow best practices for code modularity and documentation.

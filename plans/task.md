# Task List for RRI Project

## 1. Digital Twin (DT) Setup & Calibration
- [ ] Build FE model of the structure using STABIL
- [ ] Define random parameters: E (Log-Normal), ρ (Normal), RH/T (GEV), qw (Gumbel), δdeg (Gamma)
- [ ] (Optional) Calibrate E, ρ using measured frequencies/mode shapes (model updating, <5% error)

## 2. Define Limit States & State Functions
- [ ] Implement limit state functions: g1 = R - S (strength), g2 = θmax - θallow (rotation)
- [ ] Support selective MAX for multiple state functions

## 3. Compute Reliability Index (β)
- [ ] For each random parameter vector, run FE simulation and compute limit states
- [ ] Estimate failure probability Pf using FORM or Subset Simulation
- [ ] Calculate β = -norminv(Pf)

## 4. Compute Robustness Index (RI) & RRI
- [ ] Generate local damage scenarios (e.g., remove column, reduce stiffness)
- [ ] Recompute βd for each scenario
- [ ] Compute RI = 1 - (Pf,d / Pf,0)
- [ ] Combine: RRI = w1*β + w2*RI (w1=0.6, w2=0.4 recommended)

## 5. Generate Dataset for Deep Learning
- [ ] Simulate 10,000 samples: save FE response, parameter vector, and RRI
- [ ] Structure: Input (response), Metadata (params), Output (RRI)

## 6. Train Rapid-RRI-Net
- [ ] Build architecture: 1D-CNN → LSTM → ResNet → Dense → Output
- [ ] Pre-train on simulated data, fine-tune on real data
- [ ] Use MSE loss, Adam optimizer

## 7. Evaluation & Inference
- [ ] Assess MAE (<0.02), inference speed, and domain-gap robustness

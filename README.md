# Microscope Drift Analysis

Tools for analyzing and quantifying sample drift in scanning microscopy (e.g. two-photon microscopy).

This repository implements:
- Localization of centers of large (4-micron diameter) fluorescent beads
- Removal of the drift with slowly-changing speed
- Maximum likelihood estimation (MLE) diffusion coefficients of beads

## 📌 Motivation

Microscope drift (thermal, mechanical, or piezo-induced) limits spatial
precision in high-resolution imaging. This repository provides a lightweight
framework to:

- Separate drift from stochastic motion
- Quantify noise and diffusion coefficient to provide benchmark for single particle tracking with scanning microscopy. 


## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/drkutuzov/microscope-drift
cd microscope-drift
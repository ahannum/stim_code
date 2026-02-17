# stim_code  
**PNS & CNS-Constrained Optimized Diffusion Encoding for Diffusion-Weighted Imaging**

This repository contains code for designing and implementing **Stim-CODE** diffusion waveforms, including simulation, sequence generation, reconstruction, processing, and figure creation.

---

## Repository Structure

### 1. Simulation_Code  
Contains code for generating Stim-CODE diffusion waveforms under PNS and CNS constraints.  
The generated waveforms are saved in a format compatible with Pulseq sequences for subsequent use in pulse sequence design.

### 2. Sequence_Code  
MATLAB scripts for generating diffusion-weighted imaging (DWI) and diffusion tensor imaging (DTI) sequences.  
This section also exports sequence timing parameters required for waveform simulation and constraint evaluation.

### 3. Experiment_Code  

#### 00_Reconstruction  
MATLAB scripts for reconstructing raw Siemens MRI data.

#### 01_Processing  
Python scripts for registration and processing of DWI and DTI data, including computation of diffusion-derived metrics.

#### 02_Figures  
Code for generating the figures associated with the manuscript.

---

## Overview

Stim-CODE enables the design of optimized diffusion encoding waveforms that satisfy peripheral nerve stimulation (PNS) and central nervous system (CNS) safety constraints, while achieving target diffusion weighting for advanced diffusion MRI applications.
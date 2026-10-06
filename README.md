# Stim-CODE: PNS and CNS Constraint-Optimized Diffusion Encoding

Stim-CODE extends the [GrOpt](https://github.com/cmr-group/gropt-dev/tree/main/gropt) toolbox to enable **peripheral nerve stimulation (PNS)** and **cardiac nerve stimulation (CNS)** constraint–optimized diffusion-encoding waveform design.

It provides tools for generating diffusion-encoding gradients in MRI that satisfy hardware, sequence, and physiological constraints. The SAFE model (Hebrank, ISMRM, 2000) is incorporated to provide vendor-specific PNS/CNS response. 

---

## Installation

Stim-CODE requires Python 3.10 or newer. Clone the repository, create a virtual
environment, and install the dependencies from the repository root:

```bash
git clone https://github.com/ahannum/stim_code.git
cd stim_code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The requirements pin `gropt==2.0.0rc16`, the pre-release version used by the
demo, and include NumPy, SciPy, Matplotlib, and JupyterLab. GrOpt is
installed from PyPI; it does not need to be cloned separately. PyPulseq is
not required: the example gradient waveforms are stored directly in T/m.

To open the demonstration locally, run:

```bash
jupyter lab Examples/demo_colab.ipynb
```

---

## Getting Started

A step-by-step demonstration is available:

- Jupyter notebook: [`Examples/demo_colab.ipynb`](Examples/demo_colab.ipynb)
- Google Colab (interactive): [Open in Colab](https://colab.research.google.com/github/ahannum/stim_code/blob/main/Examples/demo_colab.ipynb)

The demo walks through:
- Generating diffusion-encoding waveforms  
- Applying PNS/CNS constraints as (1) Constant Threshold, (2) Arbitrary envelope, and (3) Envelope based on other gradient events  
- Comparing waveforms to conventional diffusion-encoding

---


## Included .mat Files: Example Pulseq timings

The two MATLAB files in `Examples/` were exported from the Pulseq sequence used
by the demonstration for the sequence timings to design diffusion-encoding gradients. 
These are retained as in the demo we use these timings to make the the example waveforms. Timings in the 
notebook can be replaced if desired. 

`diffusion_timing_parameters.mat` contains these principal timings:

| Parameter | Value | Description |
| --- | ---: | --- |
| `rf_90_duration` | 4.700 ms | Total duration of the excitation pulse |
| `rf_90_rfCenterInclDelay` | 1.865 ms | Duration to Center of Excitation |
| `rf_180_duration` | 7.070 ms | Total Refocusing pulse duration |
| `rf_180_rfCenterInclDelay` | 3.54075 ms | Timing of Refocusing RF center |
| `T_readout` | 13.734 ms | Duration from end of diffusion-encoding to middle of readout |
| `nav_dur` | 0 ms | Duration of three line reference Navigator |

The timing file contains only these six scalars used by the final notebook demo.
Unused gradient definitions, EPI counts, and Pulseq system settings have been removed.

`diffusion_timing_parameters_waveforms.mat` contains four-axis Pulseq waveform
data (`gx`, `gy`, `gz`, and RF) for the following sequence blocks in order to
construct the envelope constraint based on these waveforms: `wave_data_rf90`
(excitation), `wave_data_rf180` (refocusing), `wave_data_epi` (readout), and
`wave_data_fatsat` (fat saturation). Unused RF time/frequency/phase metadata has
been removed. Time and RF samples are preserved; gradient amplitudes are
stored in T/m.

Gradient amplitudes in the waveform file are stored directly in T/m, matching
the optimization and SAFE inputs. `utils/safe_vec_generator.py` loads them
without unit conversion. Each variable
is a 1 x 4 MATLAB cell array ordered `[gx, gy, gz, RF]`; each cell contains
a 2 x N array of block-relative times in seconds and amplitudes (RF in Hz).
The final notebook section documents every retained variable and its use.

---


## References

**Associated work (in preparation):**

Hannum AJ, Loecher M, Chen Q, Arbes E, Setsompop K, Zaitsev M, Ennis DB.  
*Stim-CODE: PNS and CNS Constraint-Optimized Diffusion-Encoding for Neuroimaging on 200 mT/m Whole-Body Gradients.*  
__ (in preparation).

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

The requirements pin `gropt==2.0.0rc11`, the pre-release version used by the
demo, and include NumPy, SciPy, Matplotlib, PyPulseq, and JupyterLab. GrOpt is
installed from PyPI; it does not need to be cloned separately.

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
| `rf_90_duration` | 4.700 ms | Total excitation block duration |
| `rf_90_rfCenterInclDelay` | 1.865 ms | Excitation RF center, including delay |
| `rf_180_duration` | 7.070 ms | Total refocusing block duration |
| `rf_180_rfCenterInclDelay` | 3.54075 ms | Refocusing RF center, including delay |
| `timeToTE` | 13.734 ms | Time reserved for the readout contribution to TE |
| `nav_dur` | 0 ms | Navigator duration |

The same file also includes EPI timings for a 1.5 x 1.5 x 1.5 mm^3 protocol with 6/8 partial FOV including: 55 measured phase-encode lines (`Ny_meas`), 3
navigators (`nNav`), the excitation and refocusing gradient definitions, and
the Pulseq system settings. The arrays are on a raster of 10 us for gradients and
blocks, 1 us for RF, and 0.1 us for ADC samples; the RF dead time is 100 us,
the RF ring-down time is 30 us, and the ADC dead time is 10 us.

`diffusion_timing_parameters_waveforms.mat` contains four-axis Pulseq waveform
data (`gx`, `gy`, `gz`, and RF) for the following sequence blocks in order to
construct the envelope constraint:

| Block | Stored timing extent |
| --- | ---: |
| Fat saturation | 0-15.130 ms; RF active from 3.6005-11.5995 ms |
| RF90 excitation | 0-4.700 ms; RF samples from 0.14345-3.58655 ms |
| RF180 refocusing | 0-7.070 ms; RF samples from 0.66575-6.40425 ms |
| EPI readout | Up to 38.920 ms (`gx`) and 38.270 ms (`gy`) |

Gradient amplitudes in the waveform file use Pulseq's internal Hz/m units and
are converted when loaded by `utils/safe_vec_generator.py`.

---


## References

**Associated work (in preparation):**

Hannum AJ, Loecher M, Chen Q, Arbes E, Setsompop K, Zaitsev M, Ennis DB.  
*Stim-CODE: PNS and CNS Constraint-Optimized Diffusion-Encoding for Neuroimaging on 200 mT/m Whole-Body Gradients.*  
__ (in preparation).

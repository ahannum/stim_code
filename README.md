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

## References

**Associated work (in preparation):**

Hannum AJ, Loecher M, Chen Q, Arbes E, Setsompop K, Zaitsev M, Ennis DB.  
*Stim-CODE: PNS and CNS Constraint-Optimized Diffusion-Encoding for Neuroimaging on 200 mT/m Whole-Body Gradients.*  
__ (in preparation).

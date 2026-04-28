# Stim-CODE: PNS and CNS Constraint-Optimized Diffusion Encoding

Stim-CODE extends the [GrOpt](https://github.com/cmr-group/gropt-dev/tree/main/gropt) toolbox to enable **peripheral nerve stimulation (PNS)** and **cardiac nerve stimulation (CNS)** constraint–optimized diffusion-encoding waveform design.

It provides tools for generating diffusion-encoding gradients in MRI that satisfy hardware, sequence, and physiological constraints. The SAFE model (Hebrank, ISMRM, 2000) is incorporated to provide vendor-specific PNS/CNS response. 

---

## Requirements

Stim-CODE builds on the GrOpt framework:

- GrOpt toolbox:  
  https://github.com/cmr-group/gropt-dev/

Make sure GrOpt is installed and accessible in your Python environment before using this package.

---

## Getting Started

A step-by-step demonstration is available:

- Jupyter notebook: `Examples/demo.ipynb`  
- Google Colab (interactive): *[add link here]*

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

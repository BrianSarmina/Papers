Q-PIPE: Quantum-Gray Phase Injection for Pixel Encoding

This repository contains the source code, datasets, and high-resolution benchmark images for the Q-PIPE algorithm, as presented in our paper. Q-PIPE is a novel quantum algorithmic framework designed to bridge the gap between efficient quantum data representation and active feature extraction for Near-Term Intermediate Scale Quantum (NISQ) devices.

    Overview:
Loading classical high-dimensional data into quantum states is a fundamental bottleneck in Quantum Image Processing (QIMP) and Quantum Machine Learning (QML). Q-PIPE conceptualizes the state preparation phase as a parameter estimation problem. By exploiting the quantum phase kickback mechanism and optimizing the spatial traversal via a Gray-code sequence, Q-PIPE injects continuous intensity values into the relative phase and natively projects them into the computational basis.

This repository demonstrates Q-PIPE applied to Quantum Edge Detection (QED), calculating combined directional gradients directly within the quantum phase domain.

    Key Features:
- Optimal Gate Complexity: Achieves an $O(qN)$ elementary gate count via Gray-code optimization, a strict $\Theta(\log N)$ improvement over standard basis encoding (NEQR).
- Native Arithmetic Operations: Computes finite differences (e.g., Sobel directional gradients) directly during the state preparation stage without requiring deep quantum arithmetic circuits.
- Aliasing Prevention: Prevents cyclical overflow errors by strictly constraining the input amplitude normalization to the half-phase interval $[-\pi, \pi]$.
- Probability Thresholding: Mitigates finite-resolution broadening (spectral leakage) across varying image resolutions using a threshold equation:$$P_{\text{th}} \le \frac{\eta}{2^n \cdot W}$$where $2^n$ is the spatial register dimension, $W$ is the effective Dirichlet kernel width, and $\eta$ is the empirical tolerance factor.

    Requirements
To run the simulations and reproduce the results, you will need the following dependencies:
- Python 3.11+
- NumPy
- Matplotlib
- Scikit-Learn
- Pennylane (latest version)
- Qiskit (latest version)

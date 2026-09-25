Q-PIPE: Quantum-Gray Phase Injection for Pixel Encoding

Source code and benchmark images for the paper
"Q-PIPE: Practical Quantum Phase Encoding with Native Phase-Domain Arithmetic".

    Overview
Q-PIPE treats data loading as a parameter-estimation problem. Pixel intensities are written as
eigenphases of a diagonal oracle through phase kickback, and a single quantum phase estimation
stage converts all phases coherently into computational-basis states (an NEQR-form state).
Because intensities live in the phase domain, sums and differences of images are obtained by
composing oracles, without quantum arithmetic circuits. The repository demonstrates this with
quantum edge detection (QED): directional finite differences and a Sobel-type magnitude.

    Key points
- Two syntheses of the same oracle: (i) Gray-code traversal with one multi-controlled phase
  (C^nP) per pixel, where the Gray code reduces the position-marking X gates from Theta(qN log N)
  to Theta(qN); (ii) uniformly controlled Rz rotations plus one diagonal, which needs
  (q+1)N - 2 + O(q^2) CNOTs without ancillas, the same O(qN) scaling as optimized NEQR.
- Aliasing: each image is mapped onto the half-spectrum [0, pi], so differences stay in
  [-pi, pi] and are decoded without wrap-around.
- Read-out threshold: P_th <= C_D / 2^n with C_D = 1/(pi^2 J^2), derived from the QPE kernel
  (J = 2 gives C_D = 1/(4 pi^2)); the central QPE peak is kept only if P_th < 4/(pi^2 2^n).
- Noisy simulations (IBM fake backends): the C^nP circuits are limited by their compiled depth;
  the equivalent UCR circuits are ~50x shallower and preserve the edge structure.

    Repository structure
Codes/
  q_pipe_base.py                        exploratory notebook export (PennyLane)
  q_pipe_for_quantum_edge_detection.py  ideal QED benchmarks: datasets, estimation-register depth,
                                        image resolution (PennyLane)
  Q_PIPE_different_prob_thresholds.py   probability-threshold sweep (PennyLane)
  Noisy_Simulations/
    qpipe_noisy_simulation.py           noisy QED with the C^nP or UCR oracle (Qiskit)
    plot_noisy_metrics.py               noisy-simulation metrics figure and table rows
    plot_noise_mechanism.py             noise-mechanism figure (both oracles)
  Complexity/
    complexity_resources.py             qubit, CNOT-count and CNOT-depth scaling figure
Data/Noisy_Sims/                        saved results (.pkl) used by the plotting scripts
Images_Results/                         figures

    Running the noisy simulations
The fake backends are synchronized with live IBM Quantum properties, which requires an IBM Quantum
account. Credentials are read from environment variables and must never be committed:
    export QISKIT_IBM_TOKEN="..."
    export QISKIT_IBM_INSTANCE="..."
    python qpipe_noisy_simulation.py --oracle cnp
    python qpipe_noisy_simulation.py --oracle ucr
    python plot_noisy_metrics.py
    python plot_noise_mechanism.py
The plotting scripts only need the .pkl files in the same folder (no account required).
The two exploratory scripts exported from Colab start with "!pip install" lines; run them in a
notebook or remove those lines before running them as Python scripts.

    Requirements
- Python 3.11+
- NumPy, Matplotlib, scikit-learn, scikit-image
- PennyLane (ideal simulations)
- Qiskit 2.x, qiskit-aer, qiskit-ibm-runtime (a version that provides FakeKingston, FakeMarrakesh
  and FakeBerlin) for the noisy simulations

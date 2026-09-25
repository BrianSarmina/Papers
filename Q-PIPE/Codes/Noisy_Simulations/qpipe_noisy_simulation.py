# -*- coding: utf-8 -*-
"""
Q-PIPE -- noisy simulations of horizontal quantum edge detection (paper, Appendix B).

Two oracle constructions of the same unitary are supported:
  --oracle cnp : Gray-code oracle with one multi-controlled phase (C^nP) per pixel (paper, FIG. 1)
  --oracle ucr : uniformly controlled Rz rotations + one diagonal (paper, Sec. III D 3)

Protocol: first five MNIST samples (fetch_openml), 28x28 -> 4x4; half-spectrum phases
(+pi*I/255 for the image, -pi*I_shift/255 for the shifted image); 8 estimation + 4 position qubits;
4096 shots; optimization level 3; reconstruction threshold P_th = 5e-4.
Noise: FakeKingston, FakeMarrakesh and FakeBerlin, synchronized with live device properties.

Credentials are read from environment variables; never write them into this file:
    export QISKIT_IBM_TOKEN="..."      export QISKIT_IBM_INSTANCE="..."

Usage:
    python qpipe_noisy_simulation.py --oracle cnp   # -> qpipe_experiment_data_complete_4x4.pkl
    python qpipe_noisy_simulation.py --oracle ucr   # -> qpipe_experiment_data_complete_4x4_ucr.pkl
Runs are checkpointed after every image and can be resumed.
"""
import os
import argparse
import pickle
import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_absolute_error
from skimage.transform import resize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, UCRZGate, DiagonalGate
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeKingston, FakeMarrakesh, FakeBerlin

N_EST, N_POS, N_IMAGES = 8, 4, 5
SHOTS, P_TH = 4096, 0.0005
BACKENDS = {"kingston": FakeKingston, "marrakesh": FakeMarrakesh, "berlin": FakeBerlin}


# ---------------------------------------------------------------- oracles
def apply_dual_phases_with_gray_code(qc, flat_phases_1, flat_phases_2, img_wires, est_wire, power):
    """C^nP oracle: Gray-code traversal, X marking, one multi-controlled phase per pixel and image."""
    n = len(img_wires)
    gray_seq = [i ^ (i >> 1) for i in range(2**n)]
    for w in img_wires:                          # initial map: |0...0> -> |1...1>
        qc.x(w)
    target, controls = img_wires[-1], [est_wire] + list(img_wires[:-1])
    for step, pixel_idx in enumerate(gray_seq):
        if pixel_idx < len(flat_phases_1):
            for phi in (flat_phases_1[pixel_idx], flat_phases_2[pixel_idx]):
                if abs(phi) > 1e-5:
                    qc.mcp(phi * power, controls, target)
        if step < 2**n - 1:                      # transition: one X on the bit that changes
            bit = int(np.log2(pixel_idx ^ gray_seq[step + 1]))
            qc.x(img_wires[n - 1 - bit])
    for i, b in enumerate(format(gray_seq[-1], f'0{n}b')):   # uncompute: X on the zero bits of the last word
        if b == '0':
            qc.x(img_wires[i])


def apply_ucr_oracle(qc, flat_phases_1, flat_phases_2, img_wires, est_wires):
    """UCR oracle: C-U^(2^k) = UCRz_{e_k}(2^k phi) x diag(e^{i 2^k phi / 2}); the q diagonals are merged."""
    phi = np.zeros(2 ** len(img_wires))
    phi[:len(flat_phases_1)] = np.asarray(flat_phases_1) + np.asarray(flat_phases_2)  # U_A U_B^dagger
    ctrl = list(img_wires)[::-1]                 # Qiskit multiplexers read the controls LSB-first
    for k, e in enumerate(est_wires):
        qc.append(UCRZGate(list((2 ** k) * phi)), [e] + ctrl)
    qc.append(DiagonalGate(list(np.exp(0.5j * (2 ** len(est_wires) - 1) * phi))), ctrl)


def build_circuit(flat_p1, flat_p2, oracle, n_est=N_EST, n_pos=N_POS):
    total = n_est + n_pos
    qc = QuantumCircuit(total, total)
    est, pos = list(range(n_est)), list(range(n_est, total))
    qc.h(est + pos)
    if oracle == "cnp":
        for k, e in enumerate(est):
            apply_dual_phases_with_gray_code(qc, flat_p1, flat_p2, pos, e, 2 ** k)
    else:
        apply_ucr_oracle(qc, flat_p1, flat_p2, pos, est)
    qc.append(QFT(n_est, inverse=True, do_swaps=True).to_instruction(), est)
    qc.measure(range(total), range(total))
    return qc


def verify_equivalence(images):
    """Both oracles must give the same unitary / output distribution before any noisy run."""
    rng = np.random.default_rng(0)
    for ne, ni in [(2, 2), (3, 3)]:
        p1, p2 = rng.uniform(0, np.pi, 2**ni), -rng.uniform(0, np.pi, 2**ni)
        a, b = QuantumCircuit(ne + ni), QuantumCircuit(ne + ni)
        for k in range(ne):
            apply_dual_phases_with_gray_code(a, p1, p2, list(range(ne, ne + ni)), k, 2**k)
        apply_ucr_oracle(b, p1, p2, list(range(ne, ne + ni)), list(range(ne)))
        assert Operator(a).equiv(Operator(b))
    for img in images:
        p1, p2, _ = phases(img)
        probs = [Statevector(build_circuit(p1, p2, o).remove_final_measurements(inplace=False)).probabilities()
                 for o in ("cnp", "ucr")]
        assert np.abs(probs[0] - probs[1]).max() < 1e-9
    print("  both oracles implement the same unitary")


# ---------------------------------------------------------------- data and read-out
def load_images():
    X = fetch_openml('mnist_784', version=1, cache=True, parser='auto').data.values.reshape(-1, 28, 28)
    out = []
    for idx in range(N_IMAGES):
        im = resize(X[idx], (4, 4), anti_aliasing=True, preserve_range=True)
        out.append(im / np.max(im) * 255.0)
    return out


def phases(img):
    """Half-spectrum phases for horizontal QED; I(x,-1) = 0 at the boundary."""
    shifted = np.roll(img, shift=1, axis=1)
    shifted[:, 0] = 0
    return (img.flatten() / 255.0) * np.pi, (-shifted.flatten() / 255.0) * np.pi, np.abs(img - shifted)


def reconstruct_image_from_probs(probs_dict, original_shape, prob_threshold=P_TH, n_estimation_wires=N_EST):
    """Probability-weighted magnitude read-out |Delta I| = 2 I_max min(f, 1-f) (paper, Eq. 17)."""
    weighted, total = np.zeros(original_shape), np.zeros(original_shape)
    for bitstring, p in probs_dict.items():
        if p > prob_threshold:
            est_int = int(bitstring[:n_estimation_wires][::-1], 2)
            pixel_idx = int(bitstring[n_estimation_wires:], 2)
            if pixel_idx < np.prod(original_shape):
                r, c = np.unravel_index(pixel_idx, original_shape)
                raw = est_int / 2**n_estimation_wires * 255.0
                weighted[r, c] += p * 2.0 * min(raw, 255.0 - raw)
                total[r, c] += p
    return weighted / np.where(total == 0, 1.0, total)


def classical_fidelity(p_ideal, p_noise):
    """Bhattacharyya overlap between two output distributions."""
    keys = set(p_ideal) | set(p_noise)
    return sum(np.sqrt(p_ideal.get(k, 0.0) * p_noise.get(k, 0.0)) for k in keys)


def simulate(qc, backend, grad, shape):
    sim_ideal, sim_noise = AerSimulator(), AerSimulator.from_backend(backend)
    t_ideal = transpile(qc, backend=sim_ideal, optimization_level=3)
    t_noise = transpile(qc, backend=backend, optimization_level=3)
    p_ideal = {k[::-1]: v / SHOTS for k, v in sim_ideal.run(t_ideal, shots=SHOTS).result().get_counts().items()}
    p_noise = {k[::-1]: v / SHOTS for k, v in sim_noise.run(t_noise, shots=SHOTS).result().get_counts().items()}
    mae = lambda p: mean_absolute_error(grad.flatten(), reconstruct_image_from_probs(p, shape).flatten())
    return {"fidelity": classical_fidelity(p_ideal, p_noise), "mae_noise": mae(p_noise), "mae_ideal": mae(p_ideal),
            "depth_noise": t_noise.depth(), "depth_ideal": t_ideal.depth(),
            "prob_noise": p_noise, "prob_ideal": p_ideal}


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", choices=["cnp", "ucr"], default="ucr")
    args = ap.parse_args()
    suffix = "" if args.oracle == "cnp" else "_ucr"
    out_file, ckpt_file = f"qpipe_experiment_data_complete_4x4{suffix}.pkl", f"qpipe_checkpoint_4x4{suffix}.pkl"

    images = load_images()
    verify_equivalence(images)

    service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                   token=os.environ["QISKIT_IBM_TOKEN"],
                                   instance=os.environ["QISKIT_IBM_INSTANCE"])
    backends = {}
    for name, cls in BACKENDS.items():
        backends[name] = cls()
        backends[name].refresh(service)          # synchronize calibration with the live device
        print(f"  Fake{name.capitalize()} synchronized")

    results, start = {name: [] for name in backends}, 0
    if os.path.exists(ckpt_file):
        with open(ckpt_file, "rb") as f:
            results = pickle.load(f)["results"]
        start = len(results["kingston"])
        print(f"  resuming from image {start + 1}")

    for idx in range(start, N_IMAGES):
        t0 = time.time()
        p1, p2, grad = phases(images[idx])
        qc = build_circuit(p1, p2, args.oracle)
        for name, be in backends.items():
            results[name].append(simulate(qc, be, grad, images[idx].shape))
            r = results[name][-1]
            print(f"  image {idx + 1} Fake{name.capitalize()}: F={r['fidelity']:.3f} "
                  f"MAE={r['mae_noise']:.1f} depth={r['depth_noise']}")
        with open(ckpt_file, "wb") as f:
            pickle.dump({"results": results}, f)
        print(f"  image {idx + 1} done ({(time.time() - t0) / 60:.1f} min)")

    with open(out_file, "wb") as f:
        pickle.dump({"images_continuous": images,
                     "results_kingston": results["kingston"],
                     "results_marrakesh": results["marrakesh"],
                     "results_berlin": results["berlin"],
                     "metadata": {"oracle": args.oracle, "shots": SHOTS, "prob_threshold": P_TH,
                                  "optimization_level": 3, "image_resolution": "4x4"}}, f)
    print(f"Saved {out_file}")

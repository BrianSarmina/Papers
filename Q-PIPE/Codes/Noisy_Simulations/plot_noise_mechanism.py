# -*- coding: utf-8 -*-
"""
Q-PIPE -- FIG. 12 (mechanism of noise-induced degradation) for both oracle constructions.
Reads the saved results; no simulation is run.

Reads:  qpipe_experiment_data_complete_4x4.pkl      (C^nP oracle, original runs)
        qpipe_experiment_data_complete_4x4_ucr.pkl  (UCR oracle)
Writes: qpipe_noise_mechanism.pdf / .png
        + prints the per-instance numbers quoted in the caption / text

Top row   : classical gradient | ideal Q-PIPE | noisy C^nP | noisy UCR   (same backend, same image)
Bottom    : rank-sorted output probabilities (ideal, noisy C^nP, noisy UCR), uniform floor, P_th
LAYOUT    : "wide"   -> 1x4 heat maps, for a figure* environment (recommended)
            "column" -> 2x2 heat maps, for a single-column figure environment
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

CNP_FILE = "qpipe_experiment_data_complete_4x4.pkl"
UCR_FILE = "qpipe_experiment_data_complete_4x4_ucr.pkl"
BACKEND_KEY, BACKEND_NAME = "results_kingston", "FakeKingston"
IMAGE_IDX = 0                     # instance used in the original FIG. 12
P_TH, N_EST, N_QUBITS = 5e-4, 8, 12
LAYOUT = "wide"
OUT = "qpipe_noise_mechanism"


def reconstruct_image_from_probs(probs_dict, original_shape, prob_threshold=0.0005, n_estimation_wires=8):
    """Same read-out as qpipe_noisy_simulation.py (paper, Eq. 17)."""
    weighted_mag_sum = np.zeros(original_shape, dtype=float)
    prob_sum = np.zeros(original_shape, dtype=float)
    n_pixels = np.prod(original_shape)
    for bitstring, p in probs_dict.items():
        if p > prob_threshold:
            est_bin = bitstring[:n_estimation_wires]
            pos_bin = bitstring[n_estimation_wires:]
            est_int = int(est_bin[::-1], 2)
            pixel_idx = int(pos_bin, 2)
            if pixel_idx < n_pixels:
                row, col = np.unravel_index(pixel_idx, original_shape)
                raw_val = (est_int / (2**n_estimation_wires)) * 255.0
                edge_magnitude = 2.0 * min(raw_val, 255.0 - raw_val)
                weighted_mag_sum[row, col] += p * edge_magnitude
                prob_sum[row, col] += p
    safe_prob_sum = np.where(prob_sum == 0, 1.0, prob_sum)
    return weighted_mag_sum / safe_prob_sum


cnp = pickle.load(open(CNP_FILE, "rb"))
ucr = pickle.load(open(UCR_FILE, "rb"))
img = cnp["images_continuous"][IMAGE_IDX]
assert np.allclose(img, ucr["images_continuous"][IMAGE_IDX]), "the two files use different images"
shifted = np.roll(img, 1, axis=1); shifted[:, 0] = 0
g_classic = np.abs(img - shifted)

r_cnp = cnp[BACKEND_KEY][IMAGE_IDX]
r_ucr = ucr[BACKEND_KEY][IMAGE_IDX]
p_ideal = r_cnp["prob_ideal"]            # same ideal distribution for both oracles (same unitary)

maps = [(r"Classical $G_{\mathrm{classic}}$", g_classic),
        ("Ideal Q-PIPE", reconstruct_image_from_probs(p_ideal, img.shape, P_TH, N_EST)),
        (f"Noisy, $C^{{n}}P$ ({BACKEND_NAME})", reconstruct_image_from_probs(r_cnp["prob_noise"], img.shape, P_TH, N_EST)),
        (f"Noisy, UCR ({BACKEND_NAME})", reconstruct_image_from_probs(r_ucr["prob_noise"], img.shape, P_TH, N_EST))]

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 11})
if LAYOUT == "wide":
    fig = plt.figure(figsize=(12, 6.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.15], width_ratios=[1, 1, 1, 1, 0.06],
                          hspace=0.35, wspace=0.12)
    top = [fig.add_subplot(gs[0, i]) for i in range(4)]
    cax = fig.add_subplot(gs[0, 4])
    bottom = fig.add_subplot(gs[1, :4])
else:
    fig = plt.figure(figsize=(7, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2], width_ratios=[1, 1, 0.06], hspace=0.35, wspace=0.12)
    top = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    cax = fig.add_subplot(gs[0:2, 2])
    bottom = fig.add_subplot(gs[2, :2])

for ax, (title, m) in zip(top, maps):
    im = ax.imshow(m, cmap="magma", vmin=0, vmax=np.max(g_classic))
    for (r, c), v in np.ndenumerate(m):
        ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=9,
                color="white" if v < 0.6 * np.max(g_classic) else "black")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, cax=cax, label="Edge magnitude (0-255)")

curves = [("Ideal simulation", p_ideal, "#1f4e79", "-"),
          (f"Noisy, $C^{{n}}P$ ({BACKEND_NAME})", r_cnp["prob_noise"], "#c0392b", "-"),
          (f"Noisy, UCR ({BACKEND_NAME})", r_ucr["prob_noise"], "#2ca02c", "-")]
for label, p, col, ls in curves:
    v = np.sort(np.fromiter(p.values(), float))[::-1]
    bottom.step(np.arange(1, len(v) + 1), v, where="post", color=col, ls=ls, lw=2, label=label)
bottom.axhline(2.0 ** -N_QUBITS, color="grey", ls=":", lw=2, label=rf"Uniform floor $2^{{-{N_QUBITS}}}$")
bottom.axhline(P_TH, color="black", ls="--", lw=1.5, label=r"Threshold $P_{\mathrm{th}}=5\times10^{-4}$")
bottom.set_yscale("log"); bottom.set_ylim(1e-4, 3e-1); bottom.set_xlim(1, 2 ** N_QUBITS + 50)
bottom.set_xlabel("Rank of output bitstring (sorted, descending)")
bottom.set_ylabel("Probability")
bottom.set_title("Output distribution under NISQ noise")
bottom.grid(True, which="both", ls="--", alpha=0.35)
bottom.legend(ncol=2, fontsize=10, loc="upper right")

fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}.png", dpi=600, bbox_inches="tight")

# ------------------------------------------------------------------ numbers for the caption
mass = lambda p: sum(v for v in p.values() if v > P_TH)
print(f"Instance {IMAGE_IDX}, {BACKEND_NAME}")
for (title, m), p in zip(maps[1:], [p_ideal, r_cnp["prob_noise"], r_ucr["prob_noise"]]):
    mae = np.abs(m - g_classic).mean()
    r = np.corrcoef(m.flatten(), g_classic.flatten())[0, 1]
    print(f"  {title:32s} MAE {mae:6.2f} | Pearson r {r:5.2f} | mean {m.mean():6.1f} std {m.std():5.1f} | "
          f"mass>P_th {mass(p):.3f} | distinct bitstrings {len(p)}")

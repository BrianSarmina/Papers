# -*- coding: utf-8 -*-
"""
Q-PIPE -- FIG. 13 (noisy metrics) and Table I rows for both oracle constructions (paper, Appendix B).

Reads:  qpipe_experiment_data_complete_4x4.pkl       (original C^nP runs, unchanged)
        qpipe_experiment_data_complete_4x4_ucr.pkl   (both from qpipe_noisy_simulation.py)
Writes: qpipe_comprehensive_metrics_compare.{png,eps}  and prints LaTeX rows for Table I.

Solid lines = C^nP oracle (FIG. 1); dashed lines = UCR oracle. Colours = backend.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

FILES = {"C$^n$P": "qpipe_experiment_data_complete_4x4.pkl",
         "UCR": "qpipe_experiment_data_complete_4x4_ucr.pkl"}
BACKENDS = [("results_kingston", "FakeKingston", "blue", "o"),
            ("results_marrakesh", "FakeMarrakesh", "orange", "s"),
            ("results_berlin", "FakeBerlin", "green", "^")]
LS = {"C$^n$P": "-", "UCR": "--"}
PTH = 0.0005
OUT = "qpipe_comprehensive_metrics"

data = {k: pickle.load(open(f, "rb")) for k, f in FILES.items()}
n = len(data["UCR"]["results_kingston"]); x = np.arange(1, n + 1)


def extra_metrics(r):
    p = np.array(list(r["prob_noise"].values()))
    keys = set(r["prob_ideal"]) | set(r["prob_noise"])
    return dict(H=float(-(p * np.log2(p)).sum()), mass=float(p[p > PTH].sum()),
                TVD=0.5 * sum(abs(r["prob_ideal"].get(k, 0) - r["prob_noise"].get(k, 0)) for k in keys))


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.8))
bar_w = 0.38
for c_idx, (cons, d) in enumerate(data.items()):
    for b_idx, (key, name, col, mk) in enumerate(BACKENDS):
        R = d[key][:n]
        f = [r["fidelity"] for r in R]; m = [r["mae_noise"] for r in R]
        ax1.plot(x, f, ls=LS[cons], marker=mk, color=col, label=f"{name}, {cons} (mean {np.mean(f):.3f})")
        ax2.plot(x, m, ls=LS[cons], marker=mk, color=col, label=f"{name}, {cons} (mean {np.mean(m):.1f})")
        ax3.bar(b_idx + (c_idx - 0.5) * bar_w, np.mean([r["depth_noise"] for r in R]), bar_w,
                color=col, alpha=0.9 if cons == "UCR" else 0.45, hatch=None if cons == "UCR" else "//",
                edgecolor="black")
mae_ideal = np.mean([[r["mae_ideal"] for r in data[c][k][:n]] for c in data for k, *_ in BACKENDS], axis=0)
ax2.plot(x, mae_ideal, "k--x", lw=2, label=f"Ideal simulation (mean {np.mean(mae_ideal):.3f})")

ax1.set_title("Classical Fidelity between Ideal and Noisy Output Distributions", fontsize=13, pad=12)
ax1.set_ylabel(r"Classical Fidelity $F(P_{\mathrm{ideal}}, P_{\mathrm{noise}})$", fontsize=11)
ax2.set_title("Reconstruction Error (MAE)", fontsize=13, pad=12)
ax2.set_ylabel("Mean Absolute Error (MAE)", fontsize=11); ax2.set_yscale("log")
for a in (ax1, ax2):
    a.set_xlabel("Image Sample Index (MNIST 4x4)", fontsize=11); a.set_xticks(x)
    a.grid(True, ls="--", alpha=0.5); a.legend(fontsize=8.5, ncol=1)
ax3.set_title("Transpiled Circuit Depth (optimization level 3)", fontsize=13, pad=12)
ax3.set_yscale("log"); ax3.set_xticks(range(3)); ax3.set_xticklabels([b[1] for b in BACKENDS])
from matplotlib.patches import Patch
ax3.legend(handles=[Patch(facecolor="grey", alpha=0.45, hatch="//", edgecolor="black", label="C$^n$P oracle"),
                    Patch(facecolor="grey", alpha=0.9, edgecolor="black", label="UCR oracle")], fontsize=10)
ax3.set_ylim(1e2, None); ax3.set_ylabel("Depth"); ax3.grid(True, axis="y", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUT}.png", dpi=600, bbox_inches="tight"); plt.savefig(f"{OUT}.eps", bbox_inches="tight")

# ---------------- Table I rows ----------------
print("% Backend & Construction & F & MAE & Depth & H (bits) & Mass>P_th & TVD \\\\")
for cons, d in data.items():
    for key, name, *_ in BACKENDS:
        R = d[key][:n]; E = [extra_metrics(r) for r in R]
        f = [r["fidelity"] for r in R]; m = [r["mae_noise"] for r in R]
        print(f"{name} & {cons} & ${np.mean(f):.3f}\\pm{np.std(f):.3f}$ & ${np.mean(m):.1f}\\pm{np.std(m):.1f}$ & "
              f"${np.mean([r['depth_noise'] for r in R]):.0f}$ & ${np.mean([e['H'] for e in E]):.1f}$ & "
              f"${np.mean([e['mass'] for e in E]):.3f}$ & ${np.mean([e['TVD'] for e in E]):.2f}$ \\\\")

# elementary-gate metrics (no coupling map) and hardware 2q counts, both constructions
em = data["UCR"].get("elementary_metrics")
if em:
    for cons in ("CnP", "UCR"):
        cx = np.mean([e[cons]["kingston"]["cx"] for e in em]); dep = np.mean([e[cons]["kingston"]["depth"] for e in em])
        tq = {b: np.mean([e[cons][b]["twoq_hw"] for e in em]) for b in ("kingston", "marrakesh", "berlin")}
        print(f"% {cons}: elementary CX = {cx:.0f}, elementary depth = {dep:.0f}, hardware 2q gates = "
              + ", ".join(f"{b} {v:.0f}" for b, v in tq.items()))

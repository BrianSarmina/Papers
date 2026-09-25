"""
Q-PIPE -- FIG. 8: resource scaling of FRQI, NEQR and Q-PIPE with two syntheses in one gate set.

Convention (Referee 2, points 1-2): elementary gates {CNOT} + single-qubit, NO ancillas, same q-bit
precision, worst case (all pixels / all intensity bits nonzero), single-image encoding.

  colour  = representation (FRQI red, NEQR purple, Q-PIPE green)
  solid   = multi-controlled construction  (one C^n gate per pixel [per bit])
  dashed  = uniformly controlled rotations (UCR; Mottonen et al. 2004, Amankwah et al. 2022)

Multi-controlled costs are the CNOT count / CNOT depth of ONE gate with n controls, measured with
Qiskit 2.5.2 (transpile, basis {cx, rz, sx, x}, optimization_level=1):
  C^n R_y  (FRQI, Le et al.)     -> linear in n   (multi-controlled SU(2): linear synthesis exists)
  C^n X    (NEQR, Zhang et al.)  -> ~quadratic in n (ancilla-free)
  C^n P    (Q-PIPE, FIG. 1)      -> ~quadratic in n (multi-controlled U(2): no ancilla-free linear
                                    synthesis known; linear DEPTH only, da Silva & Park 2022)
UCR costs are exact closed forms, validated against full transpilation for n = 2..8:
  FRQI N,  NEQR qN,  Q-PIPE (q+1)N - 2 + q(q-1) + 3*floor(q/2)   (the last two terms = QFT^dagger)
Set RECOMPUTE = True to re-measure the multi-controlled costs with your Qiskit version.
Note: the QED circuit of Appendix B applies TWO C^nP per pixel (phi1, phi2); the UCR form merges them.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

Q = 8
SIDES = [2, 4, 8, 16, 32, 64, 128, 256]
RECOMPUTE = False
# n -> (CNOT count, CNOT depth), Qiskit 2.5.2
MC = {
 'mcry': {1: (2, 2), 2: (12, 12), 3: (20, 20), 4: (24, 18), 5: (40, 34), 6: (56, 38), 7: (80, 80), 8: (104, 89),
          9: (120, 108), 10: (136, 121), 11: (152, 140), 12: (168, 153), 13: (184, 172), 14: (200, 185),
          15: (216, 204), 16: (232, 217)},
 'mcx':  {1: (1, 1), 2: (6, 6), 3: (14, 14), 4: (36, 35), 5: (84, 70), 6: (124, 104), 7: (180, 134), 8: (252, 199),
          9: (332, 277), 10: (452, 360), 11: (564, 466), 12: (716, 598), 13: (852, 728), 14: (1036, 888),
          15: (1188, 1038), 16: (1404, 1230)},
 'mcp':  {1: (2, 2), 2: (6, 6), 3: (20, 18), 4: (44, 36), 5: (84, 70), 6: (140, 101), 7: (220, 174), 8: (324, 262),
          9: (444, 368), 10: (580, 488), 11: (732, 626), 12: (900, 778), 13: (1084, 948), 14: (1284, 1132),
          15: (1500, 1334), 16: (1732, 1550)}}

if RECOMPUTE:
    from qiskit import QuantumCircuit, transpile
    for key in MC:
        for m in MC[key]:
            qc = QuantumCircuit(m + 1)
            {'mcry': lambda: qc.mcry(0.37, list(range(m)), m), 'mcx': lambda: qc.mcx(list(range(m)), m),
             'mcp': lambda: qc.mcp(0.37, list(range(m)), m)}[key]()
            t = transpile(qc, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)
            MC[key][m] = (t.count_ops().get('cx', 0), t.depth(lambda i: i.operation.num_qubits == 2))


def qft_cx(q): return q * (q - 1) + 3 * (q // 2)


def costs(side, q=Q):
    n = int(np.ceil(np.log2(side * side))); N = 2 ** n
    # C^nP in Q-PIPE has n controls in total (e_k + n-1 position qubits) acting on n+1 qubits
    return {
        'qubits': {'FRQI': n + 1, 'NEQR': n + q, 'QPIPE': n + q},
        'cx': {'FRQI_MC': N * MC['mcry'][n][0], 'FRQI_UCR': N,
               'NEQR_MC': q * N * MC['mcx'][n][0], 'NEQR_UCR': q * N,
               'QPIPE_MC': q * N * MC['mcp'][n][0] + qft_cx(q), 'QPIPE_UCR': (q + 1) * N - 2 + qft_cx(q)},
        'depth': {'FRQI_MC': N * MC['mcry'][n][1], 'FRQI_UCR': N,
                  'NEQR_MC': q * N * MC['mcx'][n][1], 'NEQR_UCR': q * N,
                  'QPIPE_MC': q * N * MC['mcp'][n][1] + 2 * q, 'QPIPE_UCR': (q + 1) * N - 2 + 2 * q}}


COL = {'FRQI': '#d62728', 'NEQR': '#9467bd', 'QPIPE': '#2ca02c'}
MK = {'FRQI': 'D', 'NEQR': 'x', 'QPIPE': '^'}
LAB = {'FRQI': 'FRQI', 'NEQR': 'NEQR', 'QPIPE': 'Q-PIPE'}


def main():
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    R = [costs(s) for s in SIDES]; os.makedirs('QED', exist_ok=True)
    panels = {'qubits_both': ('qubits', 'Number of Qubits Required'),
              'gates_both': ('cx', 'CNOT Count'), 'depth_both': ('depth', 'CNOT Depth')}
    for fn, (field, ylab) in panels.items():
        fig, a = plt.subplots(figsize=(8, 6))
        for rep in ('FRQI', 'NEQR', 'QPIPE'):
            if field == 'qubits':
                a.plot(SIDES, [r['qubits'][rep] for r in R], color=COL[rep], marker=MK[rep], lw=2.2, ms=8,
                       ls='--' if rep == 'QPIPE' else '-', label=LAB[rep])
                continue
            a.plot(SIDES, [r[field][rep + '_MC'] for r in R], color=COL[rep], marker=MK[rep], lw=2.2, ms=8,
                   ls='-', label=f'{LAB[rep]} (multi-controlled)')
            a.plot(SIDES, [r[field][rep + '_UCR'] for r in R], color=COL[rep], marker=MK[rep], lw=2.2, ms=8,
                   ls='--', mfc='white', label=f'{LAB[rep]} (UCR)')
        if field != 'qubits':
            a.set_yscale('log')
        a.set_xscale('log', base=2); a.set_xticks(SIDES); a.set_xticklabels(SIDES)
        a.set_xlabel('Image Size (pixels per side)', fontweight='bold'); a.set_ylabel(ylab, fontweight='bold')
        a.grid(True, which='both', ls='--', alpha=0.35); a.legend(fontsize=10.5, loc='upper left')
        fig.tight_layout(); fig.savefig(f'QED/{fn}.png', dpi=300); plt.close(fig)
    print(f"{'side':>5} " + " ".join(f"{k:>11}" for k in R[0]['cx']))
    for s, r in zip(SIDES, R):
        print(f"{s:>5} " + " ".join(f"{v:>11}" for v in r['cx'].values()))


if __name__ == '__main__':
    main()

import pickle
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# --- Ranges ---
gamma_range = np.linspace(0.1, 2*np.pi, 10)
beta_range  = np.linspace(0.1, 2*np.pi, 10)

def build_qaoa_qnode(n, connections, fields, depth=3, dev_name="default.qubit"):
    """Return a QNode that prepares an RX–RY mixer QAOA state with
    distinct β_x and β_y per depth stage."""
    dev = qml.device(dev_name, wires=n, shots=None)

    @qml.qnode(dev)
    def circuit(params):
        # params layout: [gammas (d), betas_x (d), betas_y (d)]  -> total 3d
        d = depth
        assert len(params) == 3*d, f"Expected {3*d} parameters (γ^d, βx^d, βy^d), got {len(params)}"
        gammas  = params[0:d]
        betas_x = params[d:2*d]
        betas_y = params[2*d:3*d]

        # 0) Walsh–Hadamard
        for w in range(n):
            qml.Hadamard(wires=w)

        # 1–2) Alternating problem / mixer layers
        for layer in range(d):
            g  = gammas[layer]
            bx = betas_x[layer]
            by = betas_y[layer]

            # Problem Hamiltonian (ZZ edges + local Z fields)
            for (u, v) in connections:
                qml.IsingZZ(2*g, wires=(u, v))
            for j, h_j in enumerate(fields):
                qml.RZ(2*g*h_j, wires=j)

            # Mixer: RX then (optionally entangle) then RY, with distinct betas
            for q in range(n):
                qml.RX(bx, wires=q)

            # Anillo de CNOTs entrelazar entre RX y RY
            for c in range(n):
                qml.CNOT(wires=[c, (c+1) % n])

            for q in range(n):
                qml.RY(by, wires=q)

        return qml.state()

    return circuit

# --------- QFI sweep (sin cambios en la lógica de barrido) ----------
pick_read = open('Random_Graphs_QFI.pkl','rb')
data = pickle.load(pick_read)

all_qfi_tensors = []
all_qfi_means   = []

for G, n in data:
    fields = [attr["magnet_field"] for _, attr in G.nodes(data=True)]
    connections = list(G.edges())
    print(G)
    depth = 3
    qnode = build_qaoa_qnode(n, connections, fields, depth=depth)

    qfi_fn = qml.gradients.quantum_fisher(qnode)  # devuelve función que evalúa el QFIM
    qfi_per_grid = []

    # grid 2D en (γ0, γ1); usamos el mismo patrón (i,j) para βx y βy por capa
    for i, g0 in enumerate(gamma_range):
        for j, g1 in enumerate(gamma_range):
            g2 = g0
            # βx por capa
            bx0 = beta_range[j]
            bx1 = beta_range[i]
            bx2 = beta_range[j]
            # βy por capa
            by0 = beta_range[i]
            by1 = beta_range[j]
            by2 = beta_range[i]

            # Orden: [γ0, γ1, βx0, βx1, βy0, βy1]
            params = np.array([g0, g1, g2, bx0, bx1, bx2, by0, by1, by2], requires_grad=True)
            qfi_tensor = qfi_fn(params)  # matriz (3*depth) x (3*depth) -> aquí 6x6
            qfi_per_grid.append(qfi_tensor)
##            break
##        break
##    break

    all_qfi_tensors.append(qfi_per_grid)
    all_qfi_means.append(np.mean(qfi_per_grid, axis=0))

##fig, ax = qml.draw_mpl(qnode)(params)
##plt.show()

with open('QFI_experiment_matrices_rxry_L3_ent.pkl', 'wb') as f:
    pickle.dump(all_qfi_tensors, f)

with open('QFI_experiment_means_rxry_L3_ent.pkl', 'wb') as f:
    pickle.dump(all_qfi_means, f)

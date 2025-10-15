import scipy
import scipy.constants      # <— cargamos el submódulo de constantes físicas
import pennylane as qml
import pennylane as qml
from pennylane import numpy as np
import pickle
import matplotlib.pyplot as plt

"""### Generador de grafos."""
pick_read = open('Random_Graphs_QFI.pkl','rb')
data = pickle.load(pick_read)
##print(data)

"""### QFI QAOA"""
########################### DATA PREP ############################
gamma_range = np.linspace(0.1, 2*np.pi, 10)
beta_range  = np.linspace(0.1, 2*np.pi, 10)

def build_qaoa_qnode(n, connections, fields, depth=3, dev_name="default.qubit"):
    """Return a QNode that prepares one full Walsh-QAOA state of given depth."""
    dev = qml.device(dev_name, wires=n)

    @qml.qnode(dev)
    def circuit(params):
        # unpack   params = [γ₀, γ₁, …, β₀, β₁, …]
        gammas = params[:depth]
        betas  = params[depth:]

        # 0) Walsh-Hadamard
        for w in range(n):
            qml.Hadamard(wires=w)

        # 1-2) Alternating problem/mixer layers
        for d in range(depth):
            γ, β = gammas[d], betas[d]

            # Problem Hamiltonian
            for (u, v) in connections:
                qml.IsingZZ(2*γ, wires=(u, v))
            for j, h_j in enumerate(fields):
                qml.RZ(2*γ*h_j, wires=j)

            # Mixer (RX only here)
            for c in range(n):
                if c <= 0:
                    qml.CNOT(wires=[c, n-1])
                else:
                    qml.CNOT(wires=[c-1, c])
            for j in range(n):
                qml.RX(β, wires=j)

        return qml.state()      # state output is fine for QFIM

    return circuit

########################### QFI LOOP #############################
all_qfi_tensors      = []
all_qfi_means        = []

for G, n in data:                               # -> your `data` iterator
  fields = [attr["magnet_field"] for _, attr in G.nodes(data=True)]
  connections = list(G.edges())
  qnode = build_qaoa_qnode(n, connections, fields, depth=3)
  print(G)
  qfi_fn = qml.gradients.quantum_fisher(qnode)  # transformed function
  qfi_per_grid = []
  # double loop (γ₀ = γ_i, γ₁ = γ_j, β₀ = β_i, β₁ = β_j)
  for i, γ0 in enumerate(gamma_range):
    for j, γ1 in enumerate(gamma_range):
      γ2 = γ0
      β0 = beta_range[i]
      β1 = beta_range[j]
      β2 = beta_range[i]
    # params = np.array([γ0, β0], requires_grad=True)
      params = np.array([γ0, γ1, γ2, β0, β1, β2], requires_grad=True)
      qfi_tensor = qfi_fn(params)
      qfi_per_grid.append(qfi_tensor)
##      break
##  break
  all_qfi_tensors.append(qfi_per_grid)
  all_qfi_means.append(np.mean(qfi_per_grid, axis=0))

##fig, ax = qml.draw_mpl(qnode)(params)
##plt.show()

out_1 = open('QFI_experiment_matrices_rx_L3_ent.pkl', 'wb')
pickle.dump(all_qfi_tensors, out_1)
out_1.close()

out_2 = open('QFI_experiment_means_rx_L3_ent.pkl', 'wb')
pickle.dump(all_qfi_means, out_2)
out_2.close()


from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, Operator
from qiskit_algorithms.gradients import QFI, LinCombQGT
from qiskit_aer.primitives import Estimator
import numpy as np

#- COMBINATIONS PRO -#
def numpy_combinations(x):
    idx = np.stack(np.triu_indices(len(x), k=1), axis=-1)
    return x[idx]
#- COMBINATIONS PRO -#

def createMaxCut_qiskit(qc, q_num, gammas, betas1, betas2, combs):
  # Walsh-Hadamard transform.
  for i in range(q_num):
    qc.h(i)
  qc.barrier()
  # Phase operator (interactions).
  for comb in combs:
    qc.rzz(gammas[0], comb[0], comb[1])
  qc.barrier()
  for i in range(q_num):
    qc.rx(betas1[0], i)
  qc.barrier()
  ### ENTANGLEMENT STAGE ###
  for comb in combs:
    qc.cx(comb[0], comb[1])
  qc.barrier()
  ##########################
  for i in range(q_num):
    qc.ry(betas2[0], i)
  qc.barrier()

  ########## SECOND DEPTH #########
  # Phase operator (interactions).
  for comb in combs:
    qc.rzz(gammas[1], comb[0], comb[1])
  qc.barrier()
  for i in range(q_num):
    qc.rx(betas1[1], i)
  qc.barrier()
  ### ENTANGLEMENT STAGE ###
  for comb in combs:
    qc.cx(comb[0], comb[1])
  qc.barrier()
  ##########################
  for i in range(q_num):
    qc.ry(betas2[1], i)
  qc.barrier()
  ################################

##  ########## THIRD DEPTH #########
##  # Phase operator (interactions).
##  for comb in combs:
##    qc.rzz(gammas[2], comb[0], comb[1])
##  qc.barrier()
##  for i in range(q_num):
##    qc.rx(betas1[2], i)
##  qc.barrier()
##  ### ENTANGLEMENT STAGE ###
##  for comb in combs:
##    qc.cx(comb[0], comb[1])
##  qc.barrier()
##  ##########################
##  for i in range(q_num):
##    qc.ry(betas2[2], i)
##  qc.barrier()
##  ################################

##  print(qc.draw())

  return qc


########## MAIN #########
data = []
number_experiments = 100
number_of_parameters = 6

### RANDOM PARAMETERS DATA GENERATION ###
data = [[np.random.uniform(0, 2*np.pi) for x in range(number_of_parameters)] for y in range(number_experiments)]

### QUANTUM CIRCUIT PARAMETERS ###
q_num = 7
depth_level = 2
combs = numpy_combinations(np.arange(q_num))

exp_counter = 0
qfi_matrices = []
for sol in data:
    # Gamma parameters to be optimized.
    #gammas = np.random.rand(q_num)
    gammas = ParameterVector('γ', depth_level)
    # Beta parameters to be optimized.
    #betas = np.random.rand(q_num)
    betas1 = ParameterVector('β_1s', depth_level)
    betas2 = ParameterVector('β_2s', depth_level)
    parameters = [gammas, betas1, betas2]
    # Parameter-shift value
    s = np.pi/2

    qc = QuantumCircuit(q_num)
    qc = createMaxCut_qiskit(qc, q_num, gammas, betas1, betas2, combs)
##    break
    # Initialize QFI.
    estimator = Estimator()
    qgt = LinCombQGT(estimator)
    qfi = QFI(qgt)

    # Evaluate
    values_list = [sol]
    qfi = qfi.run(qc, values_list).result().qfis
    qfi_matrices.append(qfi)

    print(exp_counter)
    exp_counter += 1

mean_qfi_matrices = np.mean(qfi_matrices, axis=0)
print("6p entangled")
print(mean_qfi_matrices)

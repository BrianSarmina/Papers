import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

qfi_7n_com_2p_CYC_ENT = np.array([[ 9.82390453, -4.17927437],
                                  [-4.17927437, 20.92394314]])

qfi_7n_com_2p_ent = np.array([[ 9.1401046,  -4.6185107 ],
                              [-4.6185107,  20.82459084]])

qfi_7n_com_4p_CYC_ENT = np.array([[10.10125286,  3.56584473, -3.95899761, -1.41238514],
                                  [ 3.56584473, 10.3628685,  -2.11390312, -2.20321114],
                                  [-3.95899761, -2.11390312, 20.80796093,  2.20366745],
                                  [-1.41238514, -2.20321114,  2.20366745, 24.93624821]])

qfi_7n_com_4p_ent = np.array([[ 8.85119884,  3.32657112, -4.2918885,   0.49245636],
                              [ 3.32657112, 10.27173443, -1.75486687, -2.26674755],
                              [-4.2918885,  -1.75486687, 20.72673428,  3.61842182],
                              [ 0.49245636, -2.26674755,  3.61842182, 22.44842175]])

qfi_7n_com_6p_CYC_ENT = np.array([[ 9.70586052,  3.70919731,  1.80605698, -3.44668365, -1.27165825, 0.17116482],
                                  [ 3.70919731, 10.38989265,  2.66184227, -1.88667355, -2.08819698, -1.0669643 ],
                                  [ 1.80605698,  2.66184227,  9.23348072, -0.86101788, -0.82938072, -0.94918324],
                                  [-3.44668365, -1.88667355, -0.86101788, 20.89206165,  2.39165543, 0.92069668],
                                  [-1.27165825, -2.08819698, -0.82938072,  2.39165543, 24.83804523, 4.10059536],
                                  [ 0.17116482, -1.0669643,  -0.94918324,  0.92069668,  4.10059536, 24.48951363]])

qfi_7n_com_6p_ent = np.array([[ 9.42905373,  3.83360325,  1.68538971, -3.96868122,  0.70774448, 0.69032074],
                              [ 3.83360325, 10.61471214,  2.8587627,  -2.16253597, -2.65613155, 0.86404675],
                              [ 1.68538971,  2.8587627,   9.19088017, -0.74257271, -0.89784435, -1.52905956],
                              [-3.96868122, -2.16253597, -0.74257271, 21.03408764,  3.84862259, 1.515835],
                              [ 0.70774448, -2.65613155, -0.89784435,  3.84862259, 23.36684536, 3.2248798],
                              [ 0.69032074,  0.86404675, -1.52905956,  1.515835,    3.2248798, 23.22327835]])


# Create a list of matrices and their titles
qfi_matrices = [qfi_7n_com_2p_CYC_ENT, qfi_7n_com_2p_ent, qfi_7n_com_4p_CYC_ENT, qfi_7n_com_4p_ent, qfi_7n_com_6p_CYC_ENT, qfi_7n_com_6p_ent]
titles = ['QFI 2p CYC-ENT', 'QFI 2p COM-ENT', 'QFI 4p CYC-ENT', 'QFI 4p COM-ENT', 'QFI 6p CYC-ENT', 'QFI 6p COM-ENT']

# Set up the figure and axes
fig, axes = plt.subplots(3, 2, figsize=(13, 10))

# Use a single color bar for all subplots
vmin = min([matrix.min() for matrix in qfi_matrices])
vmax = max([matrix.max() for matrix in qfi_matrices])

# Plot each matrix as a heatmap
for i, ax in enumerate(axes.flat):
    sns.heatmap(qfi_matrices[i], ax=ax, annot=True, cmap='crest', linewidth=0.5, vmin=vmin, vmax=vmax, cbar=i == 0,
                cbar_ax=None if i else fig.add_axes([.91, .3, .03, .4]))
    ax.set_title(titles[i])

# Adjust layout
plt.tight_layout(rect=[0, 0, .9, 1])
plt.show()

##plt.imshow(qfi_4n_cyc_6p_ent)
##data = np.array(qfi_4n_cyc_6p_ent, dtype='float16')
##for i in range(len(qfi_4n_cyc_6p_ent)):
##  for j in range(len(qfi_4n_cyc_6p_ent)):
##      plt.annotate(str(data[i][j]), xy=(j, i), ha='center', va='center', color='white')
##plt.colorbar()

#plt.show()

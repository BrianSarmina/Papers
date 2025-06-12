import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

qfi_7n_cyc_2p_CYC_ENT = np.array([[ 6.56099125, -0.01971329],
                                  [-0.01971329,  6.98501804]])

qfi_7n_cyc_2p_ent = np.array([[ 7.69366997e+00, -5.15560150e-03],
                              [-5.15560150e-03,  6.99685547e+00]])

qfi_7n_cyc_4p_CYC_ENT = np.array([[ 7.03982449,  2.75490547,  0.02533192,  0.02179482],
                                  [ 2.75490547, 10.04136326, -0.0263187,  -0.08681629],
                                  [ 0.02533192, -0.0263187,   7.01038818,  0.05780972],
                                  [ 0.02179482, -0.08681629,  0.05780972,  8.04576378]])

qfi_7n_cyc_4p_ent = np.array([[ 7.19781429,  1.74724674, -0.04522877, -0.5709206 ],
                              [ 1.74724674, 10.9233149,  -0.49628456, -1.00851425],
                              [-0.04522877, -0.49628456,  6.97970932, -0.01594242],
                              [-0.5709206,  -1.00851425, -0.01594242,  7.73913857]])

qfi_7n_cyc_6p_CYC_ENT = np.array([[ 6.28660824e+00,  1.88295944e+00,  6.31825905e-01, -2.64120483e-02, -1.81865692e-03,  5.18771744e-02],
                                  [ 1.88295944e+00,  8.96035725e+00,  1.55012199e+00,  3.57212830e-02, 4.86550522e-02, -8.55564499e-02],
                                  [ 6.31825905e-01,  1.55012199e+00,  8.06044918e+00,  1.89002533e-01, 1.49687958e-02, -5.70648193e-03],
                                  [-2.64120483e-02,  3.57212830e-02,  1.89002533e-01,  7.01115711e+00, 6.59156418e-02, -1.64589691e-02],
                                  [-1.81865692e-03,  4.86550522e-02,  1.49687958e-02,  6.59156418e-02, 8.16518875e+00,  2.56018410e-01],
                                  [ 5.18771744e-02, -8.55564499e-02, -5.70648193e-03, -1.64589691e-02, 2.56018410e-01,  7.92851234e+00]])

qfi_7n_cyc_6p_ent = np.array([[ 7.87297623,  1.97674942,  1.14930557, -0.05451984, -0.7303738, 1.07967537],
                              [ 1.97674942, 10.97746727,  2.16864006, -0.43094063, -1.00809124, -0.03245564],
                              [ 1.14930557,  2.16864006,  9.10862289,  0.11753872, -0.56978577, -0.62149483],
                              [-0.05451984, -0.43094063,  0.11753872,  6.96096859, -0.15305588, -0.14653309],
                              [-0.7303738,  -1.00809124, -0.56978577, -0.15305588,  7.69967819, 0.88623272],
                              [ 1.07967537, -0.03245564, -0.62149483, -0.14653309,  0.88623272, 7.5150246 ]])


# Create a list of matrices and their titles
qfi_matrices = [qfi_7n_cyc_2p_CYC_ENT, qfi_7n_cyc_2p_ent, qfi_7n_cyc_4p_CYC_ENT, qfi_7n_cyc_4p_ent, qfi_7n_cyc_6p_CYC_ENT, qfi_7n_cyc_6p_ent]
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

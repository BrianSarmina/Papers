import os
import pickle
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces, fetch_openml
from sklearn.metrics import mean_absolute_error
from skimage.transform import resize
import scipy.ndimage as ndimage
import warnings
warnings.filterwarnings('ignore')

### Q-PIPE (circuito cuantico ###
def apply_dual_phases_with_gray_code(flat_phases_1, flat_phases_2, img_wires, est_wire, power):
    n = len(img_wires)
    total_states = 2**n
    gray_seq = [i ^ (i >> 1) for i in range(total_states)]

    for w in img_wires:
        qml.PauliX(wires=w)

    target_wire = img_wires[-1]
    control_wires = [est_wire] + list(img_wires[:-1])

    for step in range(total_states):
        curr_gray = gray_seq[step]
        pixel_idx = curr_gray

        if pixel_idx < len(flat_phases_1):
            phi1 = flat_phases_1[pixel_idx]
            phi2 = flat_phases_2[pixel_idx]

            if abs(phi1) > 1e-5:
                qml.ctrl(qml.PhaseShift, control=control_wires)(phi1 * power, wires=target_wire)
            if abs(phi2) > 1e-5:
                qml.ctrl(qml.PhaseShift, control=control_wires)(phi2 * power, wires=target_wire)

        if step < total_states - 1:
            next_gray = gray_seq[step + 1]
            diff = curr_gray ^ next_gray
            bit_changed_from_right = int(np.log2(diff))
            wire_to_flip = img_wires[n - 1 - bit_changed_from_right]
            qml.PauliX(wires=wire_to_flip)

    last_gray_bin = format(gray_seq[-1], f'0{n}b')
    for i, bit in enumerate(last_gray_bin):
        if bit == '0':
            qml.PauliX(wires=img_wires[i])

## Q-PIPE (main para ejecucion)
def run_qpipe_dynamic_threshold(raw_img, n_estimation_wires, prob_threshold):
    h, w = raw_img.shape
    n_pixels = h * w
    n_image_wires = int(np.ceil(np.log2(n_pixels)))
    total_wires = n_estimation_wires + n_image_wires

    dev = qml.device('default.qubit', wires=total_wires)

    @qml.qnode(dev)
    def edge_circuit(flat_p1, flat_p2):
        est_wires = list(range(n_estimation_wires))
        tgt_wires = list(range(n_estimation_wires, total_wires))

        for wire in est_wires + tgt_wires:
            qml.Hadamard(wires=wire)

        for i, est_wire in enumerate(reversed(est_wires)):
            power = 2**i
            apply_dual_phases_with_gray_code(flat_p1, flat_p2, tgt_wires, est_wire, power)

        qml.adjoint(qml.QFT)(wires=est_wires)
        return qml.probs(wires=est_wires + tgt_wires)

    shifted_x = np.roll(raw_img, shift=1, axis=1); shifted_x[:, 0] = 0
    shifted_y = np.roll(raw_img, shift=1, axis=0); shifted_y[0, :] = 0

    phases_base = (raw_img.flatten() / 255.0) * np.pi
    phases_neg_x = (-shifted_x.flatten() / 255.0) * np.pi
    phases_neg_y = (-shifted_y.flatten() / 255.0) * np.pi

    def extract_magnitude(probs): # Solución para "spectral leakage". 
        weighted_mag_sum = np.zeros(raw_img.shape, dtype=float)
        prob_sum = np.zeros(raw_img.shape, dtype=float)

        for k, p in enumerate(probs):
            if p > prob_threshold:
                bin_k = format(k, f'0{total_wires}b')
                est_int = int(bin_k[:n_estimation_wires], 2)
                pixel_idx = int(bin_k[n_estimation_wires:], 2)

                if pixel_idx < n_pixels:
                    row, col = np.unravel_index(pixel_idx, raw_img.shape)
                    raw_val = (est_int / (2**n_estimation_wires)) * 255.0
                    edge_magnitude = 2.0 * min(raw_val, 255.0 - raw_val)
                    weighted_mag_sum[row, col] += p * edge_magnitude
                    prob_sum[row, col] += p

        safe_prob_sum = np.where(prob_sum == 0, 1.0, prob_sum)
        return weighted_mag_sum / safe_prob_sum

    probs_x = edge_circuit(phases_base, phases_neg_x)
    probs_y = edge_circuit(phases_base, phases_neg_y)

    q_gx = extract_magnitude(probs_x)
    q_gy = extract_magnitude(probs_y)
    quantum_sobel = np.sqrt(q_gx**2 + q_gy**2)

    c_gx = np.abs(raw_img - shifted_x)
    c_gy = np.abs(raw_img - shifted_y)
    classic_sobel = np.sqrt(c_gx**2 + c_gy**2)

    return mean_absolute_error(classic_sobel.flatten(), quantum_sobel.flatten())

### Datasets (con resolucion fija 22x22 o 24x24) ###
def get_datasets_24x24(n_instances=5):
    datasets = {}
    resize_size = 22
    print("Loading and scaling datasets to 22x22...")
    
    mnist_data = fetch_openml('mnist_784', version=1, cache=True, parser='auto').data.values
    mnist_imgs = mnist_data.reshape(-1, 28, 28)[:n_instances]
    datasets['MNIST'] = [resize(img, (resize_size, resize_size), anti_aliasing=True, preserve_range=True) for img in mnist_imgs]

    faces = fetch_olivetti_faces().images[:n_instances]
    datasets['Olivetti Faces'] = [resize(img, (resize_size, resize_size), anti_aliasing=True) * 255.0 for img in faces]

    f_mnist = fetch_openml('Fashion-MNIST', version=1, cache=True, parser='auto').data.values
    f_mnist_imgs = f_mnist.reshape(-1, 28, 28)[:n_instances]
    datasets['Fashion-MNIST'] = [resize(img, (resize_size, resize_size), anti_aliasing=True, preserve_range=True) for img in f_mnist_imgs]

    med_images = []
    for _ in range(n_instances):
        base = np.zeros((resize_size, resize_size))
        cx, cy = np.random.randint(8, 16, size=2)
        base[cx-4:cx+4, cy-4:cy+4] = 150.0
        base = ndimage.gaussian_filter(base, sigma=1.5)
        noise = np.random.normal(1, 0.25, (resize_size, resize_size))
        img_med = np.clip(base * noise, 0, 255)
        med_images.append(img_med)
    datasets['Medical Synth'] = med_images

    return datasets

### MAIN ###
n_instances = 10 # Aumentar para resultados finales
datasets = get_datasets_24x24(n_instances)
thresholds = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
checkpoint_file = "qpipe_threshold_checkpoint_22pixels_10_INST.pkl"

if os.path.exists(checkpoint_file): # Cargar progreso en caso de falla.
    print(f"\n[INFO] Found checkpoint file: {checkpoint_file}. Loading previous progress...")
    with open(checkpoint_file, 'rb') as f:
        saved_data = pickle.load(f)
        results_mean = saved_data.get('results_mean', {})
        results_std = saved_data.get('results_std', {})
else:
    results_mean = {name: [] for name in datasets.keys()}
    results_std = {name: [] for name in datasets.keys()}

print("\n--- Starting Probability Threshold Analysis ---")
for name, images in datasets.items():
    # Verifica si el dataset actual ya se procesó por completo en una corrida anterior
    if name in results_mean and len(results_mean[name]) == len(thresholds):
        print(f"\n[SKIP] Dataset: {name} (Loaded from checkpoint)")
        continue
        
    print(f"\nProcessing Dataset: {name}")
    # Reiniciar la lista de este dataset en caso de un progreso a medias que fallo.
    results_mean[name] = []
    results_std[name] = []
    
    for t in thresholds:
        print(f"  -> Testing Threshold: {t}")
        mae_list = []
        for i, img in enumerate(images):
            mae = run_qpipe_dynamic_threshold(img, n_estimation_wires=8, prob_threshold=t)
            mae_list.append(mae)
            
        results_mean[name].append(np.mean(mae_list))
        results_std[name].append(np.std(mae_list))
        
    # Guardar el progreso al terminar exitosamente el dataset completo.
    print(f"[SAVE] Checkpointing progress for {name}...")
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({'results_mean': results_mean, 'results_std': results_std}, f)

## Grafica comparativa
plt.figure(figsize=(10, 7))

colors = ['darkblue', 'darkgreen', 'darkred', 'purple']
markers = ['o', 's', '^', 'D']

for (name, means), color, marker in zip(results_mean.items(), colors, markers):
    
    if not means: continue # Validar que existan los datos.
    
    means_arr = np.array(means)
    stds_arr = np.array(results_std[name])
    
    plt.plot(thresholds, means_arr, marker=marker, color=color, linewidth=2, label=name)
    plt.fill_between(thresholds, means_arr - stds_arr, means_arr + stds_arr, color=color, alpha=0.1)

plt.xscale('log') 
plt.gca().invert_xaxis() 

plt.title('Impact of Probability Threshold on Q-PIPE Accuracy\n(Resolution: $22\\times22$ pixels | 8 Estimation Qubits)', fontsize=14, pad=15)
plt.xlabel('Probability Threshold (Log Scale, Decreasing $\\rightarrow$)', fontsize=12)
plt.ylabel('Mean Absolute Error (Intensity Levels 0-255)', fontsize=12)
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend(loc='upper right', title="Datasets")

plt.tight_layout()
plt.savefig("qpipe_threshold_analysis_22pixels_10_INST.png", dpi=600)
plt.savefig("qpipe_threshold_analysis_22pixels_10_INST.eps")
plt.show()
print("Threshold comparative chart successfully generated.")

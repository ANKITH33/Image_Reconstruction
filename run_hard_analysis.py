import os
import shutil
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float

# --- Configuration ---
DATASET_DIR = "dataset_small"
RESULTS_DIR = "hard_version_results_small"
NOISE_LEVELS = [5] #, 8, 10]
LAMBDA_VALUES = [0.1, 1, 10]
NUM_SAMPLES_TO_SAVE = 3
NUM_IMAGES_TO_PROCESS = 5

# --- Helper Functions ---
def solve_hard_problem(X_corr, lambda_val):
    M, N = X_corr.shape
    X = cp.Variable((M, N))
    data_fidelity = cp.sum_squares(X - X_corr)
    regularizer = lambda_val * cp.tv(X)
    objective = cp.Minimize(data_fidelity + regularizer)
    constraints = [0 <= X, X <= 255]
    problem = cp.Problem(objective, constraints)
    # Changed solver to SCS, which is more robust for SOCPs
    problem.solve(solver='SCS', verbose=False)

    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return X.value
    return None

def calculate_mse(img1, img2):
    return np.mean(np.square(img1 - img2))

# --- Main Script ---
if os.path.exists(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)
os.makedirs(RESULTS_DIR)

ground_truth_dir = os.path.join(DATASET_DIR, "ground_truth")
image_files = sorted(os.listdir(ground_truth_dir))[:NUM_IMAGES_TO_PROCESS]

print(f"Processing {len(image_files)} fixed images for the HARD problem.")

all_mse_results = {level: {lam: [] for lam in LAMBDA_VALUES} for level in NOISE_LEVELS}

for noise_level in NOISE_LEVELS:
    print(f"--- Processing Noise Level: {noise_level}% ---")
    noisy_dir = os.path.join(DATASET_DIR, f"noise_{noise_level}")

    for i, filename in enumerate(image_files):
        X_true_path = os.path.join(ground_truth_dir, filename)
        X_corr_path = os.path.join(noisy_dir, filename)

        X_true = img_as_float(io.imread(X_true_path)) * 255.0
        X_corr = img_as_float(io.imread(X_corr_path)) * 255.0

        for lam in LAMBDA_VALUES:
            X_recon = solve_hard_problem(X_corr, lam)

            if X_recon is not None:
                mse = calculate_mse(X_recon, X_true)
                all_mse_results[noise_level][lam].append(mse)

                if i < NUM_SAMPLES_TO_SAVE:
                    sample_dir = os.path.join(RESULTS_DIR, "samples", f"noise_{noise_level}", f"lambda_{lam}")
                    os.makedirs(sample_dir, exist_ok=True)
                    save_path = os.path.join(sample_dir, f"recon_{filename}")
                    io.imsave(save_path, np.clip(X_recon, 0, 255).astype(np.uint8), check_contrast=False)

    print(f"Finished processing for noise level {noise_level}%.")

# --- Analysis and Plotting ---
plots_dir = os.path.join(RESULTS_DIR, "plots")
os.makedirs(plots_dir)

for noise_level in NOISE_LEVELS:
    avg_mse_per_lambda = {lam: np.mean(all_mse_results[noise_level][lam]) for lam in LAMBDA_VALUES}

    lambdas = list(avg_mse_per_lambda.keys())
    avg_mses = list(avg_mse_per_lambda.values())

    plt.figure()
    plt.plot(lambdas, avg_mses, marker='o', color='r')
    plt.xscale('log')
    plt.xlabel('Lambda (λ)')
    plt.ylabel('Average Mean Squared Error (MSE)')
    plt.title(f'Hard Problem (TV): MSE vs. Lambda for {noise_level}% Noise')
    plt.grid(True, which="both", ls="--")
    plot_path = os.path.join(plots_dir, f"mse_vs_lambda_noise_{noise_level}.png")
    plt.savefig(plot_path)
    plt.close()

print(f"\nAnalysis complete. Plots saved in '{plots_dir}'.")
print(f"Sample images saved in '{os.path.join(RESULTS_DIR, 'samples')}'.")

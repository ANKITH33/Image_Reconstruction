import os
import shutil
import time
import cvxpy as cp
import numpy as np
from skimage import io, img_as_float

# --- Configuration ---
DATASET_DIR = "dataset_small"
RESULTS_DIR = "optima/easy"
# Dictionary mapping noise_level to the specific lambda to use
OPTIMA_PAIRS = {
    5: 1.0,
    8: 1.2,
    10: 1.4
}

# --- Helper Functions ---
def solve_easy_problem(X_corr, lambda_val):
    M, N = X_corr.shape
    X = cp.Variable((M, N))
    data_fidelity = cp.sum_squares(X - X_corr)
    regularizer = lambda_val * (cp.sum_squares(X[:, 1:] - X[:, :-1]) + cp.sum_squares(X[1:, :] - X[:-1, :]))
    objective = cp.Minimize(data_fidelity + regularizer)
    constraints = [0 <= X, X <= 255]
    problem = cp.Problem(objective, constraints)

    start_time = time.time()
    problem.solve(solver='OSQP', verbose=False)
    end_time = time.time()

    solve_time = end_time - start_time

    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return problem.value, solve_time, X.value
    return None, None, None

def calculate_mse(img1, img2):
    return np.mean(np.square(img1 - img2))

# --- Main Script ---
if os.path.exists(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)
os.makedirs(RESULTS_DIR)

ground_truth_dir = os.path.join(DATASET_DIR, "ground_truth")
image_files = sorted(os.listdir(ground_truth_dir))

print("--- Running Optimal Value and Time Analysis for EASY Problem ---")

for noise_level, lambda_val in OPTIMA_PAIRS.items():
    print(f"\n[INFO] Processing Noise: {noise_level}%, Lambda: {lambda_val}")

    noisy_dir = os.path.join(DATASET_DIR, f"noise_{noise_level}")
    output_dir = os.path.join(RESULTS_DIR, f"noise_{noise_level}_lambda_{lambda_val}")
    os.makedirs(output_dir)

    total_obj_val = 0
    total_solve_time = 0
    total_mse = 0
    images_processed = 0

    for filename in image_files:
        X_true_path = os.path.join(ground_truth_dir, filename)
        X_corr_path = os.path.join(noisy_dir, filename)

        X_true = img_as_float(io.imread(X_true_path)) * 255.0
        X_corr = img_as_float(io.imread(X_corr_path)) * 255.0

        obj_val, solve_time, X_recon = solve_easy_problem(X_corr, lambda_val)

        if X_recon is not None:
            total_obj_val += obj_val
            total_solve_time += solve_time
            total_mse += calculate_mse(X_recon, X_true)
            images_processed += 1

            save_path = os.path.join(output_dir, f"recon_{filename}")
            io.imsave(save_path, np.clip(X_recon, 0, 255).astype(np.uint8), check_contrast=False)

    if images_processed > 0:
        avg_obj_val = total_obj_val / images_processed
        avg_solve_time = total_solve_time / images_processed
        avg_mse = total_mse / images_processed

        print(f"  Average Objective Value: {avg_obj_val:.2f}")
        print(f"  Average Solve Time: {avg_solve_time:.4f} seconds")
        print(f"  Average MSE (vs. Ground Truth): {avg_mse:.2f}")
    else:
        print("  No images were processed for this configuration.")

print("\nDone. Optimal images saved in 'optima/easy/'.")

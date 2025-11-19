import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.color import rgb2gray
import time

# --- Configuration ---
IMAGE_NAME = 'camera' # Options: 'camera', 'lena', 'astronaut', 'text', 'coins'
NOISE_LEVEL_PERCENT = 30 # Percentage of pixels to corrupt
LAMBDA_EASY = 10 # Example lambda for the easy problem
LAMBDA_HARD = 10 # Example lambda for the hard problem

# --- 1. Data Loading and Noise Generation ---

def load_and_corrupt_image(image_name, noise_level_percent):
    """Loads a clean image and adds salt-and-pepper noise."""
    if image_name == 'camera':
        X_true = data.camera()
    elif image_name == 'lena':
        X_true = rgb2gray(data.lena())
    elif image_name == 'astronaut':
        X_true = rgb2gray(data.astronaut())
    elif image_name == 'text':
        X_true = data.text()
    elif image_name == 'coins':
        X_true = data.coins()
    else:
        raise ValueError(f"Unknown image name: {image_name}")

    # Convert to float and scale to [0, 1] if not already
    X_true = img_as_float(X_true)

    # Scale to [0, 255] for our problem formulation
    X_true = X_true * 255.0

    # Generate salt-and-pepper noise
    # Note: skimage's random_noise generates values in [0,1],
    # so we need to work with that range and then scale back.
    amount = noise_level_percent / 100.0
    X_corr_float_0_1 = random_noise(X_true / 255.0, mode='s&p', amount=amount)
    X_corr = (X_corr_float_0_1 * 255.0).astype(np.float32)

    # Ensure values are within [0, 255] range after noise
    X_corr = np.clip(X_corr, 0, 255)

    print(f"Loaded '{image_name}' image of shape {X_true.shape}")
    print(f"Added {noise_level_percent}% salt-and-pepper noise.")
    return X_true, X_corr

X_true, X_corr = load_and_corrupt_image(IMAGE_NAME, NOISE_LEVEL_PERCENT)
M, N = X_true.shape # Get image dimensions

# --- 2. The 'Easy' Problem: L2 Regularized Denoising ---

def solve_easy_problem(X_corr, lambda_val):
    """Solves the L2 regularized denoising problem."""
    print(f"\n--- Solving Easy Problem (L2) with λ = {lambda_val} ---")
    X = cp.Variable((M, N))

    # Data Fidelity Term: ||X - X_corr||_F^2
    data_fidelity = cp.sum_squares(X - X_corr)

    # Regularization Term: λ * ||∇X||_F^2
    # Horizontal and Vertical Gradients
    # For simplicity, we'll use a basic diff which results in slightly smaller matrices
    # In a more rigorous implementation, padding or boundary handling would be needed.
    # CVXPY handles this internally with its gradient operator 'cp.diff'
    grad_h = X[:, 1:] - X[:, :-1] # Horizontal differences
    grad_v = X[1:, :] - X[:-1, :] # Vertical differences

    regularizer = lambda_val * (cp.sum_squares(grad_h) + cp.sum_squares(grad_v))

    objective = cp.Minimize(data_fidelity + regularizer)
    constraints = [0 <= X, X <= 255]

    problem = cp.Problem(objective, constraints)

    start_time = time.time()
    problem.solve(solver='ECOS', verbose=False) # ECOS is a good default for QCQP
    end_time = time.time()

    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        print(f"Easy Problem solved in {end_time - start_time:.2f} seconds.")
        print(f"Final Objective Value (Easy): {problem.value:.2f}")
        return X.value
    else:
        print(f"Easy Problem did not converge. Status: {problem.status}")
        return None

X_recon_easy = solve_easy_problem(X_corr, LAMBDA_EASY)

# --- 3. Display Results (Initial Visualization) ---

if X_recon_easy is not None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes.ravel()

    ax[0].imshow(X_true, cmap=plt.cm.gray)
    ax[0].set_title("Original Clean Image")
    ax[0].axis('off')

    ax[1].imshow(X_corr, cmap=plt.cm.gray)
    ax[1].set_title(f"Corrupted Image ({NOISE_LEVEL_PERCENT}% noise)")
    ax[1].axis('off')

    ax[2].imshow(X_recon_easy, cmap=plt.cm.gray)
    ax[2].set_title(f"Easy Problem Reconstruction (λ={LAMBDA_EASY})")
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()

    # Calculate and print MSE for the easy problem
    mse_easy = np.mean(np.square(X_recon_easy - X_true))
    print(f"Mean Squared Error (Easy Problem): {mse_easy:.2f}")

import os
import random
import shutil
import kagglehub
from skimage import io, color, util, img_as_ubyte

# --- Configuration ---
KAGGLE_DATASET_ID = "rajneesh231/salt-and-pepper-noise-images"
FINAL_DATASET_DIR = "dataset_small"
TEMP_DOWNLOAD_DIR = "temp_kaggle_download"
NUM_IMAGES = 15
PATCH_SIZE = 128
NOISE_LEVELS = [5, 8, 10]

# --- Main Script ---
print("--- Starting Full Dataset Preparation ---")

# 1. Cleanup old directories
if os.path.exists(FINAL_DATASET_DIR):
    shutil.rmtree(FINAL_DATASET_DIR)
    print(f"Removed old directory: '{FINAL_DATASET_DIR}'")
if os.path.exists(TEMP_DOWNLOAD_DIR):
    shutil.rmtree(TEMP_DOWNLOAD_DIR)
    print(f"Removed old directory: '{TEMP_DOWNLOAD_DIR}'")

# 2. Download dataset to a temporary location
print(f"Downloading dataset '{KAGGLE_DATASET_ID}'...")
cached_path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
shutil.copytree(cached_path, TEMP_DOWNLOAD_DIR, dirs_exist_ok=True)
print("Download complete.")

# 3. Create final directory structure
ground_truth_dir = os.path.join(FINAL_DATASET_DIR, "ground_truth")
os.makedirs(ground_truth_dir)
for level in NOISE_LEVELS:
    os.makedirs(os.path.join(FINAL_DATASET_DIR, f"noise_{level}"))
print(f"Created final directory structure in '{FINAL_DATASET_DIR}'.")

# 4. Select, process, and save images
source_images_dir = os.path.join(TEMP_DOWNLOAD_DIR, "Ground_truth")
all_images = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if len(all_images) < NUM_IMAGES:
    raise ValueError(f"Source has {len(all_images)} images, but {NUM_IMAGES} were requested.")

selected_images = random.sample(all_images, NUM_IMAGES)
print(f"Randomly selected {NUM_IMAGES} images to process.")

for filename in selected_images:
    # Load and convert to grayscale
    img_path = os.path.join(source_images_dir, filename)
    img_color = io.imread(img_path)
    img_gray = color.rgb2gray(img_color)

    # Crop a central patch
    h, w = img_gray.shape
    if h < PATCH_SIZE or w < PATCH_SIZE:
        print(f"Skipping {filename}, too small for {PATCH_SIZE}x{PATCH_SIZE} patch.")
        continue

    start_y = (h - PATCH_SIZE) // 2
    start_x = (w - PATCH_SIZE) // 2
    patch_gray = img_gray[start_y : start_y + PATCH_SIZE, start_x : start_x + PATCH_SIZE]

    # Save the ground truth patch
    patch_ubyte = img_as_ubyte(patch_gray)
    io.imsave(os.path.join(ground_truth_dir, filename), patch_ubyte, check_contrast=False)

    # Generate and save noisy versions
    for level in NOISE_LEVELS:
        amount = level / 100.0
        noisy_patch = util.random_noise(patch_gray, mode='s&p', amount=amount)
        noisy_patch_ubyte = img_as_ubyte(noisy_patch)

        noise_dir = os.path.join(FINAL_DATASET_DIR, f"noise_{level}")
        io.imsave(os.path.join(noise_dir, filename), noisy_patch_ubyte, check_contrast=False)

# 5. Final cleanup of temporary download
shutil.rmtree(TEMP_DOWNLOAD_DIR)
print(f"Cleaned up temporary directory: '{TEMP_DOWNLOAD_DIR}'")

print(f"\n--- Dataset preparation complete. ---")
print(f"Final dataset is ready in '{FINAL_DATASET_DIR}'.")

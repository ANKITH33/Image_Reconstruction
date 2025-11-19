import kagglehub
import shutil
import os

LOCAL_IMAGE_DIR = "imgs"

if not os.path.exists(LOCAL_IMAGE_DIR):
    os.makedirs(LOCAL_IMAGE_DIR)

cached_dataset_path = kagglehub.dataset_download("rajneesh231/salt-and-pepper-noise-images")

shutil.copytree(cached_dataset_path, LOCAL_IMAGE_DIR, dirs_exist_ok=True)

print(f"\nDone.")

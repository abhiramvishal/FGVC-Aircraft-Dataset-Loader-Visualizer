# a1_part1.py
# ---------------------------------------------
# FGVC-Aircraft Dataset Loader & Visualizer
# For COMP8430 Assignment 1 - Part 1
# ---------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ========== CONFIGURATION ========== #
image_dir = r'fgvc-aircraft-2013b\data\images'  # Folder containing .jpg images

train_file     = r'fgvc-aircraft-2013b\data\images_variant_train.txt'
test_file      = r'fgvc-aircraft-2013b\data\images_variant_test.txt'
trainval_file  = r'fgvc-aircraft-2013b\data\images_variant_trainval.txt'
val_file       = r'fgvc-aircraft-2013b\data\images_variant_val.txt'

# ========== LOADER FUNCTION ========== #
def load_image_labels(file_path):
    """
    Loads image IDs and their corresponding variant class from the FGVC-Aircraft .txt file.
    Returns a DataFrame with image_id, variant, and image_path.
    """
    image_ids = []
    variants = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            image_ids.append(parts[0])
            variants.append(" ".join(parts[1:]))

    df = pd.DataFrame({'image_id': image_ids, 'variant': variants})
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
    return df

# ========== PRELOAD DATAFRAMES ========== #
train_df     = load_image_labels(train_file)
test_df      = load_image_labels(test_file)
trainval_df  = load_image_labels(trainval_file)
val_df       = load_image_labels(val_file)

print("Train Images:", len(train_df))
print("Test Images:", len(test_df))
print("Train+Val Images:", len(trainval_df))
print("Validation Images:", len(val_df))

# ========== VISUALIZATION FUNCTION ========== #
def show_images(df, title, n_images=10):
    """
    Displays n_images from the DataFrame using matplotlib.
    Each image is shown with its variant label.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    n_cols = 5
    n_rows = (n_images + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 3 * n_rows))

    for i, (_, row) in enumerate(df.sample(n_images).iterrows()):
        try:
            img = Image.open(row['image_path'])
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(img)
            plt.title(row['variant'], fontsize=8)
            plt.axis('off')
        except Exception as e:
            print(f"Error loading {row['image_path']}: {e}")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
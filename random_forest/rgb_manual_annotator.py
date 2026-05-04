"""
rgb_manual_annotator.py

Manual annotation tool for generating cell masks on pseudoRGB images using Napari.

Description:
This script provides an interactive interface to manually annotate immune cells
on pseudoRGB tiles derived from FLIM data. The user paints cell regions directly
on the image to generate a binary mask, which is saved for downstream supervised
learning tasks.

Functionality:
- Loads a pseudoRGB image (single tile).
- Initializes an empty label mask.
- Allows manual painting using Napari's label layer.
- Saves the mask as a TIFF file by pressing the 's' key.

Annotation guidelines:
- Label value = 1 → immune cells
- Label value = 0 → background / non-cell regions

Only annotate:
- Small, cell-like structures (typically orange-toned)

Do NOT annotate:
- Melanin (intense red regions)
- Elastin / structural clumps
- Bright edges or imaging artifacts
- Background or noise

Output:
- Binary mask (uint8) aligned with the input image.
- Used for:
    - Pixel-wise Random Forest training
    - Positive patch extraction

Context in pipeline:
This script is part of the ground-truth generation stage for training
a supervised cell segmentation model. The resulting masks are later used
to extract labeled pixels and build the training dataset.

Usage:
- Paint cells using the brush tool on the "immune_cells_mask" layer
- Press 's' to save the mask automatically

"""

import napari
import numpy as np
from skimage.io import imread
from pathlib import Path
import tifffile as tiff


# ==== CONFIGURATION ====

RGB_IMAGE_PATH = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit04/Mosaic08_4x4_FOV600_z155_32Sp/random_forest/rgb/Im_00016_pseudoRGB.tif"

OUTPUT_MASK_PATH = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit04/Mosaic08_4x4_FOV600_z155_32Sp/random_forest/mask/Im_00016_mask.tif"


# ==== FUNCTIONS ====

def save_mask(layer):
    """
    Save the label mask as a TIFF file.
    """
    output_path = Path(OUTPUT_MASK_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask = layer.data.astype(np.uint8)
    tiff.imwrite(output_path, mask)

    print(f"Mask saved at: {output_path}")


def main():
    # Load pseudoRGB image
    img = imread(RGB_IMAGE_PATH)

    # Initialize empty mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Create Napari viewer
    viewer = napari.Viewer()

    # Add image layer
    viewer.add_image(img, name="pseudoRGB")

    # Add label layer for annotation
    labels_layer = viewer.add_labels(
        mask,
        name="immune_cells_mask"
    )

    # Keyboard shortcut to save mask
    @viewer.bind_key("s")
    def save_mask_shortcut(viewer):
        save_mask(labels_layer)

    # Instructions
    print("\n===== INSTRUCTIONS =====")
    print("1. Select the layer: 'immune_cells_mask'")
    print("2. Use the brush/paint tool")
    print("3. Paint ONLY real immune cells (label = 1)")
    print("4. DO NOT paint:")
    print("   - melanin (intense red regions)")
    print("   - elastin / clumps")
    print("   - bright edges or artifacts")
    print("   - background or noise")
    print("5. Press key 's' to save the mask")
    print(f"\nOutput path: {OUTPUT_MASK_PATH}")
    print("========================\n")

    napari.run()


if __name__ == "__main__":
    main()
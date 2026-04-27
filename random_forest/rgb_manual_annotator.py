import napari
import numpy as np
from skimage.io import imread
from pathlib import Path
import tifffile as tiff


RGB_IMAGE_PATH = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01/Mosaic07_4x4_FOV600_z150_32Sp/random_forest/rgb/Im_00006_pseudoRGB.tif"

OUTPUT_MASK_PATH = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01/Mosaic07_4x4_FOV600_z150_32Sp/random_forest/mask/Im_00006_mask.tif"


def save_mask(layer):
    output_path = Path(OUTPUT_MASK_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask = layer.data.astype(np.uint8)
    tiff.imwrite(output_path, mask)

    print(f"Máscara guardada en: {output_path}")


def main():
    img = imread(RGB_IMAGE_PATH)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    viewer = napari.Viewer()

    viewer.add_image(img, name="pseudoRGB")

    labels_layer = viewer.add_labels(
        mask,
        name="immune_cells_mask"
    )

    @viewer.bind_key("s")
    def save_mask_shortcut(viewer):
        save_mask(labels_layer)

    print("\n===== INSTRUCCIONES =====")
    print("1. Seleccioná la layer: immune_cells_mask")
    print("2. Usá brush/paint")
    print("3. Pintá SOLO células naranjas pequeñas con label = 1")
    print("4. NO pintes fondo, elastina, clumps, melanina, etc.")
    print("5. Presioná tecla 's' para guardar automáticamente")
    print(f"\nSe guardará en: {OUTPUT_MASK_PATH}")
    print("========================\n")

    napari.run()


if __name__ == "__main__":
    main()
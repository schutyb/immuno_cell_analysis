from pathlib import Path
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader


PATIENT_FOLDER = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
OUTPUT_PDF = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/positive_patches_review.pdf"

PATCH_DIR_NAME = "patch"
PATCH_MASK_DIR_NAME = "patch_mask"

PATCH_SIZE = 128
PAIRS_PER_PAGE = 8
DPI = 600


def find_patch_pairs(patient_folder):
    patient_folder = Path(patient_folder)

    patch_files = sorted(
        patient_folder.glob(f"visit*/Mosaic*/random_forest/{PATCH_DIR_NAME}/*.png")
    )

    pairs = []

    for patch_path in patch_files:
        rf_dir = patch_path.parent.parent
        mask_dir = rf_dir / PATCH_MASK_DIR_NAME
        mask_path = mask_dir / f"{patch_path.stem}_mask.png"

        if mask_path.exists():
            visit = patch_path.parts[
                [p.lower().startswith("visit") for p in patch_path.parts].index(True)
            ]
            mosaic = patch_path.parts[
                [p.lower().startswith("mosaic") for p in patch_path.parts].index(True)
            ]
            pairs.append((visit, mosaic, patch_path, mask_path))
        else:
            print(f"Missing mask for: {patch_path.name}")

    return pairs


def draw_header(c, visit, mosaic, page_w, page_h):
    c.setFont("Helvetica-Bold", 14)
    c.drawString(0.5 * inch, page_h - 0.45 * inch, f"{visit} | {mosaic}")

    c.setFont("Helvetica", 9)
    c.drawString(0.5 * inch, page_h - 0.65 * inch, "RGB patch | corresponding mask")


def draw_pair(c, patch_path, mask_path, x, y, img_size):
    rgb = Image.open(patch_path).convert("RGB")
    mask = Image.open(mask_path).convert("RGB")

    c.drawImage(ImageReader(rgb), x, y, width=img_size, height=img_size)
    c.drawImage(ImageReader(mask), x + img_size + 0.12 * inch, y, width=img_size, height=img_size)

    c.setFont("Helvetica", 6)
    c.drawString(x, y - 0.10 * inch, patch_path.stem[:65])


def make_pdf():
    pairs = find_patch_pairs(PATIENT_FOLDER)

    if len(pairs) == 0:
        raise RuntimeError("No patch/mask pairs found.")

    page_w, page_h = letter
    c = canvas.Canvas(OUTPUT_PDF, pagesize=letter, pageCompression=0)

    img_size = 1.15 * inch
    x_positions = [0.55 * inch, 4.20 * inch]
    y_start = page_h - 2.0 * inch
    y_step = 1.65 * inch

    current_group = None
    count_on_page = 0

    for visit, mosaic, patch_path, mask_path in pairs:
        group = (visit, mosaic)

        if group != current_group or count_on_page >= PAIRS_PER_PAGE:
            if current_group is not None:
                c.showPage()

            current_group = group
            count_on_page = 0
            draw_header(c, visit, mosaic, page_w, page_h)

        row = count_on_page // 2
        col = count_on_page % 2

        x = x_positions[col]
        y = y_start - row * y_step

        draw_pair(c, patch_path, mask_path, x, y, img_size)

        count_on_page += 1

    c.save()
    print(f"PDF saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    make_pdf()
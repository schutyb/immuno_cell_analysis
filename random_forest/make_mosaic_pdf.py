from pathlib import Path
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch


PATIENT_FOLDER = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
OUTPUT_PDF = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/p449_pseudoRGB_mosaics_record.pdf"

IMAGE_NAME = "mosaic_4x4_pseudoRGB.png"

PAGE_WIDTH = 13.33 * inch   # widescreen-like
PAGE_HEIGHT = 10.0 * inch
MARGIN = 0.35 * inch
TITLE_HEIGHT = 0.45 * inch


def find_mosaic_pngs(patient_folder):
    patient_folder = Path(patient_folder)

    files = sorted(patient_folder.glob(f"visit*/Mosaic*/random_forest/rgb/{IMAGE_NAME}"))
    return files


def get_visit_and_mosaic(path):
    parts = path.parts

    visit = next((p for p in parts if p.lower().startswith("visit")), "unknown_visit")
    mosaic = next((p for p in parts if p.lower().startswith("mosaic")), "unknown_mosaic")

    return visit, mosaic


def add_image_page(c, img_path):
    visit, mosaic = get_visit_and_mosaic(img_path)

    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    title = f"{visit} | {mosaic}"

    c.setPageSize((PAGE_WIDTH, PAGE_HEIGHT))

    # title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN, PAGE_HEIGHT - MARGIN, title)

    c.setFont("Helvetica", 9)
    c.drawString(MARGIN, PAGE_HEIGHT - MARGIN - 0.18 * inch, img_path.name)

    # available image area
    avail_w = PAGE_WIDTH - 2 * MARGIN
    avail_h = PAGE_HEIGHT - 2 * MARGIN - TITLE_HEIGHT

    scale = min(avail_w / img_w, avail_h / img_h)

    draw_w = img_w * scale
    draw_h = img_h * scale

    x = (PAGE_WIDTH - draw_w) / 2
    y = MARGIN

    c.drawImage(
        ImageReader(img),
        x,
        y,
        width=draw_w,
        height=draw_h,
        preserveAspectRatio=True,
        mask="auto"
    )

    c.showPage()


def main():
    image_paths = find_mosaic_pngs(PATIENT_FOLDER)

    print(f"Found {len(image_paths)} mosaic PNG files")

    if len(image_paths) == 0:
        raise RuntimeError("No mosaic PNG files found.")

    c = canvas.Canvas(OUTPUT_PDF, pageCompression=0)

    for img_path in image_paths:
        print(f"Adding: {img_path}")
        add_image_page(c, img_path)

    c.save()

    print(f"\nPDF saved:")
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
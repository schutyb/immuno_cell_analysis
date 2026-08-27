#!/usr/bin/env python3
"""Create one PDF page per mosaic with RGB, phasors, and pseudocolors.

The report combines existing visualization products without modifying any FLIM
or calibrated phasor data. Each landscape A3 page contains:

* the intensity RGB mosaic, normalized over the whole mosaic, on the left;
* green phasor and green phase pseudocolor on the upper right; and
* blue phasor and blue phase pseudocolor on the lower right.

The phasor panels come from the previously approved PhasorPlot output using the
brightest 35% DC per tile, ``cmin=20``, and ``RdYlGn_r``. Pseudocolors use
``reds_to_greens`` for green and ``blues_to_greens`` for blue. If blue is not
available, the lower row is explicitly marked as unavailable.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

DEFAULT_DATA_ROOT = Path("/Users/schutyb/Documents/balu_lab/dod/data_curated")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "pdf"
    / "all_patients_rgb_phasor_pseudocolor.pdf"
)
DEFAULT_PATIENTS = ("p427", "p437", "p439", "p449", "p476")

PAGE_WIDTH, PAGE_HEIGHT = landscape(A3)
MARGIN = 24.0
HEADER_HEIGHT = 58.0
FOOTER_HEIGHT = 31.0
GAP = 14.0
LEFT_WIDTH = 500.0
CARD_TITLE_HEIGHT = 25.0


@dataclass(frozen=True)
class MosaicRecord:
    patient: str
    visit: str
    mosaic: str
    acquisition_type: str
    channels: tuple[str, ...]
    intensity_rgb: Path
    phasor_plot: Path
    green_pseudocolor: Path
    blue_pseudocolor: Path | None


def natural_key(value: str | Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def record_key(row: dict[str, str], mosaic_field: str) -> tuple[str, str, str]:
    return row["patient"], row["visit"], row[mosaic_field]


def parse_channels(value: str) -> tuple[str, ...]:
    channels = tuple(part for part in value.split("+") if part)
    if not channels or any(channel not in {"green", "blue"} for channel in channels):
        raise ValueError(f"Unexpected channel list: {value!r}")
    return channels


def resolve_phasor_plot(
    manifest: Path,
    row: dict[str, str],
) -> Path:
    """Resolve a phasor PNG, including stale incremental-manifest paths."""
    recorded = Path(row["output_png"])
    if recorded.is_file():
        return recorded
    canonical = manifest.parent / row["patient"] / row["visit"] / recorded.name
    return canonical


def collect_records(
    rgb_manifest: Path,
    phasor_manifest: Path,
    pseudocolor_manifest: Path,
    patients: set[str],
) -> list[MosaicRecord]:
    rgb_lookup = {
        record_key(row, "primary_mosaic"): row
        for row in read_rows(rgb_manifest)
        if row["patient"] in patients and row.get("status") == "ok"
    }

    phasor_lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in read_rows(phasor_manifest):
        if row["patient"] not in patients:
            continue
        key = record_key(row, "mosaic")
        previous = phasor_lookup.setdefault(key, row)
        if previous["output_png"] != row["output_png"]:
            raise ValueError(f"Different phasor PNGs found for {key}")

    records: list[MosaicRecord] = []
    missing: list[str] = []
    for row in read_rows(pseudocolor_manifest):
        if row["patient"] not in patients:
            continue
        key = record_key(row, "mosaic")
        rgb_row = rgb_lookup.get(key)
        phasor_row = phasor_lookup.get(key)
        if rgb_row is None or phasor_row is None:
            missing.append(
                f"{'/'.join(key)}: RGB={rgb_row is not None}, "
                f"phasor={phasor_row is not None}"
            )
            continue

        channels = parse_channels(row["channels"])
        paths = {
            "intensity_rgb": Path(rgb_row["output_rgb"]),
            "phasor_plot": resolve_phasor_plot(phasor_manifest, phasor_row),
            "green_pseudocolor": Path(row["green_png"]),
        }
        blue_path = Path(row["blue_png"]) if "blue" in channels else None
        absent = [
            f"{name}={path}" for name, path in paths.items() if not path.is_file()
        ]
        if blue_path is not None and not blue_path.is_file():
            absent.append(f"blue_pseudocolor={blue_path}")
        if absent:
            missing.append(f"{'/'.join(key)}: " + "; ".join(absent))
            continue

        records.append(
            MosaicRecord(
                patient=key[0],
                visit=key[1],
                mosaic=key[2],
                acquisition_type=row["acquisition_type"],
                channels=channels,
                intensity_rgb=paths["intensity_rgb"],
                phasor_plot=paths["phasor_plot"],
                green_pseudocolor=paths["green_pseudocolor"],
                blue_pseudocolor=blue_path,
            )
        )

    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise RuntimeError(
            f"Incomplete inputs for {len(missing)} mosaic(s):\n{details}"
        )
    records.sort(
        key=lambda item: natural_key(f"{item.patient}/{item.visit}/{item.mosaic}")
    )
    if not records:
        raise RuntimeError("No complete mosaic records were found")
    return records


def flatten_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if "A" in image.getbands():
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def save_preview(
    source: Path,
    destination: Path,
    max_size: tuple[int, int],
    crop: tuple[int, int, int, int] | None = None,
    quality: int = 92,
) -> Path:
    with Image.open(source) as image:
        image.load()
        if crop is not None:
            image = image.crop(crop)
        image = flatten_to_rgb(image)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(
            destination,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
        )
    return destination


def prepare_page_images(
    record: MosaicRecord,
    cache_dir: Path,
    page_number: int,
) -> dict[str, Path]:
    prefix = cache_dir / f"page_{page_number:03d}"
    previews = {
        "rgb": save_preview(
            record.intensity_rgb,
            prefix.with_name(f"{prefix.name}_rgb.jpg"),
            (1800, 1800),
            quality=93,
        ),
        "green_pseudocolor": save_preview(
            record.green_pseudocolor,
            prefix.with_name(f"{prefix.name}_green_pseudocolor.jpg"),
            (1000, 1000),
            quality=92,
        ),
    }
    if record.blue_pseudocolor is not None:
        previews["blue_pseudocolor"] = save_preview(
            record.blue_pseudocolor,
            prefix.with_name(f"{prefix.name}_blue_pseudocolor.jpg"),
            (1000, 1000),
            quality=92,
        )

    with Image.open(record.phasor_plot) as image:
        width, height = image.size
    top = int(round(height * 0.08))
    if "blue" in record.channels:
        midpoint = width // 2
        phasor_crops = {
            "green_phasor": (0, top, midpoint, height),
            "blue_phasor": (midpoint, top, width, height),
        }
    else:
        phasor_crops = {"green_phasor": (0, top, width, height)}
    for name, crop in phasor_crops.items():
        previews[name] = save_preview(
            record.phasor_plot,
            prefix.with_name(f"{prefix.name}_{name}.jpg"),
            (1100, 900),
            crop=crop,
            quality=95,
        )
    return previews


def draw_fitted_image(
    pdf: canvas.Canvas,
    path: Path,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    with Image.open(path) as image:
        image_width, image_height = image.size
    scale = min(width / image_width, height / image_height)
    draw_width = image_width * scale
    draw_height = image_height * scale
    draw_x = x + (width - draw_width) / 2.0
    draw_y = y + (height - draw_height) / 2.0
    pdf.drawImage(
        ImageReader(str(path)),
        draw_x,
        draw_y,
        width=draw_width,
        height=draw_height,
        preserveAspectRatio=True,
        mask="auto",
    )


def draw_card(
    pdf: canvas.Canvas,
    title: str,
    image_path: Path,
    x: float,
    y: float,
    width: float,
    height: float,
    accent: colors.Color,
) -> None:
    pdf.setFillColor(colors.HexColor("#F7F8FA"))
    pdf.setStrokeColor(colors.HexColor("#D5DAE0"))
    pdf.setLineWidth(0.7)
    pdf.roundRect(x, y, width, height, 7, fill=1, stroke=1)
    pdf.setFillColor(accent)
    pdf.roundRect(x, y + height - 5, width, 5, 3, fill=1, stroke=0)
    pdf.setFillColor(colors.HexColor("#18212B"))
    pdf.setFont("Helvetica-Bold", 11.2)
    pdf.drawCentredString(x + width / 2.0, y + height - 19.0, title)
    padding = 7.0
    draw_fitted_image(
        pdf,
        image_path,
        x + padding,
        y + padding,
        width - 2 * padding,
        height - CARD_TITLE_HEIGHT - padding,
    )


def draw_unavailable_card(
    pdf: canvas.Canvas,
    title: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    pdf.setFillColor(colors.HexColor("#F3F4F6"))
    pdf.setStrokeColor(colors.HexColor("#C9CED6"))
    pdf.roundRect(x, y, width, height, 7, fill=1, stroke=1)
    pdf.setFillColor(colors.HexColor("#AAB2BD"))
    pdf.roundRect(x, y + height - 5, width, 5, 3, fill=1, stroke=0)
    pdf.setFillColor(colors.HexColor("#364152"))
    pdf.setFont("Helvetica-Bold", 11.2)
    pdf.drawCentredString(x + width / 2.0, y + height - 19.0, title)
    pdf.setFillColor(colors.HexColor("#697386"))
    pdf.setFont("Helvetica", 15)
    pdf.drawCentredString(x + width / 2.0, y + height / 2.0, "Blue channel unavailable")
    pdf.setFont("Helvetica", 10)
    pdf.drawCentredString(
        x + width / 2.0,
        y + height / 2.0 - 18,
        "No synthetic blue values were added",
    )


def draw_page(
    pdf: canvas.Canvas,
    record: MosaicRecord,
    previews: dict[str, Path],
    page_number: int,
    page_count: int,
) -> None:
    pdf.setFillColor(colors.white)
    pdf.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)

    title = f"{record.patient} | {record.visit} | {record.mosaic}"
    pdf.setFillColor(colors.HexColor("#15202B"))
    pdf.setFont("Helvetica-Bold", 17)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 28, title)
    pdf.setFillColor(colors.HexColor("#5B6573"))
    pdf.setFont("Helvetica", 10.5)
    pdf.drawString(
        MARGIN,
        PAGE_HEIGHT - 45,
        f"{record.acquisition_type} | intensity RGB, calibrated phasors, "
        "and phase pseudocolor",
    )
    pdf.setStrokeColor(colors.HexColor("#D7DCE2"))
    pdf.line(
        MARGIN,
        PAGE_HEIGHT - HEADER_HEIGHT,
        PAGE_WIDTH - MARGIN,
        PAGE_HEIGHT - HEADER_HEIGHT,
    )

    content_top = PAGE_HEIGHT - HEADER_HEIGHT - 8
    content_bottom = FOOTER_HEIGHT + 8
    content_height = content_top - content_bottom
    right_x = MARGIN + LEFT_WIDTH + GAP
    right_width = PAGE_WIDTH - MARGIN - right_x
    panel_width = (right_width - GAP) / 2.0
    panel_height = (content_height - GAP) / 2.0
    upper_y = content_bottom + panel_height + GAP
    lower_y = content_bottom

    draw_card(
        pdf,
        "Intensity RGB - whole-mosaic normalization",
        previews["rgb"],
        MARGIN,
        content_bottom,
        LEFT_WIDTH,
        content_height,
        colors.HexColor("#57606A"),
    )
    draw_card(
        pdf,
        "Green phasor - top 35% DC, cmin=20",
        previews["green_phasor"],
        right_x,
        upper_y,
        panel_width,
        panel_height,
        colors.HexColor("#179C52"),
    )
    draw_card(
        pdf,
        "Green pseudocolor - reds_to_greens",
        previews["green_pseudocolor"],
        right_x + panel_width + GAP,
        upper_y,
        panel_width,
        panel_height,
        colors.HexColor("#179C52"),
    )

    if "blue" in record.channels:
        draw_card(
            pdf,
            "Blue phasor - top 35% DC, cmin=20",
            previews["blue_phasor"],
            right_x,
            lower_y,
            panel_width,
            panel_height,
            colors.HexColor("#1685D1"),
        )
        draw_card(
            pdf,
            "Blue pseudocolor - blues_to_greens",
            previews["blue_pseudocolor"],
            right_x + panel_width + GAP,
            lower_y,
            panel_width,
            panel_height,
            colors.HexColor("#1685D1"),
        )
    else:
        draw_unavailable_card(
            pdf,
            "Blue phasor",
            right_x,
            lower_y,
            panel_width,
            panel_height,
        )
        draw_unavailable_card(
            pdf,
            "Blue pseudocolor",
            right_x + panel_width + GAP,
            lower_y,
            panel_width,
            panel_height,
        )

    pdf.setFillColor(colors.HexColor("#5F6875"))
    pdf.setFont("Helvetica", 8.3)
    pdf.drawString(
        MARGIN,
        17,
        "Phasors: calibrated G/S, median 7x7 twice, brightest 35% DC per tile, "
        "histogram bins >=20 counts, RdYlGn_r.",
    )
    pdf.drawRightString(
        PAGE_WIDTH - MARGIN,
        17,
        f"Page {page_number}/{page_count}",
    )


def make_pdf(records: list[MosaicRecord], output: Path, cache_dir: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial.pdf")
    partial.unlink(missing_ok=True)
    document = canvas.Canvas(str(partial), pagesize=landscape(A3), pageCompression=1)
    document.setTitle("All patients - RGB, calibrated phasors, and pseudocolors")
    document.setAuthor("immuno_cell_analysis")
    document.setSubject(
        "Whole-mosaic intensity RGB with green/blue phasors and phase pseudocolors"
    )
    try:
        for page_number, record in enumerate(records, start=1):
            print(
                f"[{page_number}/{len(records)}] {record.patient} | "
                f"{record.visit} | {record.mosaic}",
                flush=True,
            )
            previews = prepare_page_images(record, cache_dir, page_number)
            draw_page(document, record, previews, page_number, len(records))
            document.showPage()
        document.save()
        partial.replace(output)
    except Exception:
        try:
            document.save()
        except Exception:
            pass
        partial.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--rgb-manifest", type=Path)
    parser.add_argument("--phasor-manifest", type=Path)
    parser.add_argument("--pseudocolor-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patients", nargs="+", default=list(DEFAULT_PATIENTS))
    parser.add_argument("--max-mosaics", type=int)
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    calibrated_root = args.data_root / "calibrated_filtered_phasor"
    args.rgb_manifest = (
        args.rgb_manifest.expanduser().resolve()
        if args.rgb_manifest
        else args.data_root / "rgb_generation_manifest_all_patients.csv"
    )
    args.phasor_manifest = (
        args.phasor_manifest.expanduser().resolve()
        if args.phasor_manifest
        else calibrated_root
        / "phasorplot_hist2d_top35_cmin20"
        / "phasorplot_hist2d_manifest.csv"
    )
    args.pseudocolor_manifest = (
        args.pseudocolor_manifest.expanduser().resolve()
        if args.pseudocolor_manifest
        else calibrated_root
        / "phase_rgb_mosaics_mosaic_normalized"
        / "phase_rgb_manifest.csv"
    )
    args.output = args.output.expanduser().resolve()
    if args.max_mosaics is not None and args.max_mosaics < 1:
        parser.error("--max-mosaics must be positive")
    return args


def main() -> int:
    args = parse_args()
    records = collect_records(
        args.rgb_manifest,
        args.phasor_manifest,
        args.pseudocolor_manifest,
        set(args.patients),
    )
    if args.max_mosaics is not None:
        records = records[: args.max_mosaics]
    cache_dir = (
        Path(__file__).resolve().parents[1]
        / "tmp"
        / "pdfs"
        / f"{args.output.stem}_cache"
    )
    print(f"Mosaics: {len(records)}")
    print(f"Output:  {args.output}")
    make_pdf(records, args.output, cache_dir)
    print(f"Saved:   {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

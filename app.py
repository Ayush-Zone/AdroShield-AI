from pathlib import Path
import shutil
import uuid
import math
import re

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pymupdf as fitz

except ImportError:

    try:
        import fitz

    except ImportError:
        fitz = None

from flask import Flask, request, render_template_string, send_from_directory
from werkzeug.utils import secure_filename

from photo_integrity import PhotoIntegrityDetector
from document_tampering import DocumentTamperingDetector


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
PREVIEW_DIR = BASE_DIR / "previews"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}

DOCUMENT_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024


# ============================================================
# DETECTORS
# ============================================================

print("=" * 70)
print("Loading AdroShield-AI")
print("=" * 70)

print("Loading Photo Integrity Detector...")
photo_detector = PhotoIntegrityDetector()
print("Photo Integrity Detector ready.")

print("Loading Document Tampering Detector...")
document_detector = DocumentTamperingDetector()
print("Document Tampering Detector ready.")

print("=" * 70)


# ============================================================
# HELPERS
# ============================================================

def allowed_image(filename):
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


def allowed_document(filename):
    return Path(filename).suffix.lower() in DOCUMENT_EXTENSIONS


def clean_value(value):
    if value is None:
        return ""

    if isinstance(value, float):
        return round(value, 6)

    if isinstance(value, dict):
        return {
            str(k): clean_value(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [clean_value(v) for v in value]

    return value


def percentage(value):
    try:
        value = float(value)

        if value <= 1:
            value *= 100

        return f"{value:.2f}%"

    except Exception:
        return str(value)


def save_upload(file):
    original_name = secure_filename(file.filename)

    filename = (
        uuid.uuid4().hex[:10]
        + "_"
        + original_name
    )

    path = UPLOAD_DIR / filename

    file.save(path)

    return path


def create_preview(source_path):
    """Create a browser preview copy of an uploaded image."""
    source_path = Path(source_path)

    if not source_path.exists():
        return None

    preview_name = (
        uuid.uuid4().hex[:12]
        + "_"
        + secure_filename(source_path.name)
    )

    destination = PREVIEW_DIR / preview_name

    try:
        shutil.copy2(source_path, destination)
        return "/preview/" + destination.name
    except Exception:
        return None


def copy_annotations(result, output_dir):
    annotation_urls = []

    files = result.get("annotated_files", [])

    if not isinstance(files, list):
        files = [files]

    for file in files:

        if not file:
            continue

        source = Path(file)

        if not source.exists():
            continue

        destination = output_dir / source.name

        try:
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)

        except Exception:
            try:
                shutil.copy2(source, destination)
            except Exception:
                continue

        annotation_urls.append(
            "/output/"
            + output_dir.name
            + "/"
            + destination.name
        )

    return annotation_urls


def normalize_flags(result):
    flags = result.get("flags", [])

    if not isinstance(flags, list):
        flags = [flags]

    normalized = []

    for flag in flags:

        if not isinstance(flag, dict):
            continue

        normalized.append({
            "severity": flag.get("severity", ""),
            "page": flag.get("page", ""),
            "field": flag.get("field", ""),
            "check": flag.get("check", ""),
            "reason": flag.get("reason", ""),
            "bbox": flag.get("bbox", ""),
            "confidence": flag.get("confidence", ""),
        })

    return normalized


def _number(value):
    """Return a finite float when possible."""
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return None


def _bbox_xywh(bbox):
    """Normalize common bbox representations to x, y, width, height."""
    if bbox is None or bbox == "":
        return None

    values = []

    if isinstance(bbox, str):
        # Supports strings such as "[1388, 1003, 130, 41]"
        # and "1388, 1003, 130, 41".
        values = re.findall(r"[-+]?\d+(?:\.\d+)?", bbox)
    elif isinstance(bbox, (list, tuple)):
        values = list(bbox)
    elif isinstance(bbox, dict):
        # Accept either xywh or x0/y0/x1/y1 style dictionaries.
        x = _number(bbox.get("x", bbox.get("x0")))
        y = _number(bbox.get("y", bbox.get("y0")))
        w = _number(bbox.get("width", bbox.get("w")))
        h = _number(bbox.get("height", bbox.get("h")))
        if w is not None and h is not None and x is not None and y is not None:
            return x, y, abs(w), abs(h)

        x0 = _number(bbox.get("x0"))
        y0 = _number(bbox.get("y0"))
        x1 = _number(bbox.get("x1"))
        y1 = _number(bbox.get("y1"))
        if None not in (x0, y0, x1, y1):
            return min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
        return None

    if len(values) < 4:
        return None

    nums = [_number(v) for v in values[:4]]
    if any(v is None for v in nums):
        return None

    x, y, a, b = nums

    # The detector's current output uses [x, y, width, height].
    # Keep that convention while tolerating x0,y0,x1,y1 values when
    # the final coordinates are clearly larger than the origin.
    if a > 0 and b > 0:
        return x, y, a, b

    return None


def _page_sizes(document_path):
    """Return page/image sizes in the same coordinate space as the detector when possible."""
    path = Path(document_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf" and fitz is not None:
        try:
            doc = fitz.open(str(path))
            sizes = []
            for page in doc:
                rect = page.rect
                sizes.append((float(rect.width), float(rect.height)))
            doc.close()
            return sizes
        except Exception:
            return []

    if Image is not None and suffix in IMAGE_EXTENSIONS:
        try:
            with Image.open(path) as image:
                return [(float(image.width), float(image.height))]
        except Exception:
            return []

    return []


def _union_area(rectangles):
    """Exact union area for axis-aligned rectangles using x-sweep strips."""
    if not rectangles:
        return 0.0

    xs = sorted({x for r in rectangles for x in (r[0], r[2])})
    area = 0.0

    for x0, x1 in zip(xs, xs[1:]):
        if x1 <= x0:
            continue

        intervals = []
        for rx0, ry0, rx1, ry1 in rectangles:
            if rx0 < x1 and rx1 > x0 and ry1 > ry0:
                intervals.append((ry0, ry1))

        if not intervals:
            continue

        intervals.sort()
        covered = 0.0
        cy0, cy1 = intervals[0]

        for iy0, iy1 in intervals[1:]:
            if iy0 <= cy1:
                cy1 = max(cy1, iy1)
            else:
                covered += max(0.0, cy1 - cy0)
                cy0, cy1 = iy0, iy1

        covered += max(0.0, cy1 - cy0)
        area += (x1 - x0) * covered

    return area


def _visual_page_sizes(document_path, result=None):
    """Return page sizes in the coordinate space used by visual annotations.

    The document detector's bounding boxes are produced against rendered page
    images, not PDF points.  Using PDF page.rect directly therefore makes
    pixel-space boxes appear to be outside the page and can incorrectly yield
    0.00%.  When annotated images are available, their real pixel dimensions
    are the most reliable coordinate system for the boxes.
    """
    sizes = []

    # 1) Prefer the actual annotated images returned by the detector.
    if isinstance(result, dict):
        files = result.get("annotated_files", [])
        if not isinstance(files, list):
            files = [files]

        for file in files:
            if not file:
                continue
            path = Path(str(file))
            if not path.exists() or Image is None:
                continue
            try:
                with Image.open(path) as image:
                    sizes.append((float(image.width), float(image.height)))
            except Exception:
                continue

    if sizes:
        return sizes

    # 2) Normal image uploads already use pixel coordinates.
    path = Path(document_path)
    if Image is not None and path.suffix.lower() in IMAGE_EXTENSIONS:
        try:
            with Image.open(path) as image:
                return [(float(image.width), float(image.height))]
        except Exception:
            pass

    # 3) PDF fallback: use rendered pixel dimensions rather than PDF points.
    #    144 DPI gives a 2x scale over the common 72-point PDF coordinate base.
    if path.suffix.lower() == ".pdf" and fitz is not None:
        try:
            doc = fitz.open(str(path))
            rendered = []
            scale = 2.0
            for page in doc:
                rect = page.rect
                rendered.append((float(rect.width) * scale,
                                 float(rect.height) * scale))
            doc.close()
            return rendered
        except Exception:
            pass

    return []


def calculate_tampered_percentage(document_path, flags, result=None):
    """Estimate the percentage of document *content* affected by findings.

    A raw page-area calculation is misleading for documents because a small
    bounding box around a changed value can represent an entire data item,
    while the page contains large blank margins and whitespace.  For PDFs we
    therefore use content units (words) as the primary denominator and use
    visual area as a fallback / secondary signal.

    Rules:
      * metadata / structure findings never contribute to Tampered Content %
      * text/font/value findings are measured by affected text content
      * image/pixel findings are measured by visual area
      * duplicate/overlapping boxes are counted only once
    """
    path = Path(document_path)
    visual_flags = []

    for flag in flags:
        bbox = _bbox_xywh(flag.get("bbox"))
        if bbox is None:
            continue

        category = _finding_category(flag)
        if category in {"metadata", "structure"}:
            continue

        try:
            page_no = int(float(flag.get("page", 1)))
        except Exception:
            page_no = 1
        if page_no < 1:
            page_no += 1

        visual_flags.append({
            "page": page_no,
            "bbox": bbox,
            "category": category,
        })

    if not visual_flags:
        return 0.0

    # --------------------------------------------------------
    # PDF: measure affected text/content units first.
    # --------------------------------------------------------
    if path.suffix.lower() == ".pdf" and fitz is not None:
        try:
            doc = fitz.open(str(path))
            total_words = 0
            affected_words = set()
            area_rects = {}
            page_areas = {}

            # The detector annotations are rendered at 144 DPI (2x the
            # 72-point PDF coordinate system).
            scale = 2.0

            for page_index, page in enumerate(doc, start=1):
                page_width = float(page.rect.width) * scale
                page_height = float(page.rect.height) * scale
                page_areas[page_index] = page_width * page_height

                words = page.get_text("words") or []
                page_words = []
                for word_index, word in enumerate(words):
                    if len(word) < 5:
                        continue
                    x0, y0, x1, y1 = [float(v) * scale for v in word[:4]]
                    text = str(word[4]).strip()
                    if not text:
                        continue
                    page_words.append((word_index, x0, y0, x1, y1, text))

                total_words += len(page_words)

                page_flags = [f for f in visual_flags if f["page"] == page_index]
                if not page_flags:
                    continue

                for flag in page_flags:
                    x, y, w, h = flag["bbox"]
                    rect = (
                        max(0.0, min(page_width, x)),
                        max(0.0, min(page_height, y)),
                        max(0.0, min(page_width, x + abs(w))),
                        max(0.0, min(page_height, y + abs(h))),
                    )
                    rx0, ry0, rx1, ry1 = rect
                    if rx1 <= rx0 or ry1 <= ry0:
                        continue

                    # Store visual area too; used as a fallback for image
                    # findings and as a sanity check against text coverage.
                    area_rects.setdefault(page_index, []).append(rect)

                    # A word is affected when its center falls inside the
                    # suspicious region OR the two rectangles meaningfully
                    # overlap.  The overlap test handles small boxes around
                    # text baselines more reliably than center-only matching.
                    for word_index, wx0, wy0, wx1, wy1, _ in page_words:
                        ix0 = max(rx0, wx0)
                        iy0 = max(ry0, wy0)
                        ix1 = min(rx1, wx1)
                        iy1 = min(ry1, wy1)
                        if ix1 > ix0 and iy1 > iy0:
                            affected_words.add((page_index, word_index))

            doc.close()

            if total_words > 0 and affected_words:
                text_coverage = len(affected_words) / total_words * 100.0
            else:
                text_coverage = 0.0

            # Image/visual findings have no reliable word denominator.
            image_area = 0.0
            document_area = sum(page_areas.values())
            for page_no, rects in area_rects.items():
                if not rects:
                    continue
                page_union = _union_area(rects)
                page_area = page_areas.get(page_no, 0.0)
                if page_area > 0:
                    image_area += page_union

            area_coverage = (
                image_area / document_area * 100.0
                if document_area > 0 else 0.0
            )

            categories = {f["category"] for f in visual_flags}
            has_text_findings = bool(categories & {"text", "font"})
            has_image_findings = bool(categories & {"image", "object", "annotation"})

            if has_text_findings and text_coverage > 0:
                # Text coverage is the primary value because the user is
                # interested in how much actual document data is affected.
                # A small visual box around one value therefore does not become
                # an artificially tiny page-area percentage.
                value = text_coverage
                if has_image_findings and area_coverage > 0:
                    value = max(value, area_coverage)
            else:
                value = area_coverage

            return max(0.0, min(100.0, value))

        except Exception:
            # Fall through to the generic image-coordinate implementation.
            pass

    # --------------------------------------------------------
    # Raster image fallback.
    # --------------------------------------------------------
    sizes = _visual_page_sizes(document_path, result)
    if not sizes:
        return 0.0

    page_rects = {
        index: (0.0, 0.0, size[0], size[1])
        for index, size in enumerate(sizes, start=1)
    }
    by_page = {}

    for flag in visual_flags:
        page_no = flag["page"]
        if page_no not in page_rects:
            continue

        page_w = page_rects[page_no][2]
        page_h = page_rects[page_no][3]
        x, y, w, h = flag["bbox"]
        x0 = max(0.0, min(page_w, x))
        y0 = max(0.0, min(page_h, y))
        x1 = max(0.0, min(page_w, x + abs(w)))
        y1 = max(0.0, min(page_h, y + abs(h)))

        if x1 > x0 and y1 > y0:
            by_page.setdefault(page_no, []).append((x0, y0, x1, y1))

    if not by_page:
        return 0.0

    total_document_area = sum(w * h for w, h in sizes)
    suspicious_area = sum(_union_area(rects) for rects in by_page.values())

    if total_document_area <= 0:
        return 0.0

    return max(0.0, min(100.0, suspicious_area / total_document_area * 100.0))

def _finding_category(flag):
    text = " ".join([
        str(flag.get("field", "")),
        str(flag.get("check", "")),
        str(flag.get("reason", "")),
    ]).lower()

    if "metadata" in text:
        return "metadata"
    if "structure" in text or "eof" in text or "increment" in text:
        return "structure"
    if "font" in text or "typeface" in text:
        return "font"
    if "image" in text or "pixel" in text or "visual" in text:
        return "image"
    if "text" in text or "word" in text or "content" in text:
        return "text"
    if "annotation" in text or "comment" in text:
        return "annotation"
    if "object" in text or "stream" in text:
        return "object"
    return "other"


def calculate_risk_score(flags, tampered_percentage=0.0):
    """
    Weighted evidence score, intentionally separate from Tampered Content %.

    Components:
      - 45% severity
      - 25% confidence proxy
      - 20% evidence diversity
      - 10% document impact

    The detector currently does not expose a per-finding confidence field,
    so confidence is derived conservatively from severity and check quality.
    If a future detector result contains `confidence`, it is used directly.
    """
    if not flags:
        return 0.0, "MINIMAL RISK"

    severity_base = {
        "informational": 5.0,
        "info": 5.0,
        "low": 15.0,
        "medium": 30.0,
        "high": 50.0,
        "critical": 75.0,
    }

    severity_values = []
    confidence_values = []
    categories = set()

    for flag in flags:
        severity = str(flag.get("severity", "medium")).strip().lower()
        base = severity_base.get(severity, 30.0)
        severity_values.append(base)
        categories.add(_finding_category(flag))

        explicit = flag.get("confidence")
        if explicit is not None:
            c = _number(explicit)
            if c is not None:
                if c > 1:
                    c /= 100.0
                confidence_values.append(max(0.0, min(1.0, c)))
                continue

        # Conservative proxy until the detector supplies confidence.
        proxy = {
            "informational": 0.70,
            "info": 0.70,
            "low": 0.72,
            "medium": 0.80,
            "high": 0.88,
            "critical": 0.94,
        }.get(severity, 0.78)

        check_text = str(flag.get("check", "")).lower()
        reason_text = str(flag.get("reason", "")).lower()
        if check_text and reason_text:
            proxy += 0.02
        confidence_values.append(min(0.98, proxy))

    # Severity score: strongest finding + diminishing contribution from
    # additional findings, preventing 16 related font findings from becoming
    # 16x the risk.
    ordered = sorted(severity_values, reverse=True)
    severity_points = 0.0
    diminishing = [1.0, 0.40, 0.20, 0.10, 0.05]
    for i, value in enumerate(ordered[:5]):
        severity_points += value * diminishing[i]
    severity_score = min(100.0, severity_points)

    confidence_score = (sum(confidence_values) / len(confidence_values)) * 100.0

    # Five or more distinct evidence categories saturates this component.
    diversity_score = min(100.0, (len(categories) / 5.0) * 100.0)

    # Impact is deliberately low-weighted: a large visual affected area raises
    # risk, but area alone does not prove malicious tampering.
    impact_score = min(100.0, float(tampered_percentage) * 2.0)

    risk = (
        severity_score * 0.45
        + confidence_score * 0.25
        + diversity_score * 0.20
        + impact_score * 0.10
    )
    risk = max(0.0, min(100.0, risk))

    if risk >= 80:
        label = "CRITICAL RISK"
    elif risk >= 60:
        label = "HIGH RISK"
    elif risk >= 40:
        label = "MODERATE RISK"
    elif risk >= 20:
        label = "LOW RISK"
    else:
        label = "MINIMAL RISK"

    return risk, label


def risk_badge_class(label):
    label = str(label).upper()
    if "CRITICAL" in label or "HIGH" in label:
        return "risk-high"
    if "MODERATE" in label:
        return "risk-medium"
    if "LOW" in label:
        return "risk-low"
    return "risk-safe"


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>AdroShield-AI</title>

<style>

:root {

    --blue: #2878ff;
    --blue-2: #3865ff;
    --purple: #8b4dff;
    --pink: #ef2180;
    --orange: #ff9b42;

    --text: #0b1020;
    --muted: #68738b;

    --border: rgba(90, 108, 150, 0.16);

    --white: #ffffff;

    --shadow:
        0 20px 55px rgba(48, 70, 130, 0.10);

}


/* ==========================================================
   RESET
   ========================================================== */

* {
    box-sizing: border-box;
}


html {
    scroll-behavior: smooth;
}


body {

    margin: 0;

    color: var(--text);

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Roboto,
        Arial,
        sans-serif;

    background:

        radial-gradient(
            ellipse 600px 420px at -5% 25%,
            rgba(64, 137, 255, 0.18),
            transparent 70%
        ),

        radial-gradient(
            ellipse 600px 420px at 105% 22%,
            rgba(255, 87, 170, 0.16),
            transparent 70%
        ),

        radial-gradient(
            ellipse 500px 300px at 50% 48%,
            rgba(128, 99, 255, 0.08),
            transparent 70%
        ),

        #fbfcff;

    min-height: 100vh;

}


/* ==========================================================
   BACKGROUND DECORATION
   ========================================================== */

body::before {

    content: "";

    position: fixed;

    width: 430px;
    height: 430px;

    left: -180px;
    top: 130px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(54, 137, 255, 0.15),
            transparent 68%
        );

    pointer-events: none;

    z-index: -1;

}


body::after {

    content: "";

    position: fixed;

    width: 430px;
    height: 430px;

    right: -170px;
    top: 110px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(255, 69, 146, 0.13),
            transparent 68%
        );

    pointer-events: none;

    z-index: -1;

}


/* ==========================================================
   MAIN CONTAINER
   ========================================================== */

.container {

    width: min(
        1135px,
        calc(100% - 36px)
    );

    margin: auto;

}


/* ==========================================================
   HEADER
   ========================================================== */

header {

    height: 89px;

    border-bottom:
        1px solid rgba(100, 110, 140, 0.12);

    display: flex;

    align-items: center;

    justify-content: space-between;

}


.brand {

    display: flex;

    align-items: center;

    gap: 14px;

}


.brand-logo {

    width: 49px;
    height: 49px;

    border-radius: 14px;

    display: flex;

    align-items: center;

    justify-content: center;

    color: white;

    background:
        linear-gradient(
            145deg,
            #183a9b,
            #132d82
        );

    box-shadow:
        0 9px 23px
        rgba(28, 69, 171, 0.22);

}


.brand-logo svg {

    width: 25px;
    height: 25px;

}


.brand-name {

    font-size: 20px;

    line-height: 1;

    font-weight: 760;

    letter-spacing: -0.55px;

}


.brand-subtitle {

    color: #77839a;

    font-size: 11px;

    margin-top: 5px;

}


.status {

    display: flex;

    align-items: center;

    gap: 10px;

    padding:
        11px 17px;

    border:
        1px solid rgba(70, 83, 120, 0.13);

    border-radius: 999px;

    background:
        rgba(255,255,255,0.72);

    box-shadow:
        0 5px 18px
        rgba(40, 50, 90, 0.04);

    color: #58647b;

    font-size: 12px;

    font-weight: 550;

}


.status-dot {

    width: 9px;
    height: 9px;

    border-radius: 50%;

    background: #16b979;

    box-shadow:
        0 0 0 5px
        rgba(22,185,121,0.10);

}


/* ==========================================================
   HERO
   ========================================================== */

.hero {

    position: relative;

    text-align: center;

    padding:
        67px 0
        42px;

}


.hero::before {

    content: "";

    position: absolute;

    width: 170px;
    height: 170px;

    left: -65px;
    top: 35px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(67, 137, 255, 0.12),
            transparent 70%
        );

    pointer-events: none;

}


.hero::after {

    content: "";

    position: absolute;

    width: 160px;
    height: 160px;

    right: -45px;
    top: 20px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(255, 72, 160, 0.10),
            transparent 70%
        );

    pointer-events: none;

}


.ai-pill {

    display: inline-flex;

    align-items: center;

    gap: 9px;

    padding:
        8px 14px;

    border-radius: 999px;

    border:
        1px solid rgba(100, 95, 255, 0.20);

    background:
        linear-gradient(
            100deg,
            rgba(239,245,255,0.95),
            rgba(255,240,252,0.95)
        );

    color: #59657d;

    font-size: 12px;

    font-weight: 620;

    box-shadow:
        0 5px 20px
        rgba(78, 87, 160, 0.05);

}


.ai-pill-icon {

    color: #377cff;

    display: flex;

}


.ai-pill-icon svg {

    width: 17px;
    height: 17px;

}


.hero h1 {

    max-width: 760px;

    margin:
        22px auto
        13px;

    font-size:
        clamp(
            43px,
            5.5vw,
            61px
        );

    line-height: 1.02;

    letter-spacing: -3px;

    font-weight: 780;

}


.hero-gradient {

    display: block;

    background:
        linear-gradient(
            95deg,
            #2677ff 10%,
            #6949ff 45%,
            #d32fbb 76%,
            #f04471 100%
        );

    -webkit-background-clip: text;

    background-clip: text;

    color: transparent;

}


.hero p {

    max-width: 710px;

    margin: auto;

    color: #687590;

    font-size: 15px;

    line-height: 1.7;

}


/* ==========================================================
   DETECTOR GRID
   ========================================================== */

.detector-grid {

    display: grid;

    grid-template-columns:
        repeat(2, minmax(0, 1fr));

    gap: 22px;

    padding-bottom: 48px;

}


/* ==========================================================
   DETECTOR CARD
   ========================================================== */

.detector {

    position: relative;

    overflow: hidden;

    padding: 25px 27px 27px;

    border-radius: 22px;

    border:
        1px solid var(--border);

    background:
        rgba(255,255,255,0.82);

    backdrop-filter: blur(12px);

    box-shadow: var(--shadow);

    transition:
        transform .2s ease,
        box-shadow .2s ease;

}


.detector:hover {

    transform: translateY(-3px);

    box-shadow:
        0 25px 65px
        rgba(50, 72, 135, 0.14);

}


.detector.blue {

    border-color:
        rgba(61, 129, 255, 0.20);

}


.detector.pink {

    border-color:
        rgba(239, 52, 135, 0.20);

}


/* decorative card glow */

.detector.blue::after {

    content: "";

    position: absolute;

    width: 220px;
    height: 220px;

    right: -105px;
    bottom: -125px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(52, 123, 255, 0.16),
            transparent 70%
        );

    pointer-events: none;

}


.detector.pink::after {

    content: "";

    position: absolute;

    width: 220px;
    height: 220px;

    right: -100px;
    bottom: -125px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(239, 44, 133, 0.13),
            transparent 70%
        );

    pointer-events: none;

}


/* ==========================================================
   CARD HEADER
   ========================================================== */

.card-header {

    display: flex;

    align-items: flex-start;

    justify-content: space-between;

}


.detector-icon {

    width: 50px;
    height: 50px;

    border-radius: 14px;

    display: flex;

    align-items: center;

    justify-content: center;

}


.detector-icon svg {

    width: 27px;
    height: 27px;

}


.blue .detector-icon {

    background:
        linear-gradient(
            145deg,
            #edf3ff,
            #e4edff
        );

    color: #2878ff;

}


.pink .detector-icon {

    background:
        linear-gradient(
            145deg,
            #fff0f7,
            #ffe5f1
        );

    color: #ed237f;

}


.detector-number {

    padding-top: 8px;

    color: #69758d;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: .15px;

}


/* ==========================================================
   CARD CONTENT
   ========================================================== */

.detector h2 {

    margin:
        22px 0
        7px;

    font-size: 20px;

    line-height: 1.25;

    letter-spacing: -.55px;

}


.detector-description {

    min-height: 39px;

    color: #687590;

    font-size: 12px;

    line-height: 1.65;

}


/* ==========================================================
   UPLOAD
   ========================================================== */

.upload {

    margin-top: 18px;

    min-height: 112px;

    border-radius: 15px;

    border:
        1px dashed;

    display: flex;

    align-items: center;

    padding: 15px 20px;

    gap: 16px;

}


.blue .upload {

    border-color:
        rgba(47, 119, 255, 0.34);

    background:
        linear-gradient(
            135deg,
            rgba(244,248,255,.9),
            rgba(250,252,255,.75)
        );

}


.pink .upload {

    border-color:
        rgba(239, 48, 135, 0.29);

    background:
        linear-gradient(
            135deg,
            rgba(255,247,251,.95),
            rgba(255,251,253,.8)
        );

}


.upload-icon {

    flex-shrink: 0;

    width: 51px;
    height: 51px;

    border-radius: 13px;

    background:
        rgba(255,255,255,.9);

    border:
        1px solid rgba(100,110,140,.12);

    display: flex;

    align-items: center;

    justify-content: center;

}


.upload-icon svg {

    width: 25px;
    height: 25px;

}


.blue .upload-icon {

    color: #2878ff;

}


.pink .upload-icon {

    color: #ed237f;

}


.upload-info {

    flex: 1;

    min-width: 0;

}


.upload-title {

    color: #151b2a;

    font-size: 12px;

    font-weight: 700;

}


.upload-note {

    color: #78849b;

    font-size: 10px;

    margin-top: 4px;

}


input[type="file"] {

    display: block;

    width: 100%;

    margin-top: 10px;

    color: #768198;

    font-size: 10px;

}


input[type="file"]::file-selector-button {

    border: 1px solid;

    border-radius: 9px;

    background: white;

    padding:
        7px 12px;

    margin-right: 8px;

    cursor: pointer;

    font-size: 11px;

    font-weight: 700;

}


.blue input[type="file"]::file-selector-button {

    color: #246cff;

    border-color:
        rgba(47,119,255,.32);

}


.pink input[type="file"]::file-selector-button {

    color: #ed237f;

    border-color:
        rgba(239,48,135,.28);

}


/* ==========================================================
   ANALYZE BUTTON
   ========================================================== */

.analyze {

    position: relative;

    z-index: 2;

    width: 100%;

    border: 0;

    margin-top: 13px;

    padding:
        13px 18px;

    border-radius: 11px;

    color: white;

    font-size: 12px;

    font-weight: 720;

    cursor: pointer;

    display: flex;

    align-items: center;

    justify-content: center;

    gap: 10px;

    transition:
        transform .18s ease,
        box-shadow .18s ease;

}


.analyze svg {

    width: 17px;
    height: 17px;

}


.blue .analyze {

    background:
        linear-gradient(
            100deg,
            #2678ff,
            #3769ff
        );

    box-shadow:
        0 9px 21px
        rgba(45,113,255,.22);

}


.pink .analyze {

    background:
        linear-gradient(
            100deg,
            #ed167e,
            #ff4276 58%,
            #ff9b3e
        );

    box-shadow:
        0 9px 21px
        rgba(239,50,124,.20);

}


.analyze:hover {

    transform: translateY(-1px);

}


.analyze:active {

    transform: translateY(0);

}


.analyze:disabled {

    opacity: .65;

    cursor: wait;

    transform: none;

}


/* ==========================================================
   RESULTS
   ========================================================== */

.results {

    padding:
        15px 0
        45px;

}


.results-heading {

    margin-bottom: 15px;

}


.results-heading h2 {

    margin: 0;

    font-size: 23px;

    letter-spacing: -.6px;

}


.results-count {

    margin-top: 5px;

    color: #8792a8;

    font-size: 12px;

}


.result-card {

    overflow: hidden;

    margin-bottom: 13px;

    background:
        rgba(255,255,255,.88);

    border:
        1px solid var(--border);

    border-radius: 17px;

    box-shadow:
        0 8px 27px
        rgba(40,58,100,.055);

}


.result-header {

    min-height: 61px;

    padding:
        13px 18px;

    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 15px;

    border-bottom:
        1px solid rgba(100,110,140,.10);

}


.result-file {

    display: flex;

    align-items: center;

    gap: 10px;

    min-width: 0;

}


.result-preview {
    width: 54px;
    height: 54px;
    flex: 0 0 54px;
    overflow: hidden;
    border-radius: 13px;
    border: 1px solid rgba(90,108,150,.15);
    background: linear-gradient(135deg,#edf3ff,#f8efff);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 5px 15px rgba(40,60,100,.08);
}

.result-preview img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}

.result-file-icon {

    width: 32px;
    height: 32px;

    flex-shrink: 0;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 9px;

    background: #f0f3f8;

    color: #69758a;

}


.result-file-icon svg {

    width: 16px;
    height: 16px;

}


.filename {

    max-width: 650px;

    overflow: hidden;

    white-space: nowrap;

    text-overflow: ellipsis;

    font-size: 14px;

    font-weight: 760;

    letter-spacing: -.15px;

}


.filetype {

    margin-top: 4px;

    color: #8994a8;

    font-size: 11px;

}


.badge {

    display: inline-flex;

    align-items: center;

    gap: 6px;

    padding:
        6px 10px;

    border-radius: 999px;

    font-size: 9px;

    font-weight: 750;

    white-space: nowrap;

}


.badge-dot {

    width: 5px;
    height: 5px;

    border-radius: 50%;

}


.badge-danger {

    background: #fff0f2;

    color: #e52e45;

}


.badge-danger .badge-dot {

    background: #ed3048;

}


.badge-success {

    background: #edfbf4;

    color: #12905d;

}


.badge-success .badge-dot {

    background: #18af70;

}


/* ==========================================================
   RESULT METRICS
   ========================================================== */

.result-body {

    padding: 18px 20px 21px;

}


.metrics {

    display: grid;

    grid-template-columns:
        repeat(4, minmax(0, 1fr));

    gap: 8px;

}


.metric {

    padding:
        15px 16px;

    border:
        1px solid rgba(100,110,140,.10);

    border-radius: 11px;

    background:
        rgba(251,252,254,.9);

}


.metric-label {

    color: #8490a6;

    font-size: 10px;

    font-weight: 700;

    margin-bottom: 7px;

}


.metric-value {

    color: #101521;

    font-size: 20px;

    font-weight: 760;

    letter-spacing: -.4px;

}


/* ==========================================================
   DOCUMENT FLAGS
   ========================================================== */

.flags {

    margin-top: 20px;

    padding-top: 18px;

    border-top:
        1px solid rgba(100,110,140,.10);

}


.section-title {

    margin-bottom: 12px;

    font-size: 14px;

    font-weight: 760;

    letter-spacing: -.15px;

}


.flag {

    padding:
        15px 16px;

    margin-top: 10px;

    border:
        1px solid rgba(100,110,140,.10);

    border-radius: 10px;

    background: white;

}


.flag-head {

    display: flex;

    align-items: center;

    gap: 10px;

    margin-bottom: 8px;

}


.severity {

    padding:
        5px 8px;

    border-radius: 7px;

    background: #f1f3f6;

    color: #626d81;

    font-size: 10px;

    font-weight: 800;

    text-transform: uppercase;

}


.flag-field {

    font-size: 13px;

    font-weight: 760;

    color: #172033;

}


.flag-info {

    color: #68758b;

    font-size: 12px;

    line-height: 1.75;

}



/* ==========================================================
   DOCUMENT EVIDENCE WORKSPACE
   ========================================================== */

.document-result-card {
    overflow: hidden;
}

.document-result-body {
    padding: 20px 21px 22px;
}

.document-metrics {
    grid-template-columns: repeat(2, minmax(0, 295px));
    gap: 10px;
    margin-bottom: 18px;
}

.document-metrics .metric {
    min-height: 82px;
    padding: 17px 18px;
    border-radius: 13px;
}

.document-metrics .metric-label {
    font-size: 10px;
    letter-spacing: .15px;
}

.document-metrics .metric-value {
    font-size: 22px;
}

.document-evidence-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(390px, .92fr);
    gap: 16px;
    min-height: 570px;
    border-top: 1px solid rgba(100,110,140,.10);
    padding-top: 18px;
}

.findings-panel,
.annotation-panel {
    min-width: 0;
    border: 1px solid rgba(90,108,150,.13);
    border-radius: 16px;
    background: rgba(250,251,254,.76);
    overflow: hidden;
}

.findings-panel {
    display: flex;
    flex-direction: column;
    min-height: 570px;
}

.findings-panel-header,
.annotation-panel-header {
    flex: 0 0 auto;
    min-height: 70px;
    padding: 16px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    background: rgba(255,255,255,.9);
    border-bottom: 1px solid rgba(100,110,140,.10);
}

.findings-panel-header .section-title,
.annotation-panel-header .section-title {
    margin: 0;
    font-size: 15px;
    letter-spacing: -.25px;
}

.findings-subtitle {
    margin-top: 4px;
    color: #8a95aa;
    font-size: 11px;
    line-height: 1.45;
}

.findings-count,
.evidence-count {
    min-width: 30px;
    height: 30px;
    padding: 0 9px;
    border-radius: 9px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #eef3ff;
    color: #356ce8;
    font-size: 11px;
    font-weight: 800;
}

.findings-scroll {
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    padding: 12px;
    scrollbar-width: thin;
    scrollbar-color: #cbd5e7 transparent;
}

.findings-scroll::-webkit-scrollbar {
    width: 7px;
}

.findings-scroll::-webkit-scrollbar-track {
    background: transparent;
}

.findings-scroll::-webkit-scrollbar-thumb {
    background: #cbd5e7;
    border-radius: 999px;
}

.flag {
    padding: 15px 16px;
    margin: 0 0 9px;
    border: 1px solid rgba(100,110,140,.11);
    border-radius: 12px;
    background: #fff;
    box-shadow: 0 4px 12px rgba(45,60,100,.035);
}

.flag:last-child {
    margin-bottom: 0;
}

.flag-head {
    margin-bottom: 10px;
}

.severity {
    padding: 5px 8px;
    border-radius: 7px;
    background: #f1f3f6;
    color: #626d81;
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
}

.flag-field {
    font-size: 13px;
    font-weight: 760;
    color: #172033;
}

.flag-info {
    color: #69758b;
    font-size: 12px;
    line-height: 1.55;
}

.flag-row,
.flag-reason {
    display: grid;
    grid-template-columns: 92px minmax(0,1fr);
    gap: 7px;
    margin-top: 5px;
}

.flag-row:first-child {
    margin-top: 0;
}

.flag-info strong {
    color: #5c6880;
    font-weight: 750;
}

.flag-reason {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid rgba(100,110,140,.08);
}

.bbox-row {
    color: #8290a7;
    font-size: 11px;
}

.annotation-panel {
    display: flex;
    flex-direction: column;
    min-height: 570px;
}

.annotation-viewer {
    flex: 1 1 auto;
    min-height: 0;
    padding: 15px;
    overflow: auto;
    background:
        radial-gradient(circle at 50% 10%, rgba(63,126,255,.06), transparent 48%),
        #f8faff;
    scrollbar-width: thin;
    scrollbar-color: #cbd5e7 transparent;
}

.annotation-viewer::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}

.annotation-viewer::-webkit-scrollbar-thumb {
    background: #cbd5e7;
    border-radius: 999px;
}

.annotation-large {
    position: relative;
    display: block;
    width: 100%;
    margin-bottom: 12px;
    border: 1px solid rgba(75,100,145,.14);
    border-radius: 12px;
    overflow: hidden;
    background: #fff;
    box-shadow: 0 9px 22px rgba(40,60,100,.09);
}

.annotation-large:last-child {
    margin-bottom: 0;
}

.annotation-large img {
    display: block;
    width: 100%;
    height: auto;
    max-height: 760px;
    object-fit: contain;
    background: white;
}

.annotation-overlay {
    position: absolute;
    left: 12px;
    right: 12px;
    bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 10px;
    border-radius: 9px;
    color: #fff;
    background: rgba(13,20,36,.76);
    backdrop-filter: blur(7px);
    font-size: 10px;
    font-weight: 700;
    opacity: 0;
    transform: translateY(4px);
    transition: .18s ease;
}

.annotation-large:hover .annotation-overlay {
    opacity: 1;
    transform: translateY(0);
}

.no-findings,
.no-annotation {
    min-height: 260px;
    padding: 30px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: #66738b;
}

.no-findings-icon,
.no-annotation-icon {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 12px;
    background: #ecfaf4;
    color: #16a66d;
    font-size: 20px;
    font-weight: 800;
}

.no-findings strong,
.no-annotation strong {
    color: #182033;
    font-size: 13px;
}

.no-findings span,
.no-annotation span {
    max-width: 270px;
    margin-top: 5px;
    font-size: 11px;
    line-height: 1.55;
}

.no-annotation-icon {
    background: #eef3ff;
    color: #4779e8;
}

@media (max-width: 900px) {
    .document-evidence-grid {
        grid-template-columns: 1fr;
        min-height: 0;
    }

    .findings-panel,
    .annotation-panel {
        min-height: 500px;
    }

    .findings-scroll {
        max-height: 500px;
    }

    .annotation-panel {
        min-height: 560px;
    }
}

@media (max-width: 560px) {
    .document-result-body {
        padding: 14px;
    }

    .document-metrics {
        grid-template-columns: repeat(2, minmax(0,1fr));
    }

    .document-evidence-grid {
        gap: 12px;
    }

    .flag-row,
    .flag-reason {
        grid-template-columns: 1fr;
        gap: 2px;
    }
}

/* ==========================================================
   DOCUMENT RESULT DETAIL
   ========================================================== */

.document-result-card .result-header {
    min-height: 76px;
    padding: 16px 20px;
}

.document-result-card .result-preview,
.document-result-card .result-file-icon {
    width: 46px;
    height: 46px;
    flex-basis: 46px;
    border-radius: 12px;
}

.document-result-card .result-file-icon svg {
    width: 20px;
    height: 20px;
}

.document-result-card .badge {
    padding: 8px 12px;
    font-size: 10px;
}

.document-result-card .badge-dot {
    width: 6px;
    height: 6px;
}

.document-result-card .metrics {
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
}

.document-result-card .metric-value.risk-high { color: #d92d45; }
.document-result-card .metric-value.risk-medium { color: #c56b13; }
.document-result-card .metric-value.risk-low { color: #9a7a12; }
.document-result-card .metric-value.risk-safe { color: #12905d; }

.metric-note {
    margin-top: 5px;
    color: #8a95aa;
    font-size: 10px;
    line-height: 1.35;
}

.risk-label {
    display: inline-flex;
    margin-top: 5px;
    font-size: 10px;
    font-weight: 750;
}

.document-result-card .metric {
    min-height: 72px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border-radius: 13px;
}

.document-result-card .flag {
    box-shadow: 0 3px 12px rgba(40,58,100,.025);
}

@media (max-width: 760px) {
    .document-result-card .metrics {
        grid-template-columns: 1fr;
    }

    .document-result-card .result-header {
        align-items: flex-start;
    }

    .document-result-card .badge {
        flex-shrink: 0;
    }
}


/* ==========================================================
   ANNOTATIONS
   ========================================================== */

.annotations {

    margin-top: 20px;

    padding-top: 18px;

    border-top:
        1px solid rgba(100,110,140,.10);

}


.annotation-title {

    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-bottom: 9px;

}


.annotation-title strong {

    font-size: 14px;

}


.annotation-title span {

    color: #8b96a9;

    font-size: 11px;

}


.annotation-grid {

    display: grid;

    grid-template-columns:
        repeat(3, minmax(0, 1fr));

    gap: 8px;

}


.annotation {

    display: block;

    overflow: hidden;

    border:
        1px solid rgba(100,110,140,.13);

    border-radius: 10px;

    background: #f5f7fa;

}


.annotation img {

    width: 100%;

    display: block;

}


/* ==========================================================
   ERROR
   ========================================================== */

.error {

    margin-bottom: 8px;

    padding:
        12px 14px;

    border:
        1px solid #ffd5dc;

    border-radius: 10px;

    background: #fff7f8;

    color: #b52f45;

    font-size: 11px;

}


/* ==========================================================
   FOOTER
   ========================================================== */

footer {

    padding:
        22px 0
        30px;

    border-top:
        1px solid rgba(100,110,140,.10);

    color: #8b95a8;

    font-size: 10px;

    display: flex;

    justify-content: space-between;

}


/* ==========================================================
   RESPONSIVE
   ========================================================== */

@media (max-width: 850px) {

    .detector-grid {

        grid-template-columns: 1fr;

    }

}


@media (max-width: 600px) {

    .container {

        width:
            calc(100% - 24px);

    }


    header {

        height: 75px;

    }


    .status {

        display: none;

    }


    .brand-logo {

        width: 43px;
        height: 43px;

    }


    .brand-name {

        font-size: 17px;

    }


    .hero {

        padding:
            45px 0
            30px;

    }


    .hero h1 {

        font-size: 39px;

        letter-spacing: -2px;

    }


    .hero p {

        font-size: 13px;

    }


    .detector {

        padding: 20px;

    }


    .upload {

        align-items: flex-start;

    }


    .metrics {

        grid-template-columns:
            repeat(2, 1fr);

    }


    .annotation-grid {

        grid-template-columns: 1fr;

    }


    .result-header {

        align-items: flex-start;

        flex-direction: column;

    }


    footer {

        flex-direction: column;

        gap: 7px;

    }

}


/* Tampered-content percentage is intentionally not displayed in document reports. */
.document-metrics .tampered-content-metric,
.document-metrics .tampered-metric,
.document-metrics [data-metric="tampered-content"],
.document-metrics [data-metric="tampered_percentage"] {
    display: none !important;
}
</style>

</head>


<body>


<div class="container">


<!-- ========================================================
     HEADER
     ======================================================== -->

<header>

    <div class="brand">

        <div class="brand-logo">

            <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.8"
            >

                <path
                    d="M12 3l7 3v5.5c0 4.4-2.7 7.8-7 9.5-4.3-1.7-7-5.1-7-9.5V6l7-3z"
                />

                <path
                    d="M8.8 12.1l2.1 2.1 4.4-4.5"
                />

            </svg>

        </div>


        <div>

            <div class="brand-name">
                AdroShield-AI
            </div>

            <div class="brand-subtitle">
                Media authenticity analysis
            </div>

        </div>

    </div>


    <div class="status">

        <span class="status-dot"></span>

        Local analysis ready

    </div>

</header>


<!-- ========================================================
     HERO
     ======================================================== -->

<section class="hero">


    <div class="ai-pill">

        <span class="ai-pill-icon">

            <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.8"
            >

                <path
                    d="M12 3l1.4 5.6L19 10l-5.6 1.4L12 17l-1.4-5.6L5 10l5.6-1.4L12 3z"
                />

                <path
                    d="M19 16l.6 2.4L22 19l-2.4.6L19 22l-.6-2.4L16 19l2.4-.6L19 16z"
                />

            </svg>

        </span>

        AI-powered integrity analysis

    </div>


    <h1>

        Verify what you see.

        <span class="hero-gradient">
            Detect what changed.
        </span>

    </h1>


    <p>

        Analyze images for AI-generated content and inspect
        documents for signs of tampering<br class="desktop">

        from one clean, private workspace.

    </p>

</section>


<!-- ========================================================
     DETECTOR CARDS
     ======================================================== -->

<div class="detector-grid">


<!-- ======================================================
     DEEPFAKE
     ====================================================== -->

<section class="detector blue">


    <div class="card-header">


        <div class="detector-icon">

            <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.8"
            >

                <rect
                    x="3"
                    y="4"
                    width="18"
                    height="16"
                    rx="3"
                />

                <circle
                    cx="8.5"
                    cy="10"
                    r="1.3"
                />

                <path
                    d="M21 15l-4.3-4.3L10 18"
                />

            </svg>

        </div>


        <div class="detector-number">
            DETECTOR 01
        </div>


    </div>


    <h2>
        Deepfake & AI Image Detection
    </h2>


    <div class="detector-description">

        Check one or multiple images for AI-generated
        or manipulated visual content.

    </div>


    <form
        method="POST"
        action="/analyze/deepfake"
        enctype="multipart/form-data"
        onsubmit="loading(this)"
    >


        <div class="upload">


            <div class="upload-icon">

                <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.8"
                >

                    <path
                        d="M12 16V5"
                    />

                    <path
                        d="M7.5 9.5L12 5l4.5 4.5"
                    />

                    <path
                        d="M5 14v4a1 1 0 001 1h12a1 1 0 001-1v-4"
                    />

                </svg>

            </div>


            <div class="upload-info">

                <div class="upload-title">
                    Select image files
                </div>

                <div class="upload-note">
                    Multiple files supported · JPG, PNG, WEBP, BMP, TIFF
                </div>


                <input
                    type="file"
                    name="files"
                    multiple
                    accept="image/*"
                    required
                >

            </div>


        </div>


        <button
            class="analyze"
            type="submit"
        >

            <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.8"
            >

                <path
                    d="M12 3l1.4 5.6L19 10l-5.6 1.4L12 17l-1.4-5.6L5 10l5.6-1.4L12 3z"
                />

            </svg>


            <span>
                Analyze Images
            </span>


            <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
            >

                <path
                    d="M5 12h14"
                />

                <path
                    d="M13 6l6 6-6 6"
                />

            </svg>

        </button>


    </form>


</section>


<!-- ======================================================
     DOCUMENT
     ====================================================== -->

<section class="detector pink">


    <div class="card-header">


        <div class="detector-icon">

            <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.8"
            >

                <path
                    d="M6 3h8l4 4v14H6z"
                />

                <path
                    d="M14 3v5h5"
                />

                <path
                    d="M9 13h6"
                />

                <path
                    d="M9 17h5"
                />

            </svg>

        </div>


        <div class="detector-number">
            DETECTOR 02
        </div>


    </div>


    <h2>
        Document Tampering Detection
    </h2>


    <div class="detector-description">

        Inspect documents and images for suspicious
        modifications and review annotated evidence.

    </div>


    <form
        method="POST"
        action="/analyze/document"
        enctype="multipart/form-data"
        onsubmit="loading(this)"
    >


        <div class="upload">


            <div class="upload-icon">

                <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.8"
                >

                    <path
                        d="M12 16V5"
                    />

                    <path
                        d="M7.5 9.5L12 5l4.5 4.5"
                    />

                    <path
                        d="M5 14v4a1 1 0 001 1h12a1 1 0 001-1v-4"
                    />

                </svg>

            </div>


            <div class="upload-info">

                <div class="upload-title">
                    Select document files
                </div>

                <div class="upload-note">
                    Multiple files supported · PDF, JPG, PNG, TIFF
                </div>


                <input
                    type="file"
                    name="files"
                    multiple
                    accept=".pdf,.jpg,.jpeg,.png,.bmp,.tiff,.tif"
                    required
                >

            </div>


        </div>


        <button
            class="analyze"
            type="submit"
        >

            <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.8"
            >

                <path
                    d="M12 3l1.4 5.6L19 10l-5.6 1.4L12 17l-1.4-5.6L5 10l5.6-1.4L12 3z"
                />

            </svg>


            <span>
                Analyze Documents
            </span>


            <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
            >

                <path
                    d="M5 12h14"
                />

                <path
                    d="M13 6l6 6-6 6"
                />

            </svg>

        </button>


    </form>


</section>


</div>


<!-- ========================================================
     ERRORS
     ======================================================== -->

{% if errors %}

<section class="results">


    <div class="results-heading">

        <h2>
            Processing issue
        </h2>

        <div class="results-count">
            Some files could not be analyzed.
        </div>

    </div>


    {% for error in errors %}

    <div class="error">
        {{ error }}
    </div>

    {% endfor %}


</section>

{% endif %}


<!-- ========================================================
     DEEPFAKE RESULTS
     ======================================================== -->

{% if deepfake_results %}

<section class="results">


    <div class="results-heading">

        <h2>
            Deepfake Analysis
        </h2>

        <div class="results-count">

            {{ deepfake_results|length }}

            image{% if deepfake_results|length != 1 %}s{% endif %}

            analyzed

        </div>

    </div>


    {% for item in deepfake_results %}

    <article class="result-card{% if item.flags %} document-result-card{% endif %}">


        <div class="result-header">


            <div class="result-file">


                {% if item.preview_url %}
                <div class="result-preview">
                    <img src="{{ item.preview_url }}" alt="{{ item.filename }}">
                </div>
                {% else %}
                <div class="result-file-icon">

                    <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="1.7"
                    >

                        <rect
                            x="3"
                            y="4"
                            width="18"
                            height="16"
                            rx="3"
                        />

                        <circle
                            cx="8.5"
                            cy="10"
                            r="1.3"
                        />

                        <path
                            d="M21 15l-4.3-4.3L10 18"
                        />

                    </svg>

                </div>
                {% endif %}

                <div>

                    <div
                        class="filename"
                        title="{{ item.filename }}"
                    >
                        {{ item.filename }}
                    </div>

                    <div class="filetype">
                        Image integrity analysis
                    </div>

                </div>


            </div>


            {% if item.decision|lower in [
                "real",
                "likely real",
                "authentic",
                "original"
            ] %}

            <div class="badge badge-success">

                <span class="badge-dot"></span>

                {{ item.decision }}

            </div>

            {% else %}

            <div class="badge badge-danger">

                <span class="badge-dot"></span>

                {{ item.decision }}

            </div>

            {% endif %}


        </div>


        <div class="result-body">


            <div class="metrics">


                <div class="metric">

                    <div class="metric-label">
                        AVERAGE AI SCORE
                    </div>

                    <div class="metric-value">
                        {{ item.average_ai_score }}
                    </div>

                </div>


                <div class="metric">

                    <div class="metric-label">
                        MAX AI SCORE
                    </div>

                    <div class="metric-value">
                        {{ item.maximum_ai_score }}
                    </div>

                </div>


                <div class="metric">

                    <div class="metric-label">
                        TOP 10% AVERAGE
                    </div>

                    <div class="metric-value">
                        {{ item.top_10_average }}
                    </div>

                </div>


                <div class="metric">

                    <div class="metric-label">
                        SUSPICIOUS TILES
                    </div>

                    <div class="metric-value">
                        {{ item.suspicious_tile_count }}
                    </div>

                </div>


            </div>


        </div>


    </article>

    {% endfor %}


</section>

{% endif %}


<!-- ========================================================
     DOCUMENT RESULTS
     ======================================================== -->

{% if document_results %}

<section class="results">

    <div class="results-heading">
        <h2>Document Analysis</h2>
        <div class="results-count">
            {{ document_results|length }}
            document{% if document_results|length != 1 %}s{% endif %}
            analyzed
        </div>
    </div>

    {% for item in document_results %}

    <article class="result-card document-result-card">

        <div class="result-header">
            <div class="result-file">
                {% if item.preview_url %}
                <div class="result-preview">
                    <img src="{{ item.preview_url }}" alt="{{ item.filename }}">
                </div>
                {% else %}
                <div class="result-file-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7">
                        <path d="M6 3h8l4 4v14H6z"/>
                        <path d="M14 3v5h5"/>
                        <path d="M9 13h6"/>
                        <path d="M9 17h5"/>
                    </svg>
                </div>
                {% endif %}

                <div>
                    <div class="filename" title="{{ item.filename }}">{{ item.filename }}</div>
                    <div class="filetype">Document integrity analysis</div>
                </div>
            </div>

            {% if item.is_safe %}
            <div class="badge badge-success">
                <span class="badge-dot"></span>
                {{ item.verdict }}
            </div>
            {% else %}
            <div class="badge badge-danger">
                <span class="badge-dot"></span>
                {{ item.verdict }}
            </div>
            {% endif %}
        </div>

        <div class="result-body document-result-body">

            <div class="metrics document-metrics">
                <div class="metric">
                    <div class="metric-label">RISK SCORE</div>
                    <div class="metric-value {{ item.risk_class }}">{{ item.risk_score }}</div>
                    <div class="risk-label {{ item.risk_class }}">{{ item.risk_label }}</div>
                </div>

                <div class="metric">
                    <div class="metric-label">FLAGS</div>
                    <div class="metric-value">{{ item.flag_count }}</div>
                    <div class="metric-note">Detected evidence findings</div>
                </div>
            </div>

            <div class="document-evidence-grid">

                <!-- LEFT: scrollable findings -->
                <div class="findings-panel">
                    <div class="findings-panel-header">
                        <div>
                            <div class="section-title">Suspicious Findings</div>
                            <div class="findings-subtitle">
                                {{ item.flag_count }} finding{% if item.flag_count != 1 %}s{% endif %} detected
                            </div>
                        </div>
                        <div class="findings-count">{{ item.flag_count }}</div>
                    </div>

                    <div class="findings-scroll">
                        {% if item.flags %}
                            {% for flag in item.flags %}
                            <div class="flag">
                                <div class="flag-head">
                                    <span class="severity">{{ flag.severity }}</span>
                                    <span class="flag-field">{{ flag.field }}</span>
                                </div>

                                <div class="flag-info">
                                    <div class="flag-row">
                                        <strong>Page</strong>
                                        <span>{{ flag.page }}</span>
                                    </div>

                                    {% if flag.check %}
                                    <div class="flag-row">
                                        <strong>Check</strong>
                                        <span>{{ flag.check }}</span>
                                    </div>
                                    {% endif %}

                                    <div class="flag-reason">
                                        <strong>Reason</strong>
                                        <span>{{ flag.reason }}</span>
                                    </div>

                                    {% if flag.bbox %}
                                    <div class="flag-row bbox-row">
                                        <strong>Bounding Box</strong>
                                        <span>{{ flag.bbox }}</span>
                                    </div>
                                    {% endif %}
                                </div>
                            </div>
                            {% endfor %}
                        {% else %}
                            <div class="no-findings">
                                <div class="no-findings-icon">✓</div>
                                <strong>No suspicious findings</strong>
                                <span>The document passed the available integrity checks.</span>
                            </div>
                        {% endif %}
                    </div>
                </div>

                <!-- RIGHT: annotated evidence -->
                <div class="annotation-panel">
                    <div class="annotation-panel-header">
                        <div>
                            <div class="section-title">Annotated Evidence</div>
                            <div class="findings-subtitle">Visual locations of detected issues</div>
                        </div>
                        {% if item.annotations %}
                        <span class="evidence-count">{{ item.annotations|length }}</span>
                        {% endif %}
                    </div>

                    {% if item.annotations %}
                    <div class="annotation-viewer">
                        {% for image in item.annotations %}
                        <a href="{{ image }}" target="_blank" class="annotation-large">
                            <img src="{{ image }}" alt="Annotated document evidence">
                            <div class="annotation-overlay">
                                <span>Open full size</span>
                                <span>↗</span>
                            </div>
                        </a>
                        {% endfor %}
                    </div>
                    {% else %}
                    <div class="no-annotation">
                        <div class="no-annotation-icon">⌁</div>
                        <strong>No annotated image available</strong>
                        <span>The detector did not return visual evidence for this file.</span>
                    </div>
                    {% endif %}
                </div>

            </div>
        </div>
    </article>
    {% endfor %}
</section>

{% endif %}


<!-- ========================================================
     FOOTER
     ======================================================== -->

<footer>

    <span>
        AdroShield-AI
    </span>

    <span>
        Local processing · Media integrity analysis
    </span>

</footer>


</div>


<script>

function loading(form) {

    const button =
        form.querySelector(".analyze");

    if (!button) {
        return;
    }

    button.disabled = true;

    const span =
        button.querySelector("span");

    if (span) {
        span.textContent = "Analyzing...";
    }

}


document.addEventListener(
    "DOMContentLoaded",
    function () {

        document
            .querySelectorAll(
                'input[type="file"]'
            )
            .forEach(
                function (input) {

                    input.addEventListener(
                        "change",
                        function () {

                            const files =
                                this.files;

                            if (!files.length) {
                                return;
                            }

                            const title =
                                this
                                .closest("form")
                                ?.querySelector(
                                    ".upload-title"
                                );

                            if (!title) {
                                return;
                            }

                            if (files.length === 1) {

                                title.textContent =
                                    files[0].name;

                            } else {

                                title.textContent =
                                    files.length
                                    + " files selected";

                            }

                        }
                    );

                }
            );

    }
);

</script>


</body>

</html>
"""


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def index():

    return render_template_string(
        HTML,
        deepfake_results=None,
        document_results=None,
        errors=None
    )


# ============================================================
# DEEPFAKE
# ============================================================

@app.route(
    "/analyze/deepfake",
    methods=["POST"]
)
def analyze_deepfake():

    files = request.files.getlist("files")

    results = []
    errors = []

    if not files or all(
        not f.filename
        for f in files
    ):

        errors.append(
            "Please select at least one image."
        )

        return render_template_string(
            HTML,
            deepfake_results=None,
            document_results=None,
            errors=errors
        )


    for file in files:

        if not file.filename:
            continue

        if not allowed_image(file.filename):

            errors.append(
                f"{file.filename}: "
                "unsupported image format."
            )

            continue


        image_path = None

        try:

            image_path = save_upload(file)

            preview_url = create_preview(image_path)

            result = photo_detector.analyze(
                str(image_path)
            )

            result = clean_value(result)


            results.append({

                "filename":
                    file.filename,

                "preview_url":
                    preview_url,

                "decision":
                    result.get(
                        "decision",
                        "Unknown"
                    ),

                "average_ai_score":
                    percentage(
                        result.get(
                            "average_ai_score",
                            0
                        )
                    ),

                "maximum_ai_score":
                    percentage(
                        result.get(
                            "maximum_ai_score",
                            0
                        )
                    ),

                "top_10_average":
                    percentage(
                        result.get(
                            "top_10_average",
                            0
                        )
                    ),

                "suspicious_tile_count":
                    result.get(
                        "suspicious_tile_count",
                        0
                    )

            })


        except Exception as error:

            errors.append(
                f"{file.filename}: {error}"
            )


        finally:

            if image_path and image_path.exists():

                try:
                    image_path.unlink()
                except Exception:
                    pass


    return render_template_string(
        HTML,
        deepfake_results=results,
        document_results=None,
        errors=errors
    )


# ============================================================
# DOCUMENT
# ============================================================

@app.route(
    "/analyze/document",
    methods=["POST"]
)
def analyze_document():

    files = request.files.getlist("files")

    results = []
    errors = []


    if not files or all(
        not f.filename
        for f in files
    ):

        errors.append(
            "Please select at least one document."
        )

        return render_template_string(
            HTML,
            deepfake_results=None,
            document_results=None,
            errors=errors
        )


    job_id = uuid.uuid4().hex

    output_dir = OUTPUT_DIR / job_id

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    for file in files:

        if not file.filename:
            continue


        if not allowed_document(file.filename):

            errors.append(
                f"{file.filename}: "
                "unsupported document format."
            )

            continue


        document_path = None


        try:

            document_path = save_upload(file)

            preview_url = create_preview(document_path) if allowed_image(file.filename) else None

            result = document_detector.analyze(
                str(document_path),
                save_annotation=True,
                save_report=True,
                print_output=False
            )


            result = clean_value(result)


            flags = normalize_flags(result)

            # Tampered Content % is calculated from the union of visual
            # bounding boxes. Metadata/structure-only findings do not count.
            tampered_pct = calculate_tampered_percentage(
                document_path,
                flags,
                result
            )

            # Risk is a separate weighted evidence score. It is not the
            # percentage of the document that was visually affected.
            calculated_risk, risk_label = calculate_risk_score(
                flags,
                tampered_pct
            )

            annotations = copy_annotations(
                result,
                output_dir
            )


            results.append({

                "filename":
                    file.filename,

                "preview_url":
                    preview_url,

                "verdict":
                    result.get(
                        "verdict",
                        "Unknown"
                    ),

                "risk_score":
                    f"{calculated_risk:.0f}/100",

                "risk_label":
                    risk_label,

                "risk_class":
                    risk_badge_class(risk_label),

                "flag_count":
                    len(flags),

                "flags":
                    flags,

                "annotations":
                    annotations,

                "is_safe":
                    calculated_risk < 40 and len(flags) == 0

            })


        except Exception as error:

            errors.append(
                f"{file.filename}: {error}"
            )


        finally:

            if (
                document_path
                and document_path.exists()
            ):

                try:
                    document_path.unlink()
                except Exception:
                    pass


    return render_template_string(
        HTML,
        deepfake_results=None,
        document_results=results,
        errors=errors
    )


# ============================================================
# ANNOTATED IMAGES
# ============================================================

@app.route("/preview/<filename>")
def preview_file(filename):
    return send_from_directory(
        PREVIEW_DIR,
        secure_filename(filename)
    )


@app.route(
    "/output/<job_id>/<filename>"
)
def output_file(job_id, filename):

    return send_from_directory(
        OUTPUT_DIR / secure_filename(job_id),
        secure_filename(filename)
    )


# ============================================================
# ERRORS
# ============================================================

@app.errorhandler(413)
def too_large(error):

    return render_template_string(
        HTML,
        deepfake_results=None,
        document_results=None,
        errors=[
            "Uploaded files exceed the 200 MB limit."
        ]
    ), 413


@app.errorhandler(Exception)
def application_error(error):

    return render_template_string(
        HTML,
        deepfake_results=None,
        document_results=None,
        errors=[
            f"Application error: {error}"
        ]
    ), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("AdroShield-AI")
    print("=" * 70)
    print("Open: http://127.0.0.1:5000")
    print("=" * 70)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
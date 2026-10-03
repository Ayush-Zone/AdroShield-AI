"""
document_tampering.py

Reusable document tampering detection library.

Supports:
    - PDF
    - JPG / JPEG
    - PNG
    - BMP

Detection:
    - PDF metadata anomalies
    - EXIF anomalies
    - PDF font inconsistencies
    - OCR
    - Layout anomalies
    - Error Level Analysis (JPEG)
    - PDF text-layer vs OCR comparison
    - Date validation
    - Number formatting anomalies
    - Arithmetic / invoice total validation

Usage:

    from document_tampering import DocumentTamperingDetector

    detector = DocumentTamperingDetector()

    result = detector.analyze("invoice.pdf")

    print(result["verdict"])
    print(result["risk_score"])

    for flag in result["flags"]:
        print(flag)
"""


import io
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path


import numpy as np

from PIL import (
    ExifTags,
    Image,
    ImageChops,
    ImageDraw,
    ImageFont,
)

import pytesseract
from pytesseract import Output

WIN_TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(WIN_TESS):
    pytesseract.pytesseract.tesseract_cmd = WIN_TESS


# ---------------------------------------------------------
# Optional PyMuPDF
# ---------------------------------------------------------

try:
    import pymupdf as fitz

except ImportError:

    try:
        import fitz

    except ImportError:
        fitz = None


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

PDF_DPI = 200

SEV_WEIGHT = {
    "low": 1,
    "medium": 3,
    "high": 6,
}

EDIT_TOOLS = [
    "photoshop",
    "gimp",
    "canva",
    "illustrator",
    "paint",
    "pixlr",
    "snapseed",
    "picsart",
    "lightroom",
    "foxit phantom",
    "pdfescape",
    "sejda",
    "smallpdf",
    "ilovepdf",
    "pdf-xchange",
    "nitro",
    "acrobat pro",
    "inkscape",
    "affinity",
]


NUM_RE = re.compile(
    r"[$₹€£]?\d[\d,.]*%?"
)

AMOUNT_RE = re.compile(
    r"(?<![\w.])(?:[$₹€£]|Rs\.?\s?)?"
    r"(\d[\d,]*(?:\.\d+)?)(?![\w])"
)

MONTHS = (
    "jan|feb|mar|apr|may|jun|jul|aug|"
    "sep|oct|nov|dec"
)

DATE_PATTERNS = [

    (
        "dmy",
        re.compile(
            r"\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})\b"
        ),
    ),

    (
        "ymd",
        re.compile(
            r"\b(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})\b"
        ),
    ),

    (
        "dMy",
        re.compile(
            rf"\b(\d{{1,2}})\s+"
            rf"({MONTHS})[a-z]*\.?,?\s+"
            rf"(\d{{2,4}})\b",
            re.I,
        ),
    ),
]


KW_RE = re.compile(
    r"grand|total|sub-?total|gst|cgst|sgst|igst|"
    r"vat|tax|discount|net|balance",
    re.I,
)

SKIP_ROW = re.compile(
    r"phone|tel\b|mob|gstin|cin\b|@|pin\b|"
    r"invoice\s*(no|date)|due\s*date|\bdate\b",
    re.I,
)


# =========================================================
# DETECTOR CLASS
# =========================================================

class DocumentTamperingDetector:

    def __init__(
        self,
        pdf_dpi=PDF_DPI,
        tesseract_path=None,
    ):
        """
        Initialize detector.

        Parameters
        ----------
        pdf_dpi:
            DPI used when rendering PDF pages.

        tesseract_path:
            Optional path to tesseract.exe.

            Example:
                r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        """

        self.pdf_dpi = pdf_dpi

        # Every analysis gets its own flags
        self.flags = []

        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = (
                tesseract_path
            )

    # =====================================================
    # FLAG
    # =====================================================

    def flag(
        self,
        field,
        reason,
        severity="medium",
        bbox=None,
        page=1,
        check="rule",
    ):

        self.flags.append(
            {
                "page": page,
                "field": field,
                "reason": reason,
                "severity": severity,
                "check": check,
                "bbox": bbox,
                "marker": (
                    "yellow"
                    if check == "font"
                    else "red"
                ),
            }
        )

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def clean(text):

        return text.strip(
            ".,;:()[]|"
        )

    def is_numeric(self, text):

        return bool(
            NUM_RE.fullmatch(
                self.clean(text)
            )
        )

    def is_money(self, text):

        c = self.clean(text)

        return (
            self.is_numeric(c)
            and (
                "," in c
                or "." in c
            )
            and len(c) >= 4
        )

    @staticmethod
    def to_float(value):

        try:
            return float(
                value.replace(",", "")
            )

        except ValueError:
            return None

    @staticmethod
    def union_box(words):

        x1 = min(
            w["x"]
            for w in words
        )

        y1 = min(
            w["y"]
            for w in words
        )

        x2 = max(
            w["x"] + w["w"]
            for w in words
        )

        y2 = max(
            w["y"] + w["h"]
            for w in words
        )

        return [
            x1,
            y1,
            x2 - x1,
            y2 - y1,
        ]

    @staticmethod
    def robust_z(values):

        v = np.asarray(
            values,
            dtype=float,
        )

        med = np.median(v)

        mad = (
            np.median(
                np.abs(v - med)
            )
            or 1e-6
        )

        return (
            0.6745 * (v - med) / mad,
            med,
        )

    @staticmethod
    def parse_pdf_date(value):

        m = re.match(
            r"D:(\d{4})(\d{2})?(\d{2})?"
            r"(\d{2})?(\d{2})?(\d{2})?",
            value or "",
        )

        if not m:
            return None

        parts = [
            int(g) if g else default
            for g, default
            in zip(
                m.groups(),
                (0, 1, 1, 0, 0, 0),
            )
        ]

        try:
            return datetime(*parts)

        except ValueError:
            return None

    # =====================================================
    # METADATA
    # =====================================================

    def check_pdf_metadata(
        self,
        path,
        doc,
    ):

        meta = doc.metadata or {}

        producer = (
            (meta.get("producer") or "")
            + " "
            + (meta.get("creator") or "")
        )

        for tool in EDIT_TOOLS:

            if tool in producer.lower():

                self.flag(
                    "PDF metadata",
                    (
                        "Producer/Creator mentions "
                        f"an editing tool: "
                        f"'{producer.strip()}'"
                    ),
                    "medium",
                    check="metadata",
                )

                break

        created = self.parse_pdf_date(
            meta.get("creationDate")
        )

        modified = self.parse_pdf_date(
            meta.get("modDate")
        )

        if (
            created
            and modified
            and (
                modified - created
            ).total_seconds()
            > 60
        ):

            self.flag(
                "PDF metadata",
                (
                    f"Modified ({modified}) "
                    f"well after creation "
                    f"({created})"
                ),
                "medium",
                check="metadata",
            )

        if (
            not meta.get("creationDate")
            and not meta.get("producer")
        ):

            self.flag(
                "PDF metadata",
                (
                    "Metadata stripped "
                    "(no creation date "
                    "or producer)"
                ),
                "low",
                check="metadata",
            )

        with open(
            path,
            "rb",
        ) as f:

            raw = f.read()

        eofs = raw.count(
            b"%%EOF"
        )

        if eofs > 1:

            self.flag(
                "PDF structure",
                (
                    f"{eofs} %%EOF markers: "
                    "file was saved incrementally"
                ),
                "medium",
                check="metadata",
            )

        if (
            re.search(
                rb"/Type\s*/Annot",
                raw,
            )
            and
            re.search(
                rb"/Subtype\s*/"
                rb"(Redact|FreeText|Square)",
                raw,
            )
        ):

            self.flag(
                "PDF structure",
                (
                    "Contains redaction/free-text/"
                    "shape annotations that can cover content"
                ),
                "medium",
                check="metadata",
            )

    # =====================================================
    # IMAGE METADATA
    # =====================================================

    def check_image_metadata(
        self,
        image,
    ):

        try:
            exif = image.getexif()

        except Exception:
            exif = None

        if not exif:

            if (
                image.format or ""
            ).upper() == "JPEG":

                self.flag(
                    "EXIF",
                    (
                        "JPEG has no EXIF data "
                        "(stripped or re-exported)"
                    ),
                    "low",
                    check="metadata",
                )

            return

        tags = {
            ExifTags.TAGS.get(
                key,
                key,
            ): value
            for key, value
            in exif.items()
        }

        software = str(
            tags.get(
                "Software",
                "",
            )
        )

        for tool in EDIT_TOOLS:

            if tool in software.lower():

                self.flag(
                    "EXIF Software",
                    (
                        "Image saved by editing "
                        f"software: '{software}'"
                    ),
                    "high",
                    check="metadata",
                )

                break

        else:

            if software:

                self.flag(
                    "EXIF Software",
                    (
                        f"Software tag present: "
                        f"'{software}'"
                    ),
                    "low",
                    check="metadata",
                )

        dt = tags.get("DateTime")
        orig = tags.get("DateTimeOriginal")

        if (
            dt
            and orig
            and dt != orig
        ):

            self.flag(
                "EXIF dates",
                (
                    f"DateTime ({dt}) differs "
                    f"from DateTimeOriginal "
                    f"({orig})"
                ),
                "medium",
                check="metadata",
            )

    # =====================================================
    # PDF FONT CHECK
    # =====================================================

    @staticmethod
    def font_family(name):

        name = re.sub(
            r"^[A-Z]{6}\+",
            "",
            name or "",
        )

        previous = None

        while previous != name:

            previous = name

            name = re.sub(
                r"(?i)[\-_, ]?"
                r"(bold|italic|oblique|black|"
                r"semibold|demi|medium|light|"
                r"regular|roman|mt|ps)$",
                "",
                name,
            )

        return name.lower()

    def check_pdf_fonts(
        self,
        page,
        pno,
        scale,
    ):

        data = page.get_text(
            "dict"
        )

        numeric_spans = []
        all_spans = []

        for block in data["blocks"]:

            for line in block.get(
                "lines",
                [],
            ):

                spans = [
                    s
                    for s in line["spans"]
                    if s["text"].strip()
                ]

                for a, b in zip(
                    spans,
                    spans[1:],
                ):

                    if (
                        a["text"][-1]
                        not in " \t"
                        and
                        b["text"][0]
                        not in " \t"
                        and
                        (
                            self.font_family(
                                a["font"]
                            )
                            !=
                            self.font_family(
                                b["font"]
                            )
                            or
                            abs(
                                a["size"]
                                - b["size"]
                            )
                            > 0.5
                        )
                        and
                        a["text"][-1].isalnum()
                        and
                        b["text"][0].isalnum()
                        and
                        (
                            a["text"][-1].isdigit()
                            or
                            b["text"][0].isdigit()
                        )
                    ):

                        bb = [
                            int(
                                v * scale
                            )
                            for v in (
                                min(
                                    a["bbox"][0],
                                    b["bbox"][0],
                                ),
                                min(
                                    a["bbox"][1],
                                    b["bbox"][1],
                                ),
                            )
                        ]

                        bb += [
                            int(
                                max(
                                    a["bbox"][2],
                                    b["bbox"][2],
                                )
                                * scale
                            )
                            - bb[0],

                            int(
                                max(
                                    a["bbox"][3],
                                    b["bbox"][3],
                                )
                                * scale
                            )
                            - bb[1],
                        ]

                        self.flag(
                            (
                                a["text"][-3:]
                                + b["text"][:3]
                            ),
                            (
                                "One word is rendered "
                                "in two fonts "
                                f"({a['font']} "
                                f"{a['size']:.1f}pt / "
                                f"{b['font']} "
                                f"{b['size']:.1f}pt)"
                            ),
                            "high",
                            bb,
                            pno,
                            "font",
                        )

                for span in spans:

                    all_spans.append(
                        span
                    )

                    if self.is_numeric(
                        span["text"].strip()
                    ):

                        numeric_spans.append(
                            span
                        )

        if len(numeric_spans) >= 5:

            combos = Counter(
                self.font_family(
                    span["font"]
                )
                for span in numeric_spans
            )

            total = sum(
                combos.values()
            )

            for span in numeric_spans:

                key = self.font_family(
                    span["font"]
                )

                if (
                    combos[key]
                    / total
                    <= 0.15
                ):

                    bb = [
                        int(v * scale)
                        for v in span["bbox"]
                    ]

                    bb = [
                        bb[0],
                        bb[1],
                        bb[2] - bb[0],
                        bb[3] - bb[1],
                    ]

                    self.flag(
                        span["text"].strip(),
                        (
                            f"Number uses font "
                            f"family '{span['font']}'; "
                            f"{100 * (1 - combos[key] / total):.0f}% "
                            "of other numbers use "
                            "a different font family"
                        ),
                        "medium",
                        bb,
                        pno,
                        "font",
                    )

        weight = Counter()
        names = {}

        for span in all_spans:

            fam = self.font_family(
                span["font"]
            )

            weight[fam] += sum(
                c.isalnum()
                for c in span["text"]
            )

            names.setdefault(
                fam,
                span["font"],
            )

        total_chars = sum(
            weight.values()
        )

        if (
            total_chars >= 40
            and len(weight) > 1
        ):

            dominant, dominant_count = (
                weight.most_common(1)[0]
            )

            if (
                dominant_count
                / total_chars
                >= 0.5
            ):

                for span in all_spans:

                    text = span[
                        "text"
                    ].strip()

                    fam = self.font_family(
                        span["font"]
                    )

                    if (
                        fam != dominant
                        and
                        weight[fam]
                        / total_chars
                        <= 0.15
                        and
                        sum(
                            c.isalnum()
                            for c in text
                        )
                        >= 2
                        and
                        not self.is_numeric(
                            text
                        )
                    ):

                        bb = [
                            int(v * scale)
                            for v in span["bbox"]
                        ]

                        bb = [
                            bb[0],
                            bb[1],
                            bb[2] - bb[0],
                            bb[3] - bb[1],
                        ]

                        self.flag(
                            text,
                            (
                                f"Text set in font "
                                f"'{span['font']}' while "
                                f"{100 * dominant_count / total_chars:.0f}% "
                                f"of page uses "
                                f"'{dominant}' family"
                            ),
                            "medium",
                            bb,
                            pno,
                            "font",
                        )

        if len(
            page.get_fonts()
        ) > 6:

            self.flag(
                "Fonts",
                (
                    f"{len(page.get_fonts())} "
                    "distinct fonts embedded "
                    "on one page"
                ),
                "low",
                page=pno,
                check="font",
            )

    # =====================================================
    # OCR
    # =====================================================

    def run_ocr(
        self,
        image,
    ):

        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
        )

        words = []

        for i, text in enumerate(
            data["text"]
        ):

            text = text.strip()

            if not text:
                continue

            try:
                confidence = float(
                    data["conf"][i]
                )

            except (
                ValueError,
                TypeError,
            ):
                confidence = -1

            words.append(
                {
                    "text": text,
                    "conf": confidence,
                    "x": int(
                        data["left"][i]
                    ),
                    "y": int(
                        data["top"][i]
                    ),
                    "w": int(
                        data["width"][i]
                    ),
                    "h": int(
                        data["height"][i]
                    ),
                    "key": (
                        data["block_num"][i],
                        data["par_num"][i],
                        data["line_num"][i],
                    ),
                }
            )

        lines = []

        for word in sorted(
            words,
            key=lambda w:
            w["y"] + w["h"] / 2,
        ):

            cy = (
                word["y"]
                + word["h"] / 2
            )

            for group in lines:

                group_cy = np.mean(
                    [
                        v["y"]
                        + v["h"] / 2
                        for v in group
                    ]
                )

                if (
                    abs(
                        cy - group_cy
                    )
                    <
                    0.6
                    * max(
                        word["h"],
                        max(
                            v["h"]
                            for v in group
                        ),
                    )
                ):

                    group.append(
                        word
                    )

                    break

            else:

                lines.append(
                    [word]
                )

        for group in lines:

            group.sort(
                key=lambda w:
                w["x"]
            )

        lines.sort(
            key=lambda group:
            min(
                w["y"]
                for w in group
            )
        )

        return words, lines

    # =====================================================
    # LAYOUT
    # =====================================================

    def check_layout(
        self,
        words,
        pno,
    ):

        numbers = [
            word
            for word in words
            if self.is_money(
                word["text"]
            )
            and word["conf"] >= 0
        ]

        if len(numbers) >= 8:

            z, median = self.robust_z(
                [
                    word["h"]
                    for word in numbers
                ]
            )

            for word, zi in zip(
                numbers,
                z,
            ):

                if (
                    abs(zi) > 4
                    and
                    abs(
                        word["h"]
                        - median
                    )
                    / median
                    > 0.35
                ):

                    self.flag(
                        word["text"],
                        (
                            f"Digit height "
                            f"{word['h']}px vs "
                            f"typical {median:.0f}px "
                            "for numbers on the page"
                        ),
                        "low",
                        [
                            word["x"],
                            word["y"],
                            word["w"],
                            word["h"],
                        ],
                        pno,
                        "layout",
                    )

        for word in numbers:

            if (
                0
                <= word["conf"]
                < 40
            ):

                self.flag(
                    word["text"],
                    (
                        f"Low OCR confidence "
                        f"({word['conf']:.0f}%) "
                        "on numeric field"
                    ),
                    "low",
                    [
                        word["x"],
                        word["y"],
                        word["w"],
                        word["h"],
                    ],
                    pno,
                    "layout",
                )

    # =====================================================
    # ELA
    # =====================================================

    def check_ela(
        self,
        image,
        words,
        quality=90,
    ):

        rgb = image.convert(
            "RGB"
        )

        buffer = io.BytesIO()

        rgb.save(
            buffer,
            "JPEG",
            quality=quality,
        )

        buffer.seek(0)

        recompressed = Image.open(
            buffer
        ).convert("RGB")

        difference = np.asarray(
            ImageChops.difference(
                rgb,
                recompressed,
            ),
            dtype=np.float32,
        ).mean(axis=2)

        scored = []

        for word in words:

            if (
                word["w"] < 4
                or word["h"] < 4
            ):
                continue

            patch = difference[
                word["y"]:
                word["y"] + word["h"],
                word["x"]:
                word["x"] + word["w"],
            ]

            if patch.size:

                scored.append(
                    (
                        word,
                        float(
                            patch.mean()
                        ),
                    )
                )

        if len(scored) < 8:
            return

        z, median = self.robust_z(
            [
                score
                for _, score
                in scored
            ]
        )

        for (
            word,
            score
        ), zi in zip(
            scored,
            z,
        ):

            if (
                zi > 5
                and score
                > 2.5 * median
            ):

                self.flag(
                    word["text"],
                    (
                        "Error-level anomaly: "
                        f"compression error "
                        f"{score:.1f} vs median "
                        f"{median:.1f}"
                    ),
                    "high",
                    [
                        word["x"],
                        word["y"],
                        word["w"],
                        word["h"],
                    ],
                    1,
                    "ela",
                )

    # =====================================================
    # PDF TEXT LAYER VS OCR
    # =====================================================

    def check_text_layer(
        self,
        page,
        words,
        pno,
        scale,
    ):

        layer = [
            (
                word[4].strip(".,"),
                word,
            )
            for word
            in page.get_text(
                "words"
            )
        ]

        layer_numbers = {
            text: word
            for text, word
            in layer
            if self.is_numeric(text)
            and len(text) >= 3
        }

        ocr_numbers = {
            word["text"].strip(".,"):
            word
            for word in words
            if self.is_numeric(
                word["text"]
            )
            and len(
                word["text"]
            ) >= 3
            and word["conf"] >= 70
        }

        if (
            not layer
            or not ocr_numbers
        ):
            return

        for text, word in (
            layer_numbers.items()
        ):

            if text not in ocr_numbers:

                bbox = [
                    int(
                        word[0] * scale
                    ),
                    int(
                        word[1] * scale
                    ),
                    int(
                        (word[2] - word[0])
                        * scale
                    ),
                    int(
                        (word[3] - word[1])
                        * scale
                    ),
                ]

                self.flag(
                    text,
                    (
                        "Present in PDF text "
                        "layer but not visible "
                        "on rendered page"
                    ),
                    "high",
                    bbox,
                    pno,
                    "layer",
                )

        for text, word in (
            ocr_numbers.items()
        ):

            if text not in layer_numbers:

                self.flag(
                    text,
                    (
                        "Visible on page but "
                        "absent from PDF text layer"
                    ),
                    "medium",
                    [
                        word["x"],
                        word["y"],
                        word["w"],
                        word["h"],
                    ],
                    pno,
                    "layer",
                )

    # =====================================================
    # DATE PARSING
    # =====================================================

    @staticmethod
    def parse_date(
        kind,
        groups,
    ):

        try:

            if kind == "ymd":

                return datetime(
                    int(groups[0]),
                    int(groups[1]),
                    int(groups[2]),
                )

            month = (
                int(groups[1])
                if kind == "dmy"
                else
                MONTHS.split("|").index(
                    groups[1][:3].lower()
                ) + 1
            )

            year = int(
                groups[2]
            )

            if year < 100:
                year += 2000

            try:

                return datetime(
                    year,
                    month,
                    int(groups[0]),
                )

            except ValueError:

                return None

        except (
            ValueError,
            IndexError,
        ):

            return None

    # =====================================================
    # RULE CHECKS
    # =====================================================

    def check_rules(
        self,
        lines,
        pno,
    ):

        now = datetime.now()

        date_styles = Counter()

        decimals = []

        for words in lines:

            text = " ".join(
                word["text"]
                for word in words
            )

            box = self.union_box(
                words
            )

            for kind, regex in DATE_PATTERNS:

                for match in regex.finditer(
                    text
                ):

                    date_styles[
                        kind
                    ] += 1

                    date = self.parse_date(
                        kind,
                        match.groups(),
                    )

                    if date is None:

                        self.flag(
                            match.group(0),
                            "Impossible calendar date",
                            "high",
                            box,
                            pno,
                            "rule",
                        )

                    elif date > now:

                        self.flag(
                            match.group(0),
                            (
                                f"Date is in the future "
                                f"({date:%d %b %Y})"
                            ),
                            "medium",
                            box,
                            pno,
                            "rule",
                        )

                    elif date.year < 1990:

                        self.flag(
                            match.group(0),
                            "Implausibly old date",
                            "low",
                            box,
                            pno,
                            "rule",
                        )

            for word in words:

                text = self.clean(
                    word["text"]
                )

                bbox = [
                    word["x"],
                    word["y"],
                    word["w"],
                    word["h"],
                ]

                if re.fullmatch(
                    r"[\dOolISBZ,.]{3,}",
                    text,
                ):

                    digits = sum(
                        c.isdigit()
                        for c in text
                    )

                    substitutions = sum(
                        c in "OolISBZ"
                        for c in text
                    )

                    if (
                        digits
                        and substitutions
                        and
                        digits
                        / len(text)
                        >= 0.5
                    ):

                        self.flag(
                            text,
                            (
                                "Letters mixed into "
                                "a number "
                                "(O/0, l/1, S/5)"
                            ),
                            "medium",
                            bbox,
                            pno,
                            "rule",
                        )

                if (
                    ","
                    in text
                    and self.is_numeric(text)
                    and len(text) >= 5
                ):

                    core = text.lstrip(
                        "$₹€£"
                    ).rstrip("%")

                    valid = (
                        re.fullmatch(
                            r"\d{1,3}"
                            r"(,\d{3})+"
                            r"(\.\d+)?",
                            core,
                        )
                        or
                        re.fullmatch(
                            r"\d{1,2}"
                            r"(,\d{2})+"
                            r",\d{3}"
                            r"(\.\d+)?",
                            core,
                        )
                    )

                    if not valid:

                        self.flag(
                            text,
                            "Malformed thousands separators",
                            "medium",
                            bbox,
                            pno,
                            "rule",
                        )

                if (
                    self.is_numeric(text)
                    and "."
                    in text
                ):

                    decimals.append(
                        (
                            len(
                                text.rstrip(
                                    "%"
                                ).split(
                                    "."
                                )[-1]
                            ),
                            word,
                        )
                    )

        if len(decimals) >= 4:

            common = Counter(
                count
                for count, _
                in decimals
            ).most_common(1)[0][0]

            for count, word in decimals:

                if (
                    count != common
                    and count <= 3
                ):

                    self.flag(
                        word["text"],
                        (
                            f"{count} decimal places "
                            f"while most amounts use "
                            f"{common}"
                        ),
                        "low",
                        [
                            word["x"],
                            word["y"],
                            word["w"],
                            word["h"],
                        ],
                        pno,
                        "rule",
                    )

        if (
            len(date_styles) > 1
        ):

            self.flag(
                "Dates",
                (
                    "Mixed date formats in "
                    "one document"
                ),
                "low",
                page=pno,
                check="rule",
            )

        self.check_totals(
            lines,
            pno,
        )

    # =====================================================
    # TOTAL CHECKS
    # =====================================================

    def merge_split_labels(
        self,
        lines,
    ):

        output = []

        i = 0

        while i < len(lines):

            current = lines[i]

            tokens = [
                self.clean(
                    word["text"]
                )
                for word in current
            ]

            if (
                i + 1 < len(lines)
                and tokens
                and len(tokens) <= 2
                and KW_RE.fullmatch(
                    tokens[0]
                )
                and not any(
                    self.is_numeric(t)
                    for t in tokens
                )
            ):

                next_tokens = [
                    self.clean(
                        word["text"]
                    ).lower()
                    for word in lines[
                        i + 1
                    ]
                ]

                if (
                    next_tokens
                    and next_tokens[0]
                    in (
                        "amount",
                        "due",
                        "payable",
                        "price",
                    )
                ):

                    output.append(
                        current
                        + lines[i + 1]
                    )

                    i += 2

                    continue

            output.append(
                current
            )

            i += 1

        return output

    @staticmethod
    def word_box(word):

        return [
            word["x"],
            word["y"],
            word["w"],
            word["h"],
        ]

    def check_totals(
        self,
        lines,
        pno,
    ):

        lines = self.merge_split_labels(
            lines
        )

        all_text = " ".join(
            word["text"]
            for line in lines
            for word in line
        )

        has_quantity = bool(
            re.search(
                r"\b(qty|quantity)\b",
                all_text,
                re.I,
            )
        )

        items = []

        subtotal = None
        tax = 0.0
        discount = 0.0
        total = None

        tax_word = None
        tax_rate = None

        for words in lines:

            tokens = [
                self.clean(
                    word["text"]
                )
                for word in words
            ]

            keyword_index = next(
                (
                    i
                    for i, token
                    in enumerate(tokens)
                    if KW_RE.fullmatch(
                        token
                    )
                    and not any(
                        self.is_money(x)
                        for x in tokens[:i]
                    )
                ),
                None,
            )

            if keyword_index is not None:

                segment = " ".join(
                    tokens[
                        keyword_index:
                    ]
                ).lower()

                values = [
                    (
                        self.to_float(
                            self.clean(
                                word["text"]
                            )
                        ),
                        word,
                    )
                    for word
                    in words[
                        keyword_index + 1:
                    ]
                    if self.is_numeric(
                        word["text"]
                    )
                    and "%"
                    not in word["text"]
                ]

                values = [
                    item
                    for item in values
                    if item[0] is not None
                ]

                if not values:
                    continue

                value, word = values[-1]

                if re.match(
                    r"sub-?\s*total",
                    segment,
                ):

                    subtotal = (
                        value,
                        word,
                    )

                elif re.match(
                    r"(grand|total|net|balance)",
                    segment,
                ):

                    total = (
                        value,
                        word,
                    )

                elif re.match(
                    r"(gst|cgst|sgst|igst|vat|tax)",
                    segment,
                ):

                    tax += value
                    tax_word = word

                    match = re.search(
                        r"(\d+(?:\.\d+)?)\s*%",
                        segment,
                    )

                    if (
                        match
                        and tax_rate is None
                    ):

                        tax_rate = float(
                            match.group(1)
                        )

                elif re.match(
                    r"discount",
                    segment,
                ):

                    discount += value

                continue

            if (
                SKIP_ROW.search(
                    " ".join(tokens)
                )
                or
                (
                    tokens
                    and
                    tokens[0].lower()
                    in (
                        "amount",
                        "due",
                        "payable",
                    )
                )
            ):

                continue

            numbers = [
                (
                    self.to_float(token),
                    word,
                    token,
                )
                for token, word
                in zip(tokens, words)
                if (
                    self.is_numeric(token)
                    and "%"
                    not in token
                    and
                    self.to_float(token)
                    is not None
                )
            ]

            money = [
                number
                for number in numbers
                if self.is_money(
                    number[2]
                )
            ]

            if not money:
                continue

            amount, amount_word, _ = (
                money[-1]
            )

            items.append(
                (
                    1,
                    amount,
                )
            )

            if (
                has_quantity
                and len(numbers) >= 3
                and self.is_money(
                    numbers[-2][2]
                )
            ):

                quantity = numbers[-3][0]
                unit_price = numbers[-2][0]

                if (
                    abs(
                        quantity
                        * unit_price
                        - amount
                    ) > 0.02
                    and
                    abs(
                        unit_price
                        - amount
                    ) > 0.02
                ):

                    self.flag(
                        amount_word["text"],
                        (
                            f"Row arithmetic: "
                            f"qty {quantity:g} x "
                            f"unit price "
                            f"{unit_price:,.2f} = "
                            f"{quantity * unit_price:,.2f}, "
                            f"but amount shown is "
                            f"{amount:,.2f}"
                        ),
                        "high",
                        self.word_box(
                            amount_word
                        ),
                        pno,
                        "rule",
                    )

        if not items:
            return

        plain_total = sum(
            value
            for _, value
            in items
        )

        base = (
            subtotal[0]
            if subtotal
            else plain_total
        )

        if (
            subtotal
            and abs(
                subtotal[0]
                - plain_total
            ) > 1.0
        ):

            self.flag(
                f"Subtotal {subtotal[0]:,.2f}",
                (
                    f"Line amounts add up to "
                    f"{plain_total:,.2f}, not "
                    "the stated subtotal"
                ),
                "high",
                self.word_box(
                    subtotal[1]
                ),
                pno,
                "rule",
            )

        if (
            tax_rate
            and tax
            and tax_word
        ):

            expected_tax = (
                base
                * tax_rate
                / 100
            )

            if abs(
                tax
                - expected_tax
            ) > 1.0:

                self.flag(
                    f"Tax {tax:,.2f}",
                    (
                        f"{tax_rate:g}% of "
                        f"{base:,.2f} is "
                        f"{expected_tax:,.2f}, "
                        f"not {tax:,.2f}"
                    ),
                    "medium",
                    self.word_box(
                        tax_word
                    ),
                    pno,
                    "rule",
                )

        if total:

            expected_total = (
                base
                + tax
                - discount
            )

            if abs(
                expected_total
                - total[0]
            ) > 1.0:

                self.flag(
                    f"Total {total[0]:,.2f}",
                    (
                        f"Expected total is "
                        f"{expected_total:,.2f}, "
                        f"but the total shown is "
                        f"{total[0]:,.2f}"
                    ),
                    "high"
                    if subtotal
                    else "medium",
                    self.word_box(
                        total[1]
                    ),
                    pno,
                    "rule",
                )

    # =====================================================
    # DATE LOGIC
    # =====================================================

    def token_date(
        self,
        token,
    ):

        for kind, regex in (
            DATE_PATTERNS[:2]
        ):

            match = regex.fullmatch(
                token
            )

            if match:

                return self.parse_date(
                    kind,
                    match.groups(),
                )

        return None

    def labeled_date(
        self,
        lines,
        label,
    ):

        count = len(label)

        for words in lines:

            tokens = [
                self.clean(
                    word["text"]
                ).lower()
                for word in words
            ]

            for i in range(
                len(tokens)
                - count
                + 1
            ):

                if tuple(
                    tokens[
                        i:i + count
                    ]
                ) == label:

                    for word in words[
                        i + count:
                    ]:

                        date = self.token_date(
                            self.clean(
                                word["text"]
                            )
                        )

                        if date:
                            return (
                                date,
                                word,
                            )

        return None

    def check_date_logic(
        self,
        lines,
        pno,
    ):

        invoice = (
            self.labeled_date(
                lines,
                (
                    "invoice",
                    "date",
                ),
            )
            or
            self.labeled_date(
                lines,
                (
                    "issue",
                    "date",
                ),
            )
        )

        due = self.labeled_date(
            lines,
            (
                "due",
                "date",
            ),
        )

        if not invoice or not due:
            return

        invoice_date, _ = invoice
        due_date, due_word = due

        all_text = " ".join(
            word["text"]
            for line in lines
            for word in line
        )

        if due_date < invoice_date:

            self.flag(
                due_word["text"],
                (
                    f"Due date "
                    f"({due_date:%d %b %Y}) "
                    f"is earlier than invoice "
                    f"date "
                    f"({invoice_date:%d %b %Y})"
                ),
                "high",
                self.word_box(
                    due_word
                ),
                pno,
                "rule",
            )

        else:

            match = re.search(
                r"within\s+(\d+)\s+days",
                all_text,
                re.I,
            )

            if (
                match
                and
                abs(
                    (
                        due_date
                        - invoice_date
                    ).days
                    - int(
                        match.group(1)
                    )
                )
                > 2
            ):

                self.flag(
                    due_word["text"],
                    (
                        f"Due date is "
                        f"{(due_date - invoice_date).days} "
                        "days after invoice date "
                        f"but terms say "
                        f"{match.group(1)} days"
                    ),
                    "medium",
                    self.word_box(
                        due_word
                    ),
                    pno,
                    "rule",
                )

    # =====================================================
    # LOAD PAGES
    # =====================================================

    def load_pages(
        self,
        path,
    ):

        path = str(path)

        extension = (
            os.path.splitext(
                path
            )[1]
            .lower()
        )

        if extension == ".pdf":

            if fitz is None:

                raise RuntimeError(
                    "PDF support requires "
                    "PyMuPDF. Install with: "
                    "pip install pymupdf"
                )

            doc = fitz.open(
                path
            )

            self.check_pdf_metadata(
                path,
                doc,
            )

            scale = (
                self.pdf_dpi
                / 72
            )

            for page_number, page in enumerate(
                doc,
                1,
            ):

                pixmap = page.get_pixmap(
                    dpi=self.pdf_dpi
                )

                image = Image.frombytes(
                    "RGB",
                    (
                        pixmap.width,
                        pixmap.height,
                    ),
                    pixmap.samples,
                )

                yield (
                    page_number,
                    image,
                    page,
                    scale,
                )

            doc.close()

        else:

            image = Image.open(
                path
            )

            self.check_image_metadata(
                image
            )

            yield (
                1,
                image,
                None,
                1,
            )

    # =====================================================
    # ANALYZE
    # =====================================================

    def analyze(
        self,
        path,
        save_report=True,
        save_annotation=True,
        print_output=False,
    ):
        """
        Analyze a PDF/image.

        Returns:
            {
                "file": ...,
                "risk_score": ...,
                "verdict": ...,
                "flags": ...,
                "annotated_files": ...
            }
        """

        path = str(
            Path(path).resolve()
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"File not found: {path}"
            )

        # Reset state
        self.flags = []

        pages_images = {}

        for (
            page_number,
            image,
            pdf_page,
            scale,
        ) in self.load_pages(path):

            pages_images[
                page_number
            ] = image

            words, lines = self.run_ocr(
                image
            )

            if pdf_page is not None:

                self.check_pdf_fonts(
                    pdf_page,
                    page_number,
                    scale,
                )

                self.check_text_layer(
                    pdf_page,
                    words,
                    page_number,
                    scale,
                )

            elif (
                image.format or ""
            ).upper() == "JPEG":

                self.check_ela(
                    image,
                    words,
                )

            self.check_layout(
                words,
                page_number,
            )

            self.check_rules(
                lines,
                page_number,
            )

            self.check_date_logic(
                lines,
                page_number,
            )

        # -------------------------------------------------
        # SCORE
        # -------------------------------------------------

        risk_score = sum(
            SEV_WEIGHT[
                flag["severity"]
            ]
            for flag in self.flags
        )

        strong_flags = [
            flag
            for flag in self.flags
            if flag["severity"]
            in (
                "medium",
                "high",
            )
        ]

        categories = {
            flag["check"]
            for flag in strong_flags
        }

        if (
            len(categories) >= 2
            or risk_score >= 12
        ):

            verdict = (
                "LIKELY TAMPERED"
            )

        elif strong_flags:

            verdict = (
                "SUSPICIOUS"
            )

        else:

            verdict = (
                "NO STRONG SIGNS OF TAMPERING"
            )

        # -------------------------------------------------
        # ANNOTATION
        # -------------------------------------------------

        annotated_files = []

        if save_annotation:

            annotated_files = self.annotate(
                path,
                pages_images,
            )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        result = {
            "file": path,
            "risk_score": risk_score,
            "verdict": verdict,
            "flags": list(
                self.flags
            ),
            "annotated_files":
                annotated_files,
        }

        # -------------------------------------------------
        # JSON REPORT
        # -------------------------------------------------

        if save_report:

            output_path = (
                os.path.splitext(path)[0]
                + "_tamper_report.json"
            )

            with open(
                output_path,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    result,
                    file,
                    indent=2,
                    default=str,
                )

            result[
                "report_file"
            ] = output_path

        # -------------------------------------------------
        # OPTIONAL CONSOLE OUTPUT
        # -------------------------------------------------

        if print_output:

            self.print_report(
                result
            )

        return result

    # =====================================================
    # ANNOTATION
    # =====================================================

    @staticmethod
    def legend_font(
        size=22,
    ):

        for name in (
            "arial.ttf",
            "DejaVuSans.ttf",
            "/usr/share/fonts/"
            "truetype/dejavu/"
            "DejaVuSans.ttf",
        ):

            try:

                return ImageFont.truetype(
                    name,
                    size,
                )

            except OSError:
                continue

        return ImageFont.load_default()

    @staticmethod
    def highlight(
        image,
        boxes,
        rgb,
        alpha,
        pad=3,
    ):

        layer = Image.new(
            "RGBA",
            image.size,
            (0, 0, 0, 0),
        )

        draw = ImageDraw.Draw(
            layer
        )

        for x, y, w, h in boxes:

            draw.rounded_rectangle(
                [
                    x - pad,
                    y - pad,
                    x + w + pad,
                    y + h + pad,
                ],
                radius=4,
                fill=rgb + (alpha,),
            )

        return Image.alpha_composite(
            image,
            layer,
        )

    @staticmethod
    def save_png(
        image,
        output,
    ):

        output = os.path.abspath(
            os.path.normpath(
                output
            )
        )

        directory = os.path.dirname(
            output
        )

        os.makedirs(
            directory,
            exist_ok=True,
        )

        try:

            with open(
                output,
                "wb",
            ) as file:

                image.save(
                    file,
                    format="PNG",
                )

            return output

        except OSError:

            fallback = os.path.join(
                tempfile.gettempdir(),
                os.path.basename(
                    output
                ),
            )

            with open(
                fallback,
                "wb",
            ) as file:

                image.save(
                    file,
                    format="PNG",
                )

            return fallback

    def annotate(
        self,
        path,
        pages_images,
    ):

        annotated_files = []

        RED = (
            255,
            0,
            0,
        )

        YELLOW = (
            255,
            214,
            0,
        )

        for page_number, image in (
            pages_images.items()
        ):

            rgba = image.convert(
                "RGBA"
            )

            page_flags = [
                flag
                for flag in self.flags
                if (
                    flag["page"]
                    == page_number
                    and flag["bbox"]
                )
            ]

            yellow = [
                flag["bbox"]
                for flag in page_flags
                if flag["marker"]
                == "yellow"
            ]

            red = [
                flag["bbox"]
                for flag in page_flags
                if flag["marker"]
                == "red"
            ]

            rgba = self.highlight(
                rgba,
                yellow,
                YELLOW,
                150,
                pad=4,
            )

            rgba = self.highlight(
                rgba,
                red,
                RED,
                110,
                pad=3,
            )

            bar_height = 52

            output = Image.new(
                "RGB",
                (
                    rgba.width,
                    rgba.height
                    + bar_height,
                ),
                "white",
            )

            output.paste(
                rgba.convert("RGB"),
                (
                    0,
                    bar_height,
                ),
            )

            draw = ImageDraw.Draw(
                output
            )

            font = self.legend_font(
                22
            )

            draw.rounded_rectangle(
                [16, 12, 44, 40],
                radius=4,
                fill=(
                    255,
                    150,
                    150,
                ),
            )

            draw.text(
                (54, 12),
                "Suspicious text / value",
                fill="black",
                font=font,
            )

            draw.rounded_rectangle(
                [320, 12, 348, 40],
                radius=4,
                fill=(
                    255,
                    226,
                    90,
                ),
            )

            draw.text(
                (358, 12),
                "Font change",
                fill="black",
                font=font,
            )

            base = os.path.splitext(
                path
            )[0]

            if len(pages_images) == 1:

                output_path = (
                    base
                    + "_annotated.png"
                )

            else:

                output_path = (
                    base
                    + f"_p{page_number}"
                    + "_annotated.png"
                )

            saved = self.save_png(
                output,
                output_path,
            )

            annotated_files.append(
                saved
            )

        return annotated_files

    # =====================================================
    # PRINT REPORT
    # =====================================================

    @staticmethod
    def print_report(
        result,
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            "DOCUMENT TAMPERING REPORT"
        )

        print(
            "=" * 80
        )

        print(
            f"File       : "
            f"{result['file']}"
        )

        print(
            f"Risk score : "
            f"{result['risk_score']}"
        )

        print(
            f"Verdict    : "
            f"{result['verdict']}"
        )

        print(
            "\nFlags:"
        )

        if not result["flags"]:

            print(
                "  No flags raised."
            )

        else:

            for flag in sorted(
                result["flags"],
                key=lambda item:
                -SEV_WEIGHT[
                    item["severity"]
                ],
            ):

                print(
                    f"[{flag['severity'].upper():6}] "
                    f"p{flag['page']} "
                    f"({flag['check']}) "
                    f"'{flag['field']}': "
                    f"{flag['reason']}"
                )

        print(
            "\nNote: flags are indicators, "
            "not proof. Verify with the issuer."
        )

        print(
            "=" * 80
        )


# =========================================================
# SIMPLE FUNCTION API
# =========================================================

_default_detector = None


def analyze_document(
    path,
    save_report=True,
    save_annotation=True,
    print_output=False,
):
    """
    Convenience function.

    You don't need to manually create
    DocumentTamperingDetector.
    """

    global _default_detector

    if _default_detector is None:

        _default_detector = (
            DocumentTamperingDetector()
        )

    return _default_detector.analyze(
        path,
        save_report=save_report,
        save_annotation=save_annotation,
        print_output=print_output,
    )

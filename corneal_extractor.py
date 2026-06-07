"""
Optovue Corneal Scan Data Extractor — sector OCR core.

Renders the report at 600 DPI and reads the 17 sectors of each circular map
(Pachymetry + Epithelium) plus the map-diameter labels, into a flat dict that
you can write to CSV. Designed to drop into the existing tkinter app: the
per-zone call signature is unchanged:

    res = ocr_sector(gray, os.path.join(debug_dir, f"{zone_name}_ocr.png"))
    extracted_data[zone_name] = res

All sector coordinates are for a 600 DPI render of this Optovue layout. The
layout is fixed across reports of this device, so the boxes are constant; the
boxes are deliberately a little generous because ocr_sector tightens onto the
digit blobs itself.
"""

import os

# Pin each tesseract process to a single OMP thread. The sector crops are tiny,
# so tesseract's internal multithreading buys nothing and only causes CPU
# oversubscription once we fan many OCR processes out across the thread pool
# below. Must be set before pytesseract spawns any subprocess.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import re
import shutil
import subprocess
import cv2
import numpy as np
import pandas as pd
import pytesseract
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_path

DPI = 600  # all coordinates below assume this render resolution


# --------------------------------------------------------------------------
# Core OCR: one sector -> digit string
# --------------------------------------------------------------------------
# Per zone: read at 1x and 2x scale, under each morph kernel, under each psm
# mode, then vote. The three sources of diversity each catch a distinct
# failure mode:
#   - kernels: a stronger opening preserves a thin "9" descender that a
#     gentler one loses; a stronger closing seals a "5" gap that would
#     otherwise read as "8".
#   - scale: 1x is fast and reads most cells correctly; 2x gives Tesseract
#     more pixels per stroke and rescues the descender / hook cases that 1x
#     misreads.
#   - psm: 6 is layout-aware (uniform block, reliable for tabular digits);
#     8 bypasses layout entirely (single-word, catches failures that all the
#     layout-aware modes share).
# 3 kernels × 2 scales × 2 psm = 12 OCR calls per zone, half of the original
# 4-psm ensemble's 24.
_KERNELS = [(5, 3), (5, 5), (7, 3)]
_SCALES = (1, 2)
_PSM_MODES = (6, 7, 8, 13)


def _ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


_PROBE_KERNEL = _ellipse(5)


def _binarize(gray):
    """Pick a binary mask whose minority pixel class is the digit text.

    Most cells are 2-population (dark text on a coloured fill, or white text
    on dark blue): the polarity with more digit-shaped blobs wins. Cells at
    the map's outer rim are 3-population (text + fill + page-white outside
    the circle); first Otsu lumps text + fill, so we fall back to a 2nd-pass
    Otsu on the darker half. The 3-pop fallback is gated behind "primaries
    found nothing" because it's aggressive and can over-erode a "4" into "1".
    """
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def score(cand):
        m = cand.astype("uint8") * 255
        opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, _PROBE_KERNEL)
        cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n = sum(
            1
            for c in cnts
            for b in (cv2.boundingRect(c),)
            if 38 < b[3] < 110 and 6 < b[2] < 110 and b[3] / b[2] <= 4.0
        )
        return m, n

    primary = max((score(gray < t), score(gray > t)), key=lambda r: r[1])
    if primary[1] > 0:
        return primary[0]

    below = gray[gray < t]
    if len(below) > 100:
        td, _ = cv2.threshold(below, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m, n = score(gray < td)
        if n > 0:
            return m
    return primary[0]


def _digit_groups(mask, scale):
    """Find all digit-blob clusters in a binary mask. Returns a list of
    (distance_from_centre_squared, padded_inverted_image_for_ocr) tuples,
    one per spatially-grouped run of digits. Adjacent compass labels at the
    corners get their own group (and lose the distance comparison to the
    centred number group, so they're sorted to the back when picking).
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [
        b
        for b in map(cv2.boundingRect, cnts)
        if 38 * scale < b[3] < 110 * scale
        and 6 * scale < b[2] < 110 * scale
        and b[3] / b[2] <= 4.0
    ]
    if not blobs:
        return []

    blobs.sort(key=lambda b: b[0])
    used = [False] * len(blobs)
    groups = []
    for i, b in enumerate(blobs):
        if used[i]:
            continue
        x0, y0, x1, y1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, c in enumerate(blobs):
                if used[j]:
                    continue
                cx0, cy0, cx1, cy1 = c[0], c[1], c[0] + c[2], c[1] + c[3]
                if (
                    min(y1, cy1) - max(y0, cy0) > 15 * scale
                    and max(cx0 - x1, x0 - cx1) < 48 * scale
                ):
                    x0, y0 = min(x0, cx0), min(y0, cy0)
                    x1, y1 = max(x1, cx1), max(y1, cy1)
                    used[j] = True
                    changed = True
        groups.append((x0, y0, x1, y1))

    H, W = mask.shape
    pad = 8 * scale
    out = []
    for x0, y0, x1, y1 in groups:
        sub = mask[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad]
        inv = cv2.copyMakeBorder(
            255 - sub,
            20 * scale,
            20 * scale,
            20 * scale,
            20 * scale,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        # distance from crop centre, normalised back to 1x for fair ranking
        # across scales when we pool candidates later.
        dist = (((x0 + x1) / 2 - W / 2) ** 2 + ((y0 + y1) / 2 - H / 2) ** 2) / (
            scale**2
        )
        out.append((dist, inv))
    return out


def ocr_sector(gray, debug_path=None, expected_len=None):
    """OCR the digit value in a single sector crop.

    For each scale (1x, 2x) and morph kernel, find every digit-blob cluster
    in the preprocessed crop and OCR each one under several psm modes. Pool
    reads from groups that sit near the crop centre (the actual sector value,
    not nearby compass labels) and vote — right-length reads outrank
    wrong-length ones, then most-common wins.
    """
    candidates = []  # (dist, reads, image)
    for scale in _SCALES:
        scaled = (
            gray
            if scale == 1
            else cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        )
        base = _binarize(scaled)
        for open_k, close_k in _KERNELS:
            mask = cv2.morphologyEx(base, cv2.MORPH_OPEN, _ellipse(open_k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(close_k))
            for dist, img in _digit_groups(mask, scale):
                reads = []
                for psm in _PSM_MODES:
                    txt = pytesseract.image_to_string(
                        img,
                        config=f"--psm {psm} -c tessedit_char_whitelist=0123456789 ",
                    )
                    d = "".join(re.findall(r"\d", txt))
                    if d:
                        reads.append(d)
                if reads:
                    candidates.append((dist, reads, img))

    if not candidates:
        return ""

    # Rank candidates: right-length voted reads first, multi-digit next, then
    # nearest to the crop centre. This picks the actual sector-value group
    # (and skips an adjacent compass label that happened to look digit-shaped
    # under one of the morph kernels) consistently across preprocessings.
    def vote(reads):
        if expected_len is not None:
            r = [x for x in reads if len(x) == expected_len]
            if r:
                reads = r
        return max(Counter(reads).items(), key=lambda kv: (kv[1], len(kv[0])))[0]

    def rank(c):
        dist, reads, _ = c
        v = vote(reads)
        wrong_len = expected_len is not None and len(v) != expected_len
        return (wrong_len, len(v) < 2, dist)

    candidates.sort(key=rank)

    # Pool reads from candidates spatially coincident with the top-ranked one
    # (~50 px slack at 1x) so all (scale, kernel) variants of the same digit
    # group contribute to the final vote.
    best_dist = candidates[0][0]
    pooled = [
        r for d, reads, _ in candidates if abs(d - best_dist) < 2500 for r in reads
    ]

    if debug_path:
        cv2.imwrite(debug_path, candidates[0][2])
    return vote(pooled)


def read_diameters(gray, debug_path=None):
    """Read the 'map diameters' label strip (e.g. '2.0, 5.0, 6.0').

    The decimal dots render too small to be reliable across Tesseract builds,
    so we OCR digits-only and reconstruct one decimal place from the right.
    Pass the grayscale crop of the diameter zone ([2134, 2330, 418, 68] at 600 DPI).
    """
    if debug_path:
        cv2.imwrite(debug_path, gray)
    txt = pytesseract.image_to_string(
        gray, config=r"--psm 7 -c tessedit_char_whitelist=0123456789/ "
    )
    segments = re.findall(r"\d+", txt)
    values = [f"{s[:-1]}.{s[-1]}" for s in segments if len(s) >= 2]
    return ", ".join(values)


# --------------------------------------------------------------------------
# Sector coordinates (cx, cy, w, h), centre of box, at 600 DPI.
# Validated 17/17 on each map against the reference report.
# --------------------------------------------------------------------------
# Sector keys are positional, not anatomical: the maps are always laid out
# the same way on the page regardless of which eye the report is for, so we
# read each cell by its visual position and let downstream callers translate
# top_left ↔ SN/ST etc. once they know the eye. This keeps OCR free of any
# header-parsing or label-swap step.
PACHYMETRY_BOXES = {
    "center": (3111, 3480, 240, 150),
    "inner_top": (3107, 2923, 240, 150),
    "inner_top_right": (3503, 3087, 240, 150),
    "inner_right": (3635, 3524, 240, 150),
    "inner_bottom_right": (3503, 3872, 240, 150),
    "inner_bottom": (3114, 4033, 240, 150),
    "inner_bottom_left": (2718, 3872, 240, 150),
    "inner_left": (2620, 3440, 240, 150),
    "inner_top_left": (2718, 3087, 240, 150),
    "outer_top": (3107, 2603, 240, 150),
    "outer_top_right": (3724, 2838, 240, 150),
    "outer_right": (3945, 3400, 240, 150),
    "outer_bottom_right": (3710, 4095, 240, 150),
    "outer_bottom": (3114, 4353, 240, 150),
    "outer_bottom_left": (2540, 4070, 200, 120),
    "outer_left": (2315, 3550, 240, 150),
    "outer_top_left": (2525, 2865, 240, 150),
}

EPITHELIUM_BOXES = {
    "center": (5066, 3480, 240, 150),
    "inner_top": (5066, 2923, 240, 150),
    "inner_top_right": (5458, 3087, 240, 150),
    "inner_right": (5588, 3524, 240, 150),
    "inner_bottom_right": (5498, 3872, 200, 120),
    "inner_bottom": (5066, 4031, 240, 150),
    "inner_bottom_left": (4693, 3832, 240, 150),
    "inner_left": (4565, 3440, 240, 150),
    "inner_top_left": (4673, 3087, 240, 150),
    "outer_top": (5064, 2603, 240, 150),
    "outer_top_right": (5620, 2870, 200, 120),
    "outer_right": (5840, 3400, 200, 120),
    "outer_bottom_right": (5620, 4070, 200, 120),
    "outer_bottom": (5068, 4353, 240, 150),
    "outer_bottom_left": (4482, 4076, 200, 120),
    "outer_left": (4270, 3550, 200, 120),
    "outer_top_left": (4480, 2858, 240, 150),
}

DIAMETER_BOX = (2134, 2330, 418, 68)  # x, y, w, h (top-left), 600 DPI

# "Right / OD" or "Left / OS" eye label, top-right of the page at 600 DPI.
EYE_LABEL_BOX = (5800, 400, 800, 300)


def read_eye(img):
    """Return 'OD' or 'OS' by OCR'ing the top-right header label.

    Tesseract often mangles the rendered "Left / OS" into "Lef:' 0S" and
    "Right / OD" into "Righ': oD", so we upper-case, drop spaces and treat
    "0"->"O" before matching. Returns "" if neither pattern matches.
    """
    x, y, w, h = EYE_LABEL_BOX
    strip = img[y : y + h, x : x + w]
    if strip.ndim == 3:
        strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    txt = pytesseract.image_to_string(strip, config="--psm 7")
    norm = txt.upper().replace("0", "O").replace(" ", "")
    if "OD" in norm or "RIGHT" in norm:
        return "OD"
    if "OS" in norm or "LEFT" in norm:
        return "OS"
    return ""


# expected digit count per map -> used to flag suspicious reads
EXPECTED_LEN = {"pachymetry": 3, "epithelium": 2}


# --------------------------------------------------------------------------
# Patient demographics — pulled from the PDF's embedded text layer (no OCR).
# --------------------------------------------------------------------------
# The Optovue report header is real text (not part of the rasterised maps), so
# poppler's pdftotext reads it directly and exactly. pdf2image already pulls in
# poppler, so pdftotext is available without adding a dependency. Each field is
# a "Label: value" pair on the header lines; values run up to a 2+ space gap
# (the next column) or end of line.
PATIENT_FIELDS = {
    "patient": "Patient",
    "exam_date": "Exam Date",
    "dob": r"DOB\(age\)",
    "gender": "Gender",
}


def extract_patient_info(pdf_path):
    """Return {patient, exam_date, dob, gender} from the PDF's text layer.

    Uses pdftotext (poppler) rather than OCR — the header is selectable text.
    Missing fields come back as "". The DOB column reads "29/10/1941 (79)";
    we keep only the date and drop the parenthesised age.
    """
    exe = shutil.which("pdftotext")
    if not exe:
        return {k: "" for k in PATIENT_FIELDS}
    text = subprocess.run(
        [exe, "-layout", pdf_path, "-"],
        capture_output=True,
        text=True,
    ).stdout

    info = {}
    for key, label in PATIENT_FIELDS.items():
        # value = everything after "Label:" up to a 2+ space column gap or EOL
        m = re.search(rf"{label}:\s*(.*?)(?:\s{{2,}}|$)", text, re.MULTILINE)
        val = m.group(1).strip() if m else ""
        if key == "dob":
            val = re.sub(r"\s*\(.*\)\s*$", "", val)  # strip trailing "(age)"
        info[key] = val
    return info


# --------------------------------------------------------------------------
# Extraction driver
# --------------------------------------------------------------------------
def _crop_center(img, box):
    cx, cy, w, h = box
    return img[cy - h // 2 : cy + h // 2, cx - w // 2 : cx + w // 2]


def extract_corneal_data(pdf_path, debug_dir=None):
    """Extract all sectors from both maps + diameters.

    Returns (data, warnings):
      data     : {"pachymetry_center": "589", "pachymetry_inner_top": "600",
                  "epithelium_outer_top_left": "54", ...,
                  "map_diameters": "2.0, 5.0, 6.0"}
      warnings : list of zone names whose read has the wrong digit count
                 (3 for pachymetry, 2 for epithelium) or came back empty.

    Sector keys are positional (top, top_left, top_right, right, bottom_right,
    bottom, bottom_left, left) so the same code works for OD and OS reports —
    callers translate to anatomical sectors (SN/ST/IN/IT/T/N) once they know
    which eye the report is for.
    """
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    page = convert_from_path(pdf_path, dpi=DPI)[0]
    img = np.array(page)[:, :, ::-1].copy()  # PIL RGB -> OpenCV BGR

    # Patient demographics from the text layer (no OCR), then the eye label.
    data = dict(extract_patient_info(pdf_path))
    data["eye"] = read_eye(img)
    warnings = []

    # Each sector is independent and dominated by tesseract subprocess time
    # (which releases the GIL), so fan the 34 zones out across a thread pool.
    # Results are identical to the serial version — only the wall time drops.
    def read_zone(args):
        map_name, sector, box, want = args
        zone_name = f"{map_name}_{sector}"
        gray = cv2.cvtColor(_crop_center(img, box), cv2.COLOR_BGR2GRAY)
        dbg = os.path.join(debug_dir, f"{zone_name}_ocr.png") if debug_dir else None
        return zone_name, ocr_sector(gray, dbg, expected_len=want), want

    tasks = [
        (map_name, sector, box, EXPECTED_LEN[map_name])
        for map_name, boxes in (
            ("pachymetry", PACHYMETRY_BOXES),
            ("epithelium", EPITHELIUM_BOXES),
        )
        for sector, box in boxes.items()
    ]
    with ThreadPoolExecutor(max_workers=min(len(tasks), os.cpu_count() or 4)) as pool:
        for zone_name, res, want in pool.map(read_zone, tasks):
            data[zone_name] = res
            if len(res) != want:
                warnings.append(zone_name)

    # map diameters
    dx, dy, dw, dh = DIAMETER_BOX
    gdia = cv2.cvtColor(img[dy : dy + dh, dx : dx + dw], cv2.COLOR_BGR2GRAY)
    dbg = os.path.join(debug_dir, "map_diameters_ocr.png") if debug_dir else None
    data["map_diameters"] = read_diameters(gdia, dbg)

    return data, warnings


def write_csv(data, csv_path):
    """One row per zone -> CSV (matches the existing extracted_data shape)."""
    df = pd.DataFrame([{"zone": k, "value": v} for k, v in data.items()])
    df.to_csv(csv_path, sep=";", index=False)
    return csv_path


def pick_pdf():
    """Open a native file picker and return the chosen PDF path ('' if cancelled).

    Uses tkinter's filedialog, which maps to the native Cocoa open panel on
    macOS (and the native dialogs on Windows/Linux) with no extra dependency.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # hide the empty root window, show only the dialog
    root.update()
    path = filedialog.askopenfilename(
        title="Select an Optovue OCT report (PDF)",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
    )
    root.destroy()
    return path


def pick_csv_save(default_name="corneal_data.csv"):
    """Open a native 'Save As' picker and return the chosen path ('' if cancelled)."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.asksaveasfilename(
        title="Save extracted data as",
        defaultextension=".csv",
        initialfile=default_name,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    return path


if __name__ == "__main__":
    import sys

    # PDF path comes from the command line if given, otherwise from the picker.
    if len(sys.argv) > 1:
        pdf = sys.argv[1]
    else:
        pdf = pick_pdf()
        if not pdf:
            print("No file selected — nothing to do.")
            sys.exit(0)

    print(f"start: {pdf}")
    data, warnings = extract_corneal_data(pdf, debug_dir="debug")
    for k, v in data.items():
        print(f"{k:26s} {v}")
    if warnings:
        print("\nFLAGGED (check debug/*_ocr.png):", ", ".join(warnings))
    else:
        print("\nAll zones passed the digit-count check.")

    out = pick_csv_save() if len(sys.argv) <= 1 else "corneal_data.csv"
    if out:
        write_csv(data, out)
        print(f"wrote {out}")

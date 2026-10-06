#!/usr/bin/env python3
"""Generate a fillable PDF for reviewing the modality of every MRI series in a dataset.

The dataset directory is searched recursively for NIfTI files (.nii, .nii.gz) and DICOM
series. Each series gets one page showing the axial, sagittal and coronal center slice (with
--extra_slices also the slices at center -/+ 1/4 of the field of view), a multiple choice
box for the modality, a notes field and a box with header information. For segmentations
(images that hold integer labels) the slices are centered on the center of mass of the labels
and SEG is selected in advance. The filled-in PDF is read back with read_review_pdf.py.

Datasets with more than 100 series are split into several PDFs (see --max-pages).

Usage:
    python make_review_pdf.py /path/to/dataset -o modality_review.pdf
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import nibabel as nib
import numpy as np
import pydicom
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from reportlab.lib.colors import black, grey, lightgrey, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

MODALITIES = ("T1", "T1c", "T2", "FLAIR", "ADC", "DTI", "PERF", "SEG", "OTHER")
NIFTI_SUFFIXES = (".nii", ".nii.gz")
MAX_LABELS = 64  # images with more different values than this are not taken for a segmentation

# Form field names are "<prefix>_<page number>"; read_review_pdf.py relies on these. The path is
# percent-encoded because form fields with the built-in PDF fonts cannot hold arbitrary characters.
FIELD_PATH, FIELD_UID, FIELD_MODALITY, FIELD_NOTES = "path", "uid", "modality", "notes"

# Page layout in points
PAGE_W = A4[0]
MARGIN = 28
CONTENT_W = PAGE_W - 2 * MARGIN
TITLE_H = 24
COLUMN_TITLE_H = 16  # column titles above the slices
ROW_LABEL_W = 19  # row labels left of the slices, only with --extra_slices
MODALITY_H = 54
NOTES_H = 60
INFO_H = 86
GAP = 8
LABEL_W = 56

PLANES = (("Axial", 2), ("Sagittal", 0), ("Coronal", 1))  # title, slicing axis of an RAS volume
ROWS = (("center \N{MINUS SIGN} \N{VULGAR FRACTION ONE QUARTER} FOV", -0.25), ("center", 0.0),
        ("center + \N{VULGAR FRACTION ONE QUARTER} FOV", 0.25))
MISSING = "\N{EN DASH}"


def figure_height(extra_slices: bool) -> float:
    """Height of the slice grid that gives every slice a square cell."""
    n_rows, label_w = (3, ROW_LABEL_W) if extra_slices else (1, 0)
    return n_rows * (CONTENT_W - label_w) / 3 + COLUMN_TITLE_H


def page_height(extra_slices: bool) -> float:
    """A4 for the 3 x 3 grid, otherwise the page is cut to its content."""
    content = 2 * MARGIN + TITLE_H + figure_height(extra_slices) + 3 * GAP + MODALITY_H + NOTES_H + INFO_H
    return A4[1] if extra_slices else content


@dataclass
class Series:
    path: str  # relative to the dataset root: the file (NIfTI) or the series folder (DICOM)
    kind: str  # "nifti" or "dicom"
    files: list[str]
    uid: str = ""  # SeriesInstanceUID (DICOM only)
    label: str = ""  # what is printed on the page


# --------------------------------------------------------------------------------------
# Dataset traversal
# --------------------------------------------------------------------------------------

def scan_directory(task: tuple[str, str, list[str]]) -> list[Series]:
    """Find the NIfTI files and DICOM series that live directly in one directory."""
    root, dirpath, filenames = task
    found = []
    dicom_files = defaultdict(list)
    for name in filenames:
        full = os.path.join(dirpath, name)
        if name.lower().endswith(NIFTI_SUFFIXES):
            rel = os.path.relpath(full, root)
            found.append(Series(rel, "nifti", [full], label=rel))
            continue
        try:
            ds = pydicom.dcmread(full, stop_before_pixels=True, specific_tags=["SeriesInstanceUID", "Rows"])
        except Exception:  # not a DICOM file
            continue
        if "Rows" in ds:  # skip DICOM objects without an image (DICOMDIR, reports, ...)
            dicom_files[str(ds.get("SeriesInstanceUID", ""))].append(full)
    rel = os.path.relpath(dirpath, root)
    for uid, files in sorted(dicom_files.items()):
        label = rel if len(dicom_files) == 1 else f"{rel}  [{uid}]"
        found.append(Series(rel, "dicom", files, uid, label))
    return found


def find_series(root: Path, pool: ProcessPoolExecutor | None) -> list[Series]:
    tasks = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        filenames = sorted(f for f in filenames if not f.startswith("."))
        if filenames:
            tasks.append((str(root), dirpath, filenames))
    results = pool.map(scan_directory, tasks, chunksize=4) if pool else map(scan_directory, tasks)
    return [series for found in results for series in found]


# --------------------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------------------

def fmt(value, unit: str = "") -> str:
    """Format a header value for display."""
    if value is None or isinstance(value, (bytes, pydicom.Sequence)):
        return MISSING
    if isinstance(value, (list, tuple, pydicom.multival.MultiValue)):
        text = "\\".join(str(v) for v in value)
    elif isinstance(value, (int, float, np.number)):
        text = f"{float(value):.5g}"
    else:
        text = str(value).strip()
    return f"{text}{unit}" if text else MISSING


def join(*parts: str, sep: str = " \N{MIDDLE DOT} ") -> str:
    return sep.join(p for p in parts if p and p != MISSING) or MISSING


def timing(tr, te, ti, flip) -> str:
    if all(v == MISSING for v in (tr, te, ti, flip)):
        return MISSING
    return f"{tr} / {te} / {ti} ms, flip {flip}\N{DEGREE SIGN}"


def to_canonical(vol: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorient a volume to RAS+ so that the axes are sagittal, coronal and axial."""
    zooms = nib.affines.voxel_sizes(affine)
    ornt = nib.orientations.io_orientation(affine)
    if np.isnan(ornt).any() or not np.all(zooms > 0):
        return vol, np.ones(3)
    out_zooms = np.ones(3)
    out_zooms[ornt[:, 0].astype(int)] = zooms
    return nib.orientations.apply_orientation(vol, ornt), out_zooms


def value_info(vol: np.ndarray, dtype) -> str:
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return f"{dtype}, no finite values"
    return f"{dtype}, range {float(finite.min()):.5g} \N{HORIZONTAL ELLIPSIS} {float(finite.max()):.5g}"


def load_nifti(path: str) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    img = nib.load(path)
    shape = img.shape
    # Only the first volume of 4D data is read
    vol = np.asanyarray(img.dataobj[(slice(None),) * 3 + (0,) * (len(shape) - 3)] if len(shape) > 3 else img.dataobj)
    if vol.dtype.names:  # RGB
        vol = np.mean([vol[name] for name in vol.dtype.names], axis=0)
    if np.iscomplexobj(vol):
        vol = np.abs(vol)
    vol = vol.reshape(vol.shape + (1,) * (3 - vol.ndim)).astype(np.float32)

    header = img.header
    zooms = header.get_zooms()
    sidecar = {}
    sidecar_path = path[: -len(".nii.gz") if path.lower().endswith(".gz") else -len(".nii")] + ".json"
    if os.path.isfile(sidecar_path):
        try:
            with open(sidecar_path) as f:
                sidecar = json.load(f)
        except (OSError, ValueError):
            pass

    def ms(key):  # BIDS sidecars store times in seconds
        return fmt(sidecar[key] * 1000) if isinstance(sidecar.get(key), (int, float)) else MISSING

    matrix = " \N{MULTIPLICATION SIGN} ".join(str(n) for n in shape[:3])
    if len(shape) > 3:
        matrix += f" \N{MULTIPLICATION SIGN} {int(np.prod(shape[3:]))} volumes"
        if len(zooms) > 3 and zooms[3] > 0:
            unit = header.get_xyzt_units()[1]
            matrix += f" (time step {zooms[3]:.4g}{'' if unit == 'unknown' else ' ' + unit})"
    info = [
        ("Format", type(img).__name__.replace("Image", "").replace("Nifti", "NIfTI-")),
        ("Description", join(fmt(sidecar.get("SeriesDescription")), fmt(header["descrip"].item().decode("latin-1")))),
        ("Protocol", fmt(sidecar.get("ProtocolName"))),
        ("Sequence", join(fmt(sidecar.get("ScanningSequence")), fmt(sidecar.get("SequenceVariant")),
                          fmt(sidecar.get("SequenceName")), fmt(sidecar.get("MRAcquisitionType")))),
        ("TR / TE / TI", timing(ms("RepetitionTime"), ms("EchoTime"), ms("InversionTime"), fmt(sidecar.get("FlipAngle")))),
        ("Contrast", fmt(sidecar.get("ContrastBolusAgent"))),
        ("Scanner", join(fmt(sidecar.get("Manufacturer")), fmt(sidecar.get("ManufacturersModelName")),
                         fmt(sidecar.get("MagneticFieldStrength"), " T"), sep=" ")),
        ("Matrix", matrix),
        ("Voxel size", " \N{MULTIPLICATION SIGN} ".join(f"{z:.2f}" for z in zooms[:3]) + " mm"),
        ("Values", value_info(vol, header.get_data_dtype())),
        ("Image type", fmt(sidecar.get("ImageType"))),
    ]
    vol, zooms = to_canonical(vol, img.affine)
    return vol, zooms, info


def functional_group(ds: pydicom.Dataset, frame: int, sequence: str, keyword: str):
    """Look up a per-frame or shared attribute of a multi-frame DICOM object."""
    per_frame = ds.get("PerFrameFunctionalGroupsSequence") or []
    shared = ds.get("SharedFunctionalGroupsSequence") or []
    for group in (per_frame[frame] if frame < len(per_frame) else None, shared[0] if shared else None):
        if group is not None and group.get(sequence) and keyword in group[sequence][0]:
            return group[sequence][0][keyword].value
    return None


def pixels(ds: pydicom.Dataset) -> np.ndarray:
    """Pixel data as float32 with shape (frames, rows, columns)."""
    arr = ds.pixel_array.astype(np.float32)
    if int(ds.get("SamplesPerPixel", 1)) > 1:
        arr = arr.mean(axis=-1)
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        arr = arr.max() - arr
    return arr.reshape((-1,) + arr.shape[-2:])


def load_dicom(files: list[str]) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    datasets = [ds for ds in (pydicom.dcmread(f) for f in files) if "PixelData" in ds]
    if not datasets:
        raise ValueError("series contains no pixel data")
    is_seg = datasets[0].get("Modality") == "SEG"
    multiframe = int(datasets[0].get("NumberOfFrames", 1)) > 1 or "PerFrameFunctionalGroupsSequence" in datasets[0]

    if multiframe:
        ds = datasets[0]
        frames = list(pixels(ds))
        n = len(frames)
        positions = [functional_group(ds, i, "PlanePositionSequence", "ImagePositionPatient") for i in range(n)]
        orientation = functional_group(ds, 0, "PlaneOrientationSequence", "ImageOrientationPatient")
        spacing = functional_group(ds, 0, "PixelMeasuresSequence", "PixelSpacing") or ds.get("PixelSpacing")
        thickness = (functional_group(ds, 0, "PixelMeasuresSequence", "SpacingBetweenSlices")
                     or functional_group(ds, 0, "PixelMeasuresSequence", "SliceThickness"))
        if is_seg and len(ds.get("SegmentSequence", [])) > 1:  # combine the segments into a label map
            for i in range(n):
                number = functional_group(ds, i, "SegmentIdentificationSequence", "ReferencedSegmentNumber") or 1
                frames[i] = (frames[i] > 0) * np.float32(number)
        else:
            for i in range(n):
                slope = functional_group(ds, i, "PixelValueTransformationSequence", "RescaleSlope")
                intercept = functional_group(ds, i, "PixelValueTransformationSequence", "RescaleIntercept")
                frames[i] = frames[i] * float(slope or 1) + float(intercept or 0)
    else:
        # Keep the slices that share the most common matrix size and orientation (drops stray localizers)
        def geometry(d):
            orientation = d.get("ImageOrientationPatient")
            return d.Rows, d.Columns, tuple(round(float(x), 2) for x in orientation) if orientation else None

        common = Counter(geometry(d) for d in datasets).most_common(1)[0][0]
        datasets = sorted((d for d in datasets if geometry(d) == common), key=lambda d: int(d.get("InstanceNumber") or 0))
        ds = datasets[0]
        frames = [pixels(d)[0] * float(d.get("RescaleSlope") or 1) + float(d.get("RescaleIntercept") or 0)
                  for d in datasets]
        positions = [d.get("ImagePositionPatient") for d in datasets]
        orientation = ds.get("ImageOrientationPatient")
        spacing = ds.get("PixelSpacing") or ds.get("ImagerPixelSpacing")
        thickness = ds.get("SliceThickness")

    # Slice index of every frame from its position along the slice normal
    spacing = [float(x) for x in spacing] if spacing else [1.0, 1.0]
    has_geometry = orientation is not None and all(p is not None for p in positions)
    if has_geometry:
        row_dir, col_dir = np.array(orientation[:3], float), np.array(orientation[3:], float)
        normal = np.cross(row_dir, col_dir)
        distance = np.round([np.dot(np.array(p, float), normal) for p in positions], 2)
        unique = np.unique(distance)
        dz = float(np.median(np.diff(unique))) if len(unique) > 1 else float(thickness or 1)
        index = np.rint((distance - unique[0]) / dz).astype(int)
        if index.max() + 1 > 3 * len(unique):  # irregular spacing: stack in order instead
            index = np.searchsorted(unique, distance)
        origin = np.array(positions[int(np.argmin(distance))], float)
    else:
        row_dir, col_dir, normal = np.eye(3)
        dz = float(thickness or 1)
        index = np.arange(len(frames))
        origin = np.zeros(3)

    rows, cols = frames[0].shape
    vol = np.zeros((cols, rows, index.max() + 1), np.float32)
    filled = np.zeros(vol.shape[2], bool)
    for frame, k in zip(frames, index):
        if is_seg:
            vol[:, :, k] = np.maximum(vol[:, :, k], frame.T)
        elif not filled[k]:  # 4D data: keep the first volume
            vol[:, :, k] = frame.T
        filled[k] = True
    n_volumes = Counter(index.tolist()).most_common(1)[0][1]

    affine = np.eye(4)
    affine[:3, :3] = np.column_stack([row_dir * spacing[1], col_dir * spacing[0], normal * dz])
    affine[:3, 3] = origin
    affine[:2] *= -1  # DICOM LPS -> RAS

    def tag(keyword, sequence=None, fg_keyword=None):
        value = ds.get(keyword)
        if value in (None, "") and sequence:
            value = functional_group(ds, 0, sequence, fg_keyword or keyword)
        return value

    matrix = f"{cols} \N{MULTIPLICATION SIGN} {rows} \N{MULTIPLICATION SIGN} {vol.shape[2]}"
    if n_volumes > 1 and not is_seg:
        matrix += f" \N{MULTIPLICATION SIGN} {n_volumes} volumes"
    if has_geometry:
        matrix += f" ({('sagittal', 'coronal', 'axial')[int(np.argmax(np.abs(normal)))]})"
    sop_class = ds.SOPClassUID.name if "SOPClassUID" in ds else ""
    info = [
        ("Format", join(f"DICOM {fmt(ds.get('Modality'))}", sop_class)),
        ("Description", fmt(ds.get("SeriesDescription"))),
        ("Protocol", fmt(ds.get("ProtocolName"))),
        ("Sequence", join(fmt(ds.get("ScanningSequence")), fmt(ds.get("SequenceVariant")),
                          fmt(ds.get("SequenceName") or ds.get("PulseSequenceName")), fmt(ds.get("MRAcquisitionType")))),
        ("TR / TE / TI", timing(fmt(tag("RepetitionTime", "MRTimingAndRelatedParametersSequence")),
                                fmt(tag("EchoTime", "MREchoSequence", "EffectiveEchoTime")),
                                fmt(tag("InversionTime", "MRModifierSequence", "InversionTimes")),
                                fmt(tag("FlipAngle", "MRTimingAndRelatedParametersSequence")))),
        ("Contrast", fmt(ds.get("ContrastBolusAgent"))),
        ("Scanner", join(fmt(ds.get("Manufacturer")), fmt(ds.get("ManufacturerModelName")),
                         fmt(ds.get("MagneticFieldStrength"), " T"), sep=" ")),
        ("Matrix", matrix),
        ("Voxel size", f"{spacing[1]:.2f} \N{MULTIPLICATION SIGN} {spacing[0]:.2f} \N{MULTIPLICATION SIGN} {dz:.2f} mm"),
        ("Values", value_info(vol, ds.pixel_array.dtype)),
        ("Image type", fmt(ds.get("ImageType"))),
    ]
    if is_seg:
        info.append(("Segments", "; ".join(fmt(s.get("SegmentLabel")) for s in ds.get("SegmentSequence", [])) or MISSING))
    elif tag("DiffusionBValue", "MRDiffusionSequence") is not None:
        info.append(("b-value", fmt(tag("DiffusionBValue", "MRDiffusionSequence"))))

    vol, zooms = to_canonical(vol, affine)
    return vol, zooms, info


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

def display_range(vol: np.ndarray) -> tuple[float, float]:
    step = max(1, round((vol.size / 2e6) ** (1 / 3)))
    sample = vol[::step, ::step, ::step]
    sample = sample[np.isfinite(sample)]
    lo, hi = np.percentile(sample, [0.5, 99.5]) if sample.size else (0.0, 0.0)
    if hi <= lo:  # masks and other sparse images
        finite = vol[np.isfinite(vol)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    return lo, max(hi, lo + 1e-6)


def segmentation_center(vol: np.ndarray) -> np.ndarray | None:
    """Center of mass of the labelled voxels if the volume is a segmentation, otherwise None.

    A segmentation holds integer labels: apart from the background of 0 it has at most MAX_LABELS
    different values, all of them positive integers. An image without any label is not counted.
    Along every axis the center is moved to the nearest slice with a label, because the center of
    mass can lie in a gap between the labels.
    """
    step = max(1, round((vol.size / 2e6) ** (1 / 3)))
    for sample in (vol[::step, ::step, ::step], vol):  # the subsample rules out most images quickly
        labels = np.unique(sample[sample != 0])
        if labels.size > MAX_LABELS or not np.all(np.isfinite(labels) & (labels > 0) & (labels == np.rint(labels))):
            return None
    if labels.size == 0:
        return None
    labelled = vol != 0
    center = np.zeros(3, int)
    for axis in range(3):
        profile = labelled.sum(axis=tuple(a for a in range(3) if a != axis))
        slices = np.flatnonzero(profile)
        center_of_mass = np.average(np.arange(profile.size), weights=profile)
        center[axis] = slices[np.argmin(np.abs(slices - center_of_mass))]
    return center


def render_grid(vol: np.ndarray, zooms: np.ndarray, dpi: int, extra_slices: bool,
                center: np.ndarray | None = None) -> bytes:
    """Draw the slices (1 x 3, or 3 x 3 with extra slices) with matplotlib and return them as JPEG.

    The slices are taken around the center of the volume, or around center (voxel indices) if given.
    """
    rows = ROWS if extra_slices else ROWS[1:2]
    height = figure_height(extra_slices)
    fig = Figure(figsize=(CONTENT_W / 72, height / 72), dpi=dpi)
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(rows), 3, squeeze=False, gridspec_kw=dict(
        left=ROW_LABEL_W / CONTENT_W if extra_slices else 0, right=1, bottom=0, top=1 - COLUMN_TITLE_H / height,
        wspace=0.015, hspace=0.015))
    lo, hi = display_range(vol)
    for col, (title, axis) in enumerate(PLANES):
        horizontal, vertical = (a for a in range(3) if a != axis)
        n = vol.shape[axis]
        middle = (n - 1) / 2 if center is None else center[axis]
        for row, (label, offset) in enumerate(rows):
            k = int(np.clip(round(middle + offset * n), 0, n - 1))
            ax = axes[row, col]
            ax.imshow(np.take(vol, k, axis=axis).T, cmap="gray", vmin=lo, vmax=hi, origin="lower")
            ax.set_aspect(zooms[vertical] / zooms[horizontal], adjustable="datalim")
            ax.invert_xaxis()  # radiological convention: patient right / anterior on the left
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=9, pad=3)
            if col == 0 and extra_slices:
                ax.set_ylabel(label, fontsize=8, labelpad=2)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="jpeg", dpi=dpi, pil_kwargs={"quality": 85, "optimize": True})
    return buffer.getvalue()


def render_series(task: tuple[Series, int, bool]) -> tuple[bytes | None, list[tuple[str, str]], bool]:
    """Returns the slices as JPEG, the header information and whether the series is a segmentation."""
    series, dpi, extra_slices = task
    try:
        vol, zooms, info = load_nifti(series.files[0]) if series.kind == "nifti" else load_dicom(series.files)
        center = segmentation_center(vol)
        return render_grid(vol, zooms, dpi, extra_slices, center), info, center is not None
    except Exception as error:  # keep the page so that the series still appears in the review
        return None, [("Error", f"{type(error).__name__}: {error}")], False


# --------------------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------------------

def printable(text: str) -> str:
    """Restrict text to the characters of the built-in PDF fonts."""
    return text.encode("cp1252", "replace").decode("cp1252")


def truncate(text: str, font: str, size: float, width: float) -> str:
    if stringWidth(text, font, size) <= width:
        return text
    while text and stringWidth(text + "\N{HORIZONTAL ELLIPSIS}", font, size) > width:
        text = text[:-1]
    return text + "\N{HORIZONTAL ELLIPSIS}"


def wrap(text: str, font: str, size: float, width: float) -> list[str]:
    lines = [""]
    for char in text:
        if stringWidth(lines[-1] + char, font, size) > width:
            lines.append("")
        lines[-1] += char
    return lines


def draw_page(c: canvas.Canvas, number: int, total: int, series: Series, jpeg: bytes | None,
              info: list[tuple[str, str]], extra_slices: bool, segmentation: bool = False):
    form = c.acroForm
    figure_h = figure_height(extra_slices)
    y = page_height(extra_slices) - MARGIN

    # Original filename and page counter
    counter = f"{number} / {total}"
    c.setFont("Helvetica", 8)
    c.setFillColor(grey)
    c.drawRightString(PAGE_W - MARGIN, y - 9, counter)
    c.setFillColor(black)
    title_w = CONTENT_W - stringWidth(counter, "Helvetica", 8) - 12
    label = printable(series.label)
    for size in (10, 9, 8, 7):
        lines = wrap(label, "Helvetica-Bold", size, title_w)
        if len(lines) <= 2:
            break
    if len(lines) > 2:  # keep the end of the path, which holds the filename
        lines = lines[-2:]
        lines[0] = "\N{HORIZONTAL ELLIPSIS}" + lines[0][1:]
    c.setFont("Helvetica-Bold", size)
    for i, line in enumerate(lines):
        c.drawString(MARGIN, y - 9 - i * (size + 1.5), line)
    y -= TITLE_H

    # Slices
    y -= figure_h
    if jpeg:
        c.drawImage(ImageReader(io.BytesIO(jpeg)), MARGIN, y, CONTENT_W, figure_h)
    else:
        c.setFont("Helvetica", 11)
        c.drawCentredString(PAGE_W / 2, y + figure_h / 2, "This image could not be read.")

    # Multiple choice: large boxes with the label underneath. The boxes are square because reportlab
    # draws circular buttons correctly only at a size of 20 pt. SEG is selected in advance for segmentations.
    y -= GAP + MODALITY_H
    c.setStrokeColor(lightgrey)
    c.setLineWidth(0.75)
    c.rect(MARGIN, y, CONTENT_W, MODALITY_H)
    option_w = CONTENT_W / len(MODALITIES)
    radio = 28
    c.setFont("Helvetica", 10)
    for i, modality in enumerate(MODALITIES):
        center = MARGIN + (i + 0.5) * option_w
        form.radio(name=f"{FIELD_MODALITY}_{number:05d}", value=modality, selected=segmentation and modality == "SEG",
                   x=center - radio / 2,
                   y=y + MODALITY_H - 6 - radio, size=radio, buttonStyle="check", shape="square", borderWidth=1,
                   borderColor=black, fillColor=white, textColor=black, fieldFlags="noToggleToOff radio",
                   tooltip=modality)
        c.drawCentredString(center, y + 7, modality)

    # Notes
    y -= GAP + NOTES_H
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN + 6, y + NOTES_H - 12, "Notes")
    form.textfield(name=f"{FIELD_NOTES}_{number:05d}", x=MARGIN + LABEL_W, y=y, width=CONTENT_W - LABEL_W, height=NOTES_H,
                   fontName="Helvetica", fontSize=9, borderWidth=0.75, borderColor=lightgrey, fillColor=white,
                   textColor=black, fieldFlags="multiline", maxlen=None, tooltip="Notes")

    # Header information, two columns
    y -= GAP + INFO_H
    c.rect(MARGIN, y, CONTENT_W, INFO_H)
    per_column = 6
    column_w = CONTENT_W / 2
    key_w = 54
    for i, (key, value) in enumerate(info[: 2 * per_column]):
        full_width = len(info) == 1  # error message
        x = MARGIN + 6 + (i // per_column) * column_w
        line_y = y + INFO_H - 13 - (i % per_column) * 13
        c.setFont("Helvetica", 8)
        c.setFillColor(grey)
        c.drawString(x, line_y, key)
        c.setFillColor(black)
        value_w = (CONTENT_W if full_width else column_w) - key_w - 12
        c.drawString(x + key_w, line_y, truncate(printable(value), "Helvetica", 8, value_w))

    # Hidden fields carry the identity of the series to read_review_pdf.py
    hidden = dict(x=0, y=0, width=1, height=1, annotationFlags="hidden", fieldFlags="readOnly", maxlen=None,
                  fontName="Helvetica", fontSize=1, borderWidth=0)
    form.textfield(name=f"{FIELD_PATH}_{number:05d}", value=quote(series.path, safe="/ "), **hidden)
    if series.uid:
        form.textfield(name=f"{FIELD_UID}_{number:05d}", value=series.uid, **hidden)
    c.showPage()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", type=Path, help="dataset directory, searched recursively for NIfTI files and DICOM series")
    parser.add_argument("-o", "--output", type=Path, default=Path("modality_review.pdf"), help="output PDF (default: %(default)s)")
    parser.add_argument("--extra_slices", "--extra-slices", action="store_true",
                        help="show a 3 x 3 grid with the slices at center -/+ 1/4 of the field of view in addition "
                             "to the center slices")
    parser.add_argument("--max-pages", type=int, default=100,
                        help="split the output into parts of at most this many pages, which keeps saving fast; "
                             "0 writes a single PDF (default: %(default)s)")
    parser.add_argument("--dpi", type=int, default=150, help="resolution of the slice images (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="number of parallel processes (default: %(default)s)")
    args = parser.parse_args()

    root = args.dataset.resolve()
    if not root.is_dir():
        parser.error(f"{args.dataset} is not a directory")

    with ProcessPoolExecutor(args.workers) if args.workers > 1 else contextlib.nullcontext() as pool:
        print(f"Searching {root} ...", file=sys.stderr)
        all_series = find_series(root, pool)
        if not all_series:
            sys.exit(f"No NIfTI files or DICOM series found in {root}")
        total = len(all_series)
        print(f"Found {total} image series", file=sys.stderr)

        tasks = [(series, args.dpi, args.extra_slices) for series in all_series]
        results = pool.map(render_series, tasks) if pool else map(render_series, tasks)

        per_part = args.max_pages if args.max_pages > 0 else total
        n_parts = -(-total // per_part)
        failed = []
        for number, (series, (jpeg, info, segmentation)) in enumerate(zip(all_series, results), start=1):
            part = (number - 1) // per_part
            if (number - 1) % per_part == 0:
                output = args.output if n_parts == 1 else args.output.with_name(
                    f"{args.output.stem}_part{part + 1:02d}{args.output.suffix}")
                output.parent.mkdir(parents=True, exist_ok=True)
                c = canvas.Canvas(str(output), pagesize=(PAGE_W, page_height(args.extra_slices)))
                c.setTitle(f"MRI modality review: {root.name}")
            draw_page(c, number, total, series, jpeg, info, args.extra_slices, segmentation)
            if jpeg is None:
                failed.append(f"{series.label}: {info[0][1]}")
            print(f"[{number}/{total}] {series.label}", file=sys.stderr)
            if number % per_part == 0 or number == total:
                c.save()
                print(f"Wrote {output}", file=sys.stderr)

    if failed:
        print(f"\n{len(failed)} series could not be read (their pages show no image):", file=sys.stderr)
        for line in failed:
            print(f"  {line}", file=sys.stderr)


if __name__ == "__main__":
    main()

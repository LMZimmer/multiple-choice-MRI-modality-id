# Multiple-choice MRI modality identification

MRI sequences are often labeled incorrectly or not at all. This repository turns the check into a
form: one script renders every image series of a dataset into a fillable PDF, a clinician ticks the
modality on each page and saves the file, and a second script reads the selections back into a
table.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is required.

## 1. Generate the review PDF

```bash
python make_review_pdf.py /path/to/dataset -o modality_review.pdf
```

The dataset directory is searched recursively. Every NIfTI file (`.nii`, `.nii.gz`) and every DICOM
series (files grouped by folder and `SeriesInstanceUID`, with or without file extension) gets one
page with

- the path of the image relative to the dataset directory (the file for NIfTI, the series folder
  for DICOM),
- the axial, sagittal and coronal center slice,
- the choices T1, T1c, T2, FLAIR, ADC, DTI, PERF, SEG and OTHER,
- a field for notes,
- a box with header information.

Large PDFs are slow to save in a viewer, so a dataset with more than 100 series is split into
`modality_review_part01.pdf`, `modality_review_part02.pdf`, ... with 100 pages each. Page numbers
continue across the parts.

With `--extra_slices` the page shows a 3 × 3 grid instead: the same three planes at the center and
at center ∓ ¼ of the field of view. These pages are A4; the default pages are cut to their content
(about 60% of the height of an A4 page).

Images are reoriented so that the three columns show the anatomical planes regardless of the
acquisition plane, in radiological convention (patient right on the left of the image). For 4D data
the first volume is shown. Segmentations, including DICOM SEG objects, are shown as a mask on
their own extent. A series that cannot be read still gets a page, without an image.

An image counts as a segmentation if it holds integer labels: apart from the background of 0 it
has at most 64 different values, all of them positive integers. For these images the slices are
centered on the center of mass of the labels instead of the center of the image (with
`--extra_slices` the other two rows move along), and SEG is selected in advance. If the center of
mass lies in a gap between the labels, the nearest slice with a label is shown. An empty mask
has nothing to tell it apart from a blank image, so it gets the center slices and no selection.

| Option | Meaning |
| --- | --- |
| `-o`, `--output` | Output PDF (default `modality_review.pdf`) |
| `--extra_slices` | Show the 3 × 3 grid with the slices at center ∓ ¼ of the field of view |
| `--max-pages N` | Maximum number of pages per PDF (default 100); `0` writes a single PDF |
| `--dpi` | Resolution of the slice images (default 150) |
| `--workers` | Number of parallel processes |

## 2. Fill in the PDF

Open the PDF in a viewer that supports forms (Adobe Acrobat Reader, Preview, a web browser), click
one modality box per page, add notes where needed and **save** the file (each part, if there are
several). Do not use "Print to PDF",
which removes the form.

## 3. Read the selections

```bash
python read_review_pdf.py modality_review.pdf -o modality_mapping.csv
```

All parts can be passed at once (`python read_review_pdf.py modality_review_part*.pdf`). The CSV has
one row per image series:

| Column | Content |
| --- | --- |
| `path` | Path relative to the dataset directory, as printed on the page |
| `series_uid` | DICOM `SeriesInstanceUID`, empty for NIfTI. Distinguishes several series stored in the same folder |
| `modality` | Selected choice, empty if none was selected |
| `notes` | Content of the notes field |

Pages without a selection are listed on the terminal.

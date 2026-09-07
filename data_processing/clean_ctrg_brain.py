# coding: utf-8
"""
Brain CT Data Cleaning Pipeline for CTRG-Brain Dataset
========================================================

Implements the data cleaning pipeline described in MEPNet Section 4.1:
"We employ brain CT data cleaning pipeline (Zhang et al. 2023b) to filter
 redundant scans and ensure each sample contains 24 scans, aligning the
 data with medical standards."

Pipeline steps:
  1. Iterate over all patients in the raw CTRG-Brain-267K dataset
     (both 异常/abnormal and 正常/normal categories).
  2. For each patient, collect the axial CT slices (sorted .jpg files)
     and uniformly select 24 slices from the full stack. This removes
     redundant/repeated slice information that results from thinner
     slice acquisitions, standardising each sample to 24 slices.
  3. Copy the selected slices to a clean output directory with
     standardised filenames (scan_0001.jpg ... scan_0024.jpg).
  4. Copy the associated report file (report.txt) for abnormal patients.
  5. Generate a comprehensive metadata JSON and summary report.

Author: Adapted from the CTRG-Brain preprocessing used in
         Zhang et al. (2023b) and MEPNet (Zhang et al. 2025).
"""

import os
import sys
import json
import shutil
import argparse
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Default raw data root (CTRG-Brain-267K from the original dataset)
DEFAULT_RAW_ROOT = (
    r"D:\Code\ReportGeneration\MEPNet\Data"
    r"\CTRG-Brain-263\CTRG-Brain-267K"
)

# Default output root (placed inside the workspace for easy access)
DEFAULT_OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cleaned_data"
)

# Categories within the raw dataset
CATEGORIES = ["异常", "正常"]          # abnormal, normal

# Target number of slices per sample
TARGET_SLICES = 24

# ---------------------------------------------------------------------------
# Core cleaning logic
# ---------------------------------------------------------------------------


def select_uniform_slices(jpg_files, n_target=TARGET_SLICES):
    """
    Uniformly select *n_target* slices from a sorted list of JPG file paths.

    When the stack has **more** than *n_target* slices, this performs
    uniform down-sampling, filtering out redundant slices (the primary
    cleaning operation described in the paper).

    When the stack has **fewer** slices, we use nearest-neighbour
    up-sampling (repeating edge slices) so that downstream model code
    receives a fixed-size volume without breaking the batch dimension.

    Parameters
    ----------
    jpg_files : list of str
        Sorted (by filename) list of JPG paths for one patient.
    n_target : int
        Desired number of slices (default 24).

    Returns
    -------
    list of str
        Selected JPG paths, length == *n_target*.
    """
    n_available = len(jpg_files)
    if n_available == 0:
        return []

    # Linear-spaced indices in [0, n_available - 1]
    indices = np.linspace(0, n_available - 1, n_target)
    indices = np.round(indices).astype(int)

    return [jpg_files[i] for i in indices]


def process_patient(patient_id, patient_dir, output_dir, copy_images=True):
    """
    Process a single patient: select 24 uniformly-spaced slices
    and optionally copy them to the output directory.

    Parameters
    ----------
    patient_id : str
        Patient identifier (folder name).
    patient_dir : str
        Absolute path to the patient's raw data folder.
    output_dir : str
        Absolute path where cleaned patient data will be saved.
    copy_images : bool
        If True, copy selected slice images to the output directory.

    Returns
    -------
    dict
        Metadata dict for this patient with keys:
            patient_id, category (str),
            n_original (int), selected_files (list of str),
            has_report (bool), report_text (str or None)
    """
    # Collect and sort JPG slices
    all_files = sorted(os.listdir(patient_dir))
    jpg_files = sorted(
        [os.path.join(patient_dir, f)
         for f in all_files if f.lower().endswith(".jpg")]
    )
    n_original = len(jpg_files)

    # Select 24 uniform slices
    selected = select_uniform_slices(jpg_files, TARGET_SLICES)

    # Read report text if present
    report_path = os.path.join(patient_dir, "report.txt")
    has_report = os.path.isfile(report_path)
    report_text = None
    if has_report:
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read().strip()

    # --- Copy selected images to output ---
    if copy_images and selected:
        os.makedirs(output_dir, exist_ok=True)
        for idx, src_path in enumerate(selected, start=1):
            dst_name = f"scan_{idx:04d}.jpg"
            dst_path = os.path.join(output_dir, dst_name)
            shutil.copy2(src_path, dst_path)

        # Also copy the report file (if present)
        if has_report:
            shutil.copy2(report_path, os.path.join(output_dir, "report.txt"))

    # Metadata record
    record = {
        "patient_id": patient_id,
        "n_original": n_original,
        "n_selected": len(selected),
        "selected_files": [os.path.basename(p) for p in selected],
        "has_report": has_report,
        "report_text": report_text,
    }
    return record


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def clean_dataset(
    raw_root=DEFAULT_RAW_ROOT,
    output_root=DEFAULT_OUTPUT_ROOT,
    copy_images=True,
):
    """
    Run the full brain CT data cleaning pipeline.

    Parameters
    ----------
    raw_root : str
        Path to the raw CTRG-Brain-267K dataset.
    output_root : str
        Path where cleaned data will be written.
    copy_images : bool
        Whether to physically copy selected images.

    Returns
    -------
    dict
        Full metadata structure summarising the cleaned dataset.
    """
    print("=" * 70)
    print("  Brain CT Data Cleaning Pipeline")
    print(f"  Raw data   : {raw_root}")
    print(f"  Output     : {output_root}")
    print(f"  Target     : {TARGET_SLICES} slices per sample")
    print("=" * 70)

    os.makedirs(output_root, exist_ok=True)

    all_records = []
    stats = {"total_patients": 0, "total_original_scans": 0,
             "total_selected_scans": 0, "categories": {}}

    for cat in CATEGORIES:
        cat_path = os.path.join(raw_root, cat)
        if not os.path.isdir(cat_path):
            print(f"  [WARNING] Category folder not found: {cat_path}")
            continue

        # Collect patient directories (sort numerically)
        patient_ids = sorted(
            [d for d in os.listdir(cat_path)
             if os.path.isdir(os.path.join(cat_path, d))],
            key=lambda x: int(x)
        )

        cat_dir = os.path.join(output_root, cat)
        os.makedirs(cat_dir, exist_ok=True)

        cat_stats = {"patients": 0, "original_scans": 0,
                     "selected_scans": 0,
                     "n_original_distribution": {}}

        print(f"\n--- Processing category: {cat} ({len(patient_ids)} patients) ---")

        for pid in tqdm(patient_ids, desc=f"  [{cat}]"):
            patient_dir = os.path.join(cat_path, pid)
            patient_out = os.path.join(cat_dir, pid)

            record = process_patient(
                pid, patient_dir, patient_out, copy_images=copy_images
            )
            record["category"] = cat
            all_records.append(record)

            # Accumulate statistics
            n_orig = record["n_original"]
            cat_stats["patients"] += 1
            cat_stats["original_scans"] += n_orig
            cat_stats["selected_scans"] += record["n_selected"]
            key = str(n_orig)
            cat_stats["n_original_distribution"][key] = \
                cat_stats["n_original_distribution"].get(key, 0) + 1

        stats["total_patients"] += cat_stats["patients"]
        stats["total_original_scans"] += cat_stats["original_scans"]
        stats["total_selected_scans"] += cat_stats["selected_scans"]
        stats["categories"][cat] = cat_stats

    # --- Compute summary statistics ---
    n_before = stats["total_patients"]
    n_after = len([r for r in all_records if r["n_selected"] > 0])

    scan_before = stats["total_original_scans"]
    scan_after = stats["total_selected_scans"]
    scan_removed = scan_before - scan_after
    scan_reduction_pct = (scan_removed / scan_before * 100) if scan_before else 0

    report_count = sum(1 for r in all_records if r["has_report"])

    print("\n" + "=" * 70)
    print("  Cleaning Complete — Summary")
    print("=" * 70)
    print(f"  Patients processed      : {n_before}")
    print(f"  Patients with output    : {n_after}")
    print(f"  Patients with reports   : {report_count}")
    print(f"  Scans before cleaning   : {scan_before}")
    print(f"  Scans after cleaning    : {scan_after}  ({TARGET_SLICES}/pt)")
    print(f"  Redundant scans removed : {scan_removed}  ({scan_reduction_pct:.1f}%)")
    print("=" * 70)

    # Build final metadata
    metadata = {
        "pipeline": "CTRG-Brain Data Cleaning Pipeline",
        "reference": (
            "Zhang et al. 2023b; "
            "Section 4.1 of MEPNet (Zhang et al. 2025)"
        ),
        "target_slices_per_sample": TARGET_SLICES,
        "raw_root": raw_root,
        "output_root": output_root,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_patients": n_before,
            "total_patients_with_output": n_after,
            "total_scans_before_cleaning": scan_before,
            "total_scans_after_cleaning": scan_after,
            "redundant_scans_removed": scan_removed,
            "reduction_percentage": round(scan_reduction_pct, 2),
            "patients_with_reports": report_count,
        },
        "statistics_by_category": stats["categories"],
        "records": all_records,
    }

    # Write metadata JSON
    meta_path = os.path.join(output_root, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"\n  Metadata saved to: {meta_path}")

    return metadata


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Brain CT Data Cleaning Pipeline - "
                    "filter redundant scans, keep 24/sample"
    )
    parser.add_argument(
        "--raw-root", type=str, default=DEFAULT_RAW_ROOT,
        help="Raw data root"
    )
    parser.add_argument(
        "--output-root", type=str, default=DEFAULT_OUTPUT_ROOT,
        help="Cleaned data output root"
    )
    parser.add_argument(
        "--no-copy", action="store_true",
        help="Skip copying images (dry-run / stats only)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_dataset(
        raw_root=args.raw_root,
        output_root=args.output_root,
        copy_images=not args.no_copy,
    )

# coding: utf-8
"""
Extract ResNet101 visual features for MEPNet training.

Adapted from PCRL-MRG (Chauncey-Jheng/PCRL-MRG) resnet101_2048.py
for use with the MEPNet project.

Differences from the original:
  - Reads from the cleaned dataset (24 slices per sample, standardised names).
  - Saves features as .npy files (the format expected by MEPNet"s training code),
    not .npz as in the original.
  - Fixed a variable naming bug (fc/att were swapped in the original).
  - Added proper error handling and progress reporting.
  - Supports both CTRG-Brain and BCT-CHR datasets.

Output:
  For each patient {id}, two feature files are saved:
    fc/{id}.npy  shape (24, 2048)   — global pooled features
    att/{id}.npy  shape (24, 14, 14, 2048) — spatial attention features

Usage:
    python extract_visual_features.py
    python extract_visual_features.py --dataset ctrg
    python extract_visual_features.py --dataset bct-chr --raw-root /path/to/data

Requirements:
    pip install torch torchvision Pillow numpy tqdm
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as trn
from tqdm import tqdm

# Import the custom ResNet101
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resnet import resnet101


# ---------------------------------------------------------------------------
# Paths (adjust to match your local setup)
# ---------------------------------------------------------------------------

CLEANED_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cleaned_data"
)

CTRG_FEATURE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "CTRG_dataset"
)

WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "weights"
)

WEIGHT_PATH = os.path.join(WEIGHTS_DIR, "dan_CQ500_resnet101.pth")

# ---------------------------------------------------------------------------
# Preprocessing pipeline (same as PCRL-MRG)
# ---------------------------------------------------------------------------

preprocess = trn.Compose([
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                  [0.229, 0.224, 0.225]),  # ImageNet std
])

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(weight_path=None):
    """Load the CQ500-pretrained ResNet101 model."""
    net = resnet101(num_classes=2)
    net.eval()

    if weight_path is not None and os.path.isfile(weight_path):
        print(f"Loading pre-trained weights from: {weight_path}")
        state = torch.load(weight_path, map_location="cpu")
        net.load_state_dict(state, strict=False)
        print("Weights loaded successfully (strict=False).")
    else:
        print("WARNING: No pre-trained weights found.")
        print(f"  Expected at: {weight_path}")
        print("  Falling back to randomly initialised ResNet101.")
        print("  (Features will not contain CQ500 domain knowledge.)")

    return net


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(model, image_paths, att_size=14, device="cuda"):
    """
    Extract fc (2048-d) and att (14x14x2048) features for a list of images.

    Parameters
    ----------
    model : nn.Module
        ResNet101Features model in eval mode.
    image_paths : list of str
        Sorted paths to 24 JPG slices for one patient.
    att_size : int
        Spatial size for attention features (default 14).
    device : str
        "cuda" or "cpu".

    Returns
    -------
    fc_feat : np.ndarray  shape (24, 2048)
    att_feat : np.ndarray  shape (24, 14, 14, 2048)
    """
    n = len(image_paths)
    fc_feat = np.zeros([n, 2048], dtype=np.float32)
    att_feat = np.zeros([n, att_size, att_size, 2048], dtype=np.float32)

    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            try:
                # Load and preprocess
                img = Image.open(img_path).convert("RGB")
                I = preprocess(img).unsqueeze(0)  # [1, 3, H, W]
                I = I.to(device)

                # Forward through ResNet101
                tmp_fc, tmp_att = model(I, att_size)

                # tmp_fc:  [1, 2048]
                # tmp_att: [1, 2048, att_size, att_size]
                fc_feat[i] = tmp_fc.data.cpu().float().numpy()
                att_feat[i] = tmp_att.data.cpu().float().numpy().transpose(0, 2, 3, 1)

            except Exception as e:
                print(f"  [ERROR] Failed to process {img_path}: {e}")
                raise

    return fc_feat, att_feat


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ResNet101 visual features for MEPNet"
    )
    parser.add_argument(
        "--dataset", type=str, default="ctrg",
        choices=["ctrg", "bct-chr"],
        help="Dataset to process (default: ctrg)"
    )
    parser.add_argument(
        "--raw-root", type=str, default=None,
        help="Override cleaned data root"
    )
    parser.add_argument(
        "--feature-root", type=str, default=None,
        help="Override feature output root (default: CTRG_dataset/feature/)"
    )
    parser.add_argument(
        "--weight-path", type=str, default=WEIGHT_PATH,
        help=f"Path to dan_CQ500_resnet101.pth (default: {WEIGHT_PATH})"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda or cpu (default: auto-detect)"
    )
    parser.add_argument(
        "--att-size", type=int, default=14,
        help="Spatial attention feature size (default: 14)"
    )
    parser.add_argument(
        "--skip-att", action="store_true",
        help="Skip saving attention features (saves disk space)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("  ResNet101 Visual Feature Extraction")
    print("  Adapted from PCRL-MRG for MEPNet")
    print("=" * 65)

    # ---- Determine paths ----
    if args.raw_root is not None:
        data_root = args.raw_root
    else:
        data_root = CLEANED_DATA_ROOT

    if args.feature_root is not None:
        feat_root = args.feature_root
    elif args.dataset == "ctrg":
        # Place alongside CTRG_dataset
        feat_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "CTRG_dataset", "feature"
        )
    else:
        feat_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "BCT-CHR_dataset", "feature"
        )

    os.makedirs(os.path.join(feat_root, "fc"), exist_ok=True)
    os.makedirs(os.path.join(feat_root, "att"), exist_ok=True)

    print(f"  Data root   : {data_root}")
    print(f"  Feature root: {feat_root}")
    print(f"  Device      : {args.device}")

    # ---- Load model ----
    model = load_model(args.weight_path)
    model = model.to(args.device)
    model.eval()

    # ---- Iterate over patients ----
    # Only abnormal patients (with reports) are needed for training
    abnormal_path = os.path.join(data_root, "异常")
    if not os.path.isdir(abnormal_path):
        print(f"[ERROR] Abnormal data directory not found: {abnormal_path}")
        print("Run the data cleaning pipeline first.")
        sys.exit(1)

    patient_ids = sorted(
        [d for d in os.listdir(abnormal_path)
         if os.path.isdir(os.path.join(abnormal_path, d))],
        key=lambda x: int(x)
    )

    print(f"\n  Found {len(patient_ids)} patients to process.")
    print(f"  Total images: {len(patient_ids) * 24}")
    print(f"  Start time  : {datetime.now().strftime('%H:%M:%S')}")

    stats = {"success": 0, "skipped": 0, "error": 0}

    for pid in tqdm(patient_ids, desc="  Extracting"):
        fc_path = os.path.join(feat_root, "fc", f"{pid}.npy")
        att_path = os.path.join(feat_root, "att", f"{pid}.npy")

        # Skip if both features already exist
        if os.path.isfile(fc_path) and (args.skip_att or os.path.isfile(att_path)):
            stats["skipped"] += 1
            continue

        # Collect images (standardised names from cleaning pipeline)
        patient_dir = os.path.join(abnormal_path, pid)
        image_paths = sorted([
            os.path.join(patient_dir, f)
            for f in os.listdir(patient_dir)
            if f.lower().endswith(".jpg")
        ])

        if len(image_paths) == 0:
            print(f"  [WARN] No images for patient {pid}, skipping.")
            stats["error"] += 1
            continue

        # Extract features
        try:
            fc_feat, att_feat = extract_features(
                model, image_paths,
                att_size=args.att_size,
                device=args.device,
            )

            # Save fc features (always)
            np.save(fc_path, fc_feat)

            # Save att features (unless --skip-att)
            if not args.skip_att:
                np.save(att_path, att_feat)

            stats["success"] += 1

        except Exception as e:
            print(f"\n  [ERROR] Patient {pid}: {e}")
            stats["error"] += 1

    # ---- Summary ----
    print("\n" + "=" * 65)
    print("  Extraction Complete — Summary")
    print("=" * 65)
    print(f"  Successful    : {stats['success']}")
    print(f"  Already done  : {stats['skipped']}")
    print(f"  Errors        : {stats['error']}")
    print(f"  Total         : {sum(stats.values())}")
    print(f"  End time      : {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Feature dir   : {feat_root}")
    print(f"    fc/         : {len(patient_ids)} .npy files (24, 2048 each)")
    if not args.skip_att:
        print(f"    att/        : {len(patient_ids)} .npy files (24, 14, 14, 2048 each)")
    print("=" * 65)


if __name__ == "__main__":
    main()

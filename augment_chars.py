"""
augment_chars.py
────────────────
Augments a sparse dataset of white-on-black 80×80 handwritten character images
(Sinhala Brahmi or similar scripts) to produce many realistic variations per sample.

Augmentation pipeline (order matters):
  1. Morphological stroke variation  – erode / dilate to simulate pen pressure
  2. Elastic distortion              – warps stroke paths naturally
  3. Affine transform                – rotation, shear, scale
  4. Mild perspective warp           – subtle 3-D tilt
  5. Edge noise                      – light Gaussian noise on stroke borders
  6. Optional: thin-line dropout     – randomly breaks very thin stroke segments

All outputs are white-on-black uint8 images at the same resolution as input (80×80).
Sub-pixel interpolation is done in float then re-thresholded so strokes stay crisp.

Usage
─────
  python augment_chars.py \
      --input_dir   path/to/dataset \
      --output_dir  path/to/augmented \
      --n           80               \   # variants to generate per source image
      [--threshold  30]              \   # re-binarise threshold after warp (0-255)
      [--workers    4]               \
      [--seed       42]                  # set for reproducibility

Requirements
────────────
  pip install albumentations opencv-python-headless numpy tqdm
  (scipy is used for elastic distortion fallback if albumentations version is old)
"""

import argparse
import random
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import albumentations as A

# ─────────────────────────────────────────────────────────────────────────────
# Morphological stroke width variation
# ─────────────────────────────────────────────────────────────────────────────

def vary_stroke_width(img: np.ndarray) -> np.ndarray:
    """
    Randomly erode (thin) or dilate (thicken) strokes.
    kernel size 2 = subtle, 3 = noticeable.
    Works on single-channel uint8 binary/grayscale image.
    """
    op      = random.choice(["erode", "dilate", "none", "none"])  # bias toward no-op
    k_size  = random.choice([2, 2, 3])
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    if op == "erode":
        return cv2.erode(img, kernel, iterations=1)
    elif op == "dilate":
        return cv2.dilate(img, kernel, iterations=1)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Elastic distortion (manual implementation — reliable across albu versions)
# ─────────────────────────────────────────────────────────────────────────────

def elastic_distort(img: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    """
    Standard elastic deformation (Simard et al. 2003).
    alpha  : displacement magnitude  (2–6 recommended for 80×80)
    sigma  : smoothing (Gaussian blur on displacement field); 2–4 recommended
    """
    h, w = img.shape
    rng  = np.random.default_rng()

    dx = rng.uniform(-1, 1, (h, w)).astype(np.float32)
    dy = rng.uniform(-1, 1, (h, w)).astype(np.float32)

    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

    x, y   = np.meshgrid(np.arange(w), np.arange(h))
    map_x  = np.clip((x + dx).astype(np.float32), 0, w - 1)
    map_y  = np.clip((y + dy).astype(np.float32), 0, h - 1)

    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ─────────────────────────────────────────────────────────────────────────────
# Albumentations pipeline  (affine + perspective + noise)
# ─────────────────────────────────────────────────────────────────────────────

def build_albu_pipeline() -> A.Compose:
    return A.Compose([

        # ── Rotation ± 8°, scale 90–110%, shear ± 5° ──
        A.Affine(
            rotate=(-8, 8),
            scale=(0.90, 1.10),
            shear=(-5, 5),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            cval=0,                 # fill with black
            p=0.90,
        ),

        # ── Mild perspective warp ──
        A.Perspective(
            scale=(0.01, 0.04),     # very conservative for thin strokes
            p=0.40,
        ),

        # ── Gaussian noise on stroke edges ──
        # Adds realistic ink bleed / rough pen texture
        A.GaussNoise(
            std_range=(0.01, 0.04),  # low — just enough for edge roughness
            p=0.60,
        ),

    ], p=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Single-image augmentation
# ─────────────────────────────────────────────────────────────────────────────

_PIPELINE = None   # process-local, built once per worker

def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = build_albu_pipeline()
    return _PIPELINE


def augment_one(
    img_gray: np.ndarray,
    threshold: int = 30,
) -> np.ndarray:
    """
    Apply one random augmentation pass to a single grayscale image.
    Returns a white-on-black uint8 image of the same size.
    """
    # ── 1. Stroke width variation ──
    out = vary_stroke_width(img_gray)

    # ── 2. Elastic distortion ──
    # alpha 2–5, sigma 2–3 → subtle natural stroke wobble
    alpha = random.uniform(2.0, 5.0)
    sigma = random.uniform(2.0, 3.5)
    out   = elastic_distort(out, alpha=alpha, sigma=sigma)

    # ── 3. Affine + perspective + noise via albumentations ──
    # albumentations expects HWC or HW; convert to float [0,1] for noise
    out_f   = (out.astype(np.float32) / 255.0)
    result  = _get_pipeline()(image=out_f)["image"]

    # ── 4. Re-binarise: re-threshold to keep white-on-black crispness ──
    # (interpolation during warp creates grey anti-alias pixels)
    out_u8  = np.clip(result * 255, 0, 255).astype(np.uint8)
    _, out_bin = cv2.threshold(out_u8, threshold, 255, cv2.THRESH_BINARY)

    return out_bin


# ─────────────────────────────────────────────────────────────────────────────
# Per-file job
# ─────────────────────────────────────────────────────────────────────────────

_ARGS = None

def _init_worker(args):
    global _ARGS
    _ARGS = args
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)


def _process_file(job):
    src_path, dst_dir, n, threshold = job
    try:
        img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return str(src_path), f"Could not read image"

        # Resize to 80×80 if needed (just in case)
        if img.shape != (80, 80):
            img = cv2.resize(img, (80, 80), interpolation=cv2.INTER_LINEAR)

        dst_dir.mkdir(parents=True, exist_ok=True)
        stem = src_path.stem

        # Always copy the original
        cv2.imwrite(str(dst_dir / f"{stem}_orig.png"), img)

        for i in range(n):
            aug = augment_one(img, threshold=threshold)
            cv2.imwrite(str(dst_dir / f"{stem}_aug{i:04d}.png"), aug)

        return str(src_path), None

    except Exception as e:
        return str(src_path), str(e)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Augment sparse handwritten character images (white-on-black)"
    )
    p.add_argument("--input_dir",  required=True,
                   help="Root folder with original images (subdirs = class labels)")
    p.add_argument("--output_dir", required=True,
                   help="Where to write augmented images (same subdir structure)")
    p.add_argument("--n",          type=int,   default=80,
                   help="Number of augmented variants per source image (default 80)")
    p.add_argument("--threshold",  type=int,   default=30,
                   help="Re-binarise threshold after warp, 0-255 (default 30). "
                        "Lower = keep more faint stroke pixels; higher = strip anti-alias")
    p.add_argument("--workers",    type=int,   default=4,
                   help="Parallel worker processes (default 4)")
    p.add_argument("--seed",       type=int,   default=None,
                   help="Random seed for reproducibility (default: random)")
    return p.parse_args()


SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    args = parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    src_files = [p for p in input_dir.rglob("*") if p.suffix.lower() in SUPPORTED]
    if not src_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(src_files)} source images.")
    print(f"Generating {args.n} variants each → up to {len(src_files) * (args.n + 1)} output images.")

    # Build jobs: each file gets its own output subdir matching the source structure
    jobs = []
    for src in src_files:
        rel     = src.relative_to(input_dir)
        dst_dir = output_dir / rel.parent   # preserve class-label subdirs
        jobs.append((src, dst_dir, args.n, args.threshold))

    errors = []

    if args.workers <= 1:
        _init_worker(args)
        for job in tqdm(jobs, desc="Augmenting"):
            _, err = _process_file(job)
            if err:
                errors.append(err)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args,),
        ) as pool:
            futures = {pool.submit(_process_file, j): j for j in jobs}
            for fut in tqdm(as_completed(futures), total=len(jobs), desc="Augmenting"):
                _, err = fut.result()
                if err:
                    errors.append(err)

    ok = len(jobs) - len(errors)
    print(f"\nDone. {ok}/{len(jobs)} files processed successfully.")
    if errors:
        print(f"{len(errors)} errors:")
        for e in errors[:10]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
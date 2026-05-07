"""
stone_carve.py
──────────────
Converts a dataset of white-on-black 80×80 character images into
224×224 "stone-carved" images suitable for training VisFormer.

Workflow (mirrors the Photoshop steps):
  1. Threshold: isolate white pixels (value > 30% → 76/255)
  2. White letter  = original isolated letter (highlight side)
  3. Black letter  = inverted isolated letter (shadow — placed ON TOP)
  4. Combine layers: white copy first, then black copy offset on top
       - offset magnitude: random 2–4 px per image
       - offset direction: random quadrant (+x+y / +x-y / -x-y / -x+y)
       → black-on-top gives a deep-carved shadow look with edge highlight
  5. Random 224×224 crop from a full 2560×1440 texture (unique per image)
  6. Soft-light blend at random opacity 60–80% per image
  7. Output is already 224×224 — no separate resize step needed

Usage
─────
  python stone_carve.py \
      --input_dir  path/to/dataset \
      --output_dir path/to/output \
      --texture    tex1.jpg tex2.jpg tex3.jpg \
      [--threshold 76] [--size 224] [--workers 8]

Requirements: pip install Pillow numpy tqdm
"""

import argparse
import os
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Core effect
# ─────────────────────────────────────────────────────────────────────────────

def soft_light_blend(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """
    Photoshop 'Soft Light' blend formula (works on float arrays in [0, 1]).
    Handles both RGB and RGBA arrays; blending is done on RGB channels only.
    """
    # Work on float [0,1]
    b = base[..., :3].astype(np.float32) / 255.0
    o = overlay[..., :3].astype(np.float32) / 255.0

    # Soft-light formula (Pegtop / W3C variant)
    result = (1 - 2 * o) * b ** 2 + 2 * o * b

    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    # Re-attach alpha if present
    if base.shape[2] == 4:
        return np.dstack([result, base[..., 3]])
    return result


def apply_stone_carve(
    src_path: Path,
    dst_path: Path,
    texture_paths: list,     # list of Path/str to full-res texture files
    threshold: int = 76,     # ~30% of 255
    out_size: int = 224,
    add_blur: bool = True,
) -> None:
    """Process a single image."""

    # ── 1. Load source (grayscale or RGB, black background / white letter) ──
    src = Image.open(src_path).convert("L")          # grayscale
    src_arr = np.array(src)

    # ── 2. Isolate letter: mask where pixel > threshold ──
    mask = (src_arr > threshold).astype(np.uint8) * 255   # binary mask

    # White letter (original) – RGBA  ← highlight layer
    white_letter = np.zeros((*src_arr.shape, 4), dtype=np.uint8)
    white_letter[..., :3] = src_arr[..., np.newaxis].repeat(3, axis=2)  # gray→RGB
    white_letter[..., 3]  = mask

    # ── 3. Black letter (inverted) – goes ON TOP as shadow ──
    black_letter = white_letter.copy()
    black_letter[..., :3] = 0          # pure black where visible

    # ── 4. Random offset: magnitude 2–4 px, random quadrant ──
    magnitude  = random.randint(2, 4)
    sign_x     = random.choice([-1, 1])
    sign_y     = random.choice([-1, 1])
    dx         = sign_x * magnitude
    dy         = sign_y * magnitude

    h, w = src_arr.shape
    canvas = np.zeros((h, w, 4), dtype=np.uint8)   # transparent canvas

    def paste_rgba(dst, src_layer, dx=0, dy=0):
        """Alpha-composite src_layer onto dst with a pixel offset (handles negative offsets)."""
        src_h, src_w = src_layer.shape[:2]
        d_x0 = max(dx, 0);   d_y0 = max(dy, 0)
        d_x1 = min(dx + src_w, dst.shape[1])
        d_y1 = min(dy + src_h, dst.shape[0])
        s_x0 = d_x0 - dx;    s_y0 = d_y0 - dy
        s_x1 = s_x0 + (d_x1 - d_x0)
        s_y1 = s_y0 + (d_y1 - d_y0)

        if d_x1 <= d_x0 or d_y1 <= d_y0:
            return  # fully clipped

        src_rgb   = src_layer[s_y0:s_y1, s_x0:s_x1, :3].astype(np.float32)
        src_alpha = src_layer[s_y0:s_y1, s_x0:s_x1, 3:4].astype(np.float32) / 255.0
        dst_rgb   = dst[d_y0:d_y1, d_x0:d_x1, :3].astype(np.float32)
        dst_alpha = dst[d_y0:d_y1, d_x0:d_x1, 3:4].astype(np.float32) / 255.0

        out_alpha = src_alpha + dst_alpha * (1 - src_alpha)
        out_rgb   = np.where(
            out_alpha > 0,
            (src_rgb * src_alpha + dst_rgb * dst_alpha * (1 - src_alpha))
            / np.where(out_alpha > 0, out_alpha, 1),
            0,
        )
        dst[d_y0:d_y1, d_x0:d_x1, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        dst[d_y0:d_y1, d_x0:d_x1, 3]  = np.clip(out_alpha[..., 0] * 255, 0, 255).astype(np.uint8)

    # Randomly choose carving direction:
    # 50% dark carved-in look
    # 50% bright raised/carved look

    if random.random() < 0.5:
        # Dark carved-in appearance
        # White highlight underneath, black shadow on top
        paste_rgba(canvas, white_letter, dx=0,  dy=0)
        paste_rgba(canvas, black_letter, dx=dx, dy=dy)

    else:
        # Bright raised/carved appearance
        # Black shadow underneath, white highlight on top
        paste_rgba(canvas, black_letter, dx=0,  dy=0)
        paste_rgba(canvas, white_letter, dx=dx, dy=dy)

    letter_img = Image.fromarray(canvas, mode="RGBA")

    # ── 5. Resize carved letter smaller than canvas ──
    letter_size = 160

    letter_img = letter_img.resize((letter_size, letter_size), Image.LANCZOS)

    # Optional: subtle gaussian blur AFTER resize
    if add_blur:
        letter_img = letter_img.filter(ImageFilter.GaussianBlur(radius=0.6))

    # ── 6. Create centered 224×224 transparent canvas ──
    centered_canvas = Image.new("RGBA", (out_size, out_size), (0, 0, 0, 0))

    offset_x = (out_size - letter_size) // 2
    offset_y = (out_size - letter_size) // 2

    centered_canvas.paste(letter_img, (offset_x, offset_y), letter_img)

    letter_img = centered_canvas

    # ── 6. Random 224×224 crop from a randomly chosen full-res texture ──
    tex_path = random.choice(texture_paths)
    tex_img  = Image.open(tex_path).convert("RGB")
    max_x    = tex_img.width  - out_size
    max_y    = tex_img.height - out_size
    crop_x   = random.randint(0, max(max_x, 0))
    crop_y   = random.randint(0, max(max_y, 0))
    tex_crop = tex_img.crop((crop_x, crop_y, crop_x + out_size, crop_y + out_size))
    tex_arr  = np.array(tex_crop)

    # ── 7. Soft-light blend at random opacity 60–80% ──
    opacity    = random.uniform(0.60, 0.80)
    letter_arr = np.array(letter_img)           # RGBA
    alpha      = letter_arr[..., 3:4].astype(np.float32) / 255.0

    blended = soft_light_blend(
        np.dstack([tex_arr, np.full(tex_arr.shape[:2], 255, np.uint8)]),
        letter_arr,
    )[..., :3]

    final_rgb = (
        tex_arr.astype(np.float32) * (1 - alpha * opacity)
        + blended.astype(np.float32) * (alpha * opacity)
    )
    final_rgb = np.clip(final_rgb, 0, 255).astype(np.uint8)

    final_img = Image.fromarray(final_rgb, mode="RGB")

    # ── 8. Save ──
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    final_img.save(dst_path)



# ─────────────────────────────────────────────────────────────────────────────
# Worker (must be top-level for multiprocessing pickling)
# ─────────────────────────────────────────────────────────────────────────────

_TEXTURE_PATHS = None   # process-local list of Path objects
_ARGS          = None

def _init_worker(texture_paths, args):
    global _TEXTURE_PATHS, _ARGS
    _TEXTURE_PATHS = texture_paths
    _ARGS          = args

def _process_one(job):
    src, dst = job
    try:
        apply_stone_carve(
            src_path      = src,
            dst_path      = dst,
            texture_paths = _TEXTURE_PATHS,
            threshold     = _ARGS.threshold,
            out_size      = _ARGS.size,
        )
        return str(dst), None
    except Exception as e:
        return str(dst), str(e)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stone-carving augmentation for character datasets")
    p.add_argument("--input_dir",  required=True,  help="Root folder of original 80×80 images")
    p.add_argument("--output_dir", required=True,  help="Where to write augmented images")
    p.add_argument("--texture",    required=True,  nargs="+",
                   help="One or more stone/marble texture image paths (2560×1440 recommended)")
    p.add_argument("--threshold",  type=int,   default=76,   help="White isolation threshold 0-255 (default 76 ≈ 30%%)")
    p.add_argument("--size",       type=int,   default=224,  help="Output image size (default 224)")
    p.add_argument("--workers",    type=int,   default=4,    help="Parallel worker processes (default 4)")
    p.add_argument("--ext",        default="png",            help="Output file extension (default png)")
    return p.parse_args()


SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def main():
    args = parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Collect all image paths, preserving subdirectory structure
    src_files = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED
    ]
    if not src_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(src_files)} images.")

    # Build (src, dst) pairs
    jobs = []
    for src in src_files:
        rel  = src.relative_to(input_dir)
        dst  = output_dir / rel.with_suffix(f".{args.ext.lstrip('.')}")
        jobs.append((src, dst))

    # Collect texture paths (validate they exist)
    texture_paths = []
    for t in args.texture:
        p = Path(t)
        if not p.exists():
            print(f"Warning: texture not found, skipping: {t}")
        else:
            texture_paths.append(p)
    if not texture_paths:
        print("No valid textures found. Exiting.")
        return
    print(f"Using {len(texture_paths)} texture file(s). Crops are randomised per image.")

    errors = []

    if args.workers <= 1:
        # Single-process (easier to debug)
        _init_worker(texture_paths, args)
        for job in tqdm(jobs, desc="Processing"):
            _, err = _process_one(job)
            if err:
                errors.append((job[0], err))
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(texture_paths, args),
        ) as pool:
            futures = {pool.submit(_process_one, j): j for j in jobs}
            for fut in tqdm(as_completed(futures), total=len(jobs), desc="Processing"):
                _, err = fut.result()
                if err:
                    errors.append((futures[fut][0], err))

    print(f"\nDone. {len(jobs) - len(errors)}/{len(jobs)} images converted.")
    if errors:
        print(f"{len(errors)} errors:")
        for path, msg in errors[:10]:
            print(f"  {path}: {msg}")


if __name__ == "__main__":
    main()
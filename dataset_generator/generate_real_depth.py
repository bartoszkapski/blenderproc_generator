"""
Depth Distortion Post-Processing
=================================
Applies realistic depth camera distortion (noise, alignment, projection)
from the synthetic-rgbd-camera-model to depth_ground_truth images.

The "depth_perfect" output stays in the depth-camera frame. This avoids
introducing RGB-view occlusion holes into the ideal reference depth.

Run this AFTER generate_data.py, in the venv with open3d installed:

    source /home/hampthamanta/code_workspace/magisterka/venv_mgr/bin/activate
    python generate_real_depth.py --input <output_run_dir>

Or process all runs under the output directory:
    python generate_real_depth.py --input <output_root> --recursive
"""

import argparse
import os
import sys
import yaml

import cv2
import numpy as np


def fill_missing_depth_pixels(depth_img):
    """Fill zero-valued holes in an aligned uint16 depth image using nearest valid pixels."""
    if depth_img is None:
        return depth_img

    depth = depth_img.copy()
    mask_valid = depth > 0
    if mask_valid.all():
        return depth

    mask_invalid = (~mask_valid).astype("uint8")
    if mask_invalid.max() == 0:
        return depth

    _, labels = cv2.distanceTransformWithLabels(
        mask_invalid, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    nearest = labels.astype("int32") - 1

    valid_y, valid_x = np.where(mask_valid)
    if valid_x.size == 0:
        return depth

    valid_values = depth[valid_y, valid_x]
    fill_y, fill_x = np.where(mask_invalid)
    fill_indices = nearest[fill_y, fill_x]
    fill_indices = np.clip(fill_indices, 0, valid_values.size - 1)
    depth[fill_y, fill_x] = valid_values[fill_indices]
    return depth


def process_single_run(run_dir: str, processor, global_processed_dir: str):
    """Process one run directory: raw/ → global_processed_dir/modality/"""
    raw_dir = os.path.join(run_dir, "raw")
    
    # We use the run_dir name as a unique prefix to prevent overwriting images from different runs
    run_name = os.path.basename(run_dir)

    if not os.path.isdir(raw_dir):
        return

    rgb_out_dir = os.path.join(global_processed_dir, "rgb")
    depth_perf_out_dir = os.path.join(global_processed_dir, "depth_perfect")
    depth_noisy_out_dir = os.path.join(global_processed_dir, "depth_noisy")

    os.makedirs(rgb_out_dir, exist_ok=True)
    os.makedirs(depth_perf_out_dir, exist_ok=True)
    os.makedirs(depth_noisy_out_dir, exist_ok=True)

    depth_files = sorted(
        [
            f
            for f in os.listdir(raw_dir)
            if (f.startswith("depth_gt_") or f.startswith("depth_")) and f.endswith(".png")
        ]
    )
    if not depth_files:
        print(f"  [SKIP] No depth PNG files in {raw_dir}")
        return

    print(f"[Processing] {run_dir} ({len(depth_files)} images)")

    for depth_file in depth_files:
        depth_stem = depth_file[:-4]  # drop .png

        if depth_stem.startswith("depth_gt_"):
            sample_id = depth_stem[len("depth_gt_"):]
        elif depth_stem.startswith("depth_"):
            sample_id = depth_stem[len("depth_"):]
        else:
            print(f"  [WARN] Unsupported depth filename format: {depth_file}")
            continue

        rgb_file = f"rgb_{sample_id}.png"
        
        # Create a globally unique filename for this sample
        unique_id = f"{run_name}_{sample_id}"
        out_rgb_name = f"{unique_id}.png"
        out_perf_name = f"{unique_id}.png"
        out_noisy_name = f"{unique_id}.png"

        rgb_path = os.path.join(raw_dir, rgb_file)
        depth_path = os.path.join(raw_dir, depth_file)

        if not os.path.exists(rgb_path):
            print(f"  [WARN] RGB file not found: {rgb_file}")
            continue

        try:
            rgb_img = cv2.imread(rgb_path)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            if rgb_img is None or depth_img is None:
                print(f"  [WARN] Could not read images for pair {sample_id}")
                continue

            # Save the RGB image to the processed output.
            out_rgb_path = os.path.join(rgb_out_dir, out_rgb_name)
            if os.path.exists(out_rgb_path):
                 continue # Already processed this specific file
                 
            cv2.imwrite(out_rgb_path, rgb_img)

            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            # Keep the ideal reference depth aligned to the RGB frame.
            depth_perfect_aligned = processor.projection.get_aligned_depth_img(
                depth_img, rgb_img
            )
            depth_perfect_aligned = fill_missing_depth_pixels(depth_perfect_aligned)
            out_perfect_path = os.path.join(depth_perf_out_dir, out_perf_name)
            cv2.imwrite(out_perfect_path, depth_perfect_aligned)

            # Apply the noise model, then align the result to RGB as well.
            depth_noisy = processor.preprocessing.get_processed_image(
                depth_img, rgb_img
            )
            depth_noisy_aligned = processor.projection.get_aligned_depth_img(
                depth_noisy, rgb_img
            )
            out_noisy_path = os.path.join(depth_noisy_out_dir, out_noisy_name)
            cv2.imwrite(out_noisy_path, depth_noisy_aligned)

            print(f"  [OK] Processed sample {unique_id}")

        except Exception as e:
            print(f"  [ERROR] {depth_file}: {e}")


def find_run_dirs(root_dir: str) -> list:
    """
    Find all run directories (those containing raw/).
    Searches recursively.
    """
    runs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "raw" in dirnames:
            runs.append(dirpath)
    return sorted(runs)


def main():
    parser = argparse.ArgumentParser(
        description="Apply depth camera distortion to generated data"
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument(
        "--path",
        default=None,
        help="Optional output path override (replaces config['output'])",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    main_dir = cfg["main_folder"]
    if args.path:
        output_dir = args.path
    else:
        output_dir = os.path.join(main_dir, cfg["output"].lstrip("/"))

    camera_model_path = os.path.join(main_dir, "synthetic-rgbd-camera-model")
    params_path = os.path.join(camera_model_path, "params", "femto_mega.json")

    if camera_model_path not in sys.path:
        sys.path.insert(0, camera_model_path)
    try:
        from src.processor import ImageProcessor  # type: ignore[import-not-found]
    except ImportError as e:
        print(f"[ERROR] Could not import ImageProcessor: {e}")
        sys.exit(1)

    if not os.path.exists(params_path):
        print(f"[ERROR] Camera params not found: {params_path}")
        sys.exit(1)

    print("[INFO] Loading camera model...")
    processor = ImageProcessor(params_path=params_path)

    run_dirs = find_run_dirs(output_dir)
    if not run_dirs:
        print(f"[ERROR] No run directories found under {output_dir}")
        sys.exit(1)
        
    global_processed_dir = os.path.join(output_dir, "processed")
    print(f"[INFO] Global output directory for ML dataset: {global_processed_dir}")

    for run_dir in run_dirs:
        process_single_run(run_dir, processor, global_processed_dir)

    print("\n[DONE] All processing complete.")


if __name__ == "__main__":
    main()

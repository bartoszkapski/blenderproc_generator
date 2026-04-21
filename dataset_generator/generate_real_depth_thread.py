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
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import yaml

import cv2
import numpy as np


@dataclass(frozen=True)
class SampleJob:
    run_dir: str
    run_name: str
    sample_id: str
    depth_file: str


def resolve_camera_model_paths(main_dir: str) -> tuple[str, str]:
    """Find the actual camera model root and params file.

    Preferred layout:
    - <main_dir>/synthetic-rgbd-camera-model

    Legacy fallback (older repo layout):
    - <main_dir>/synthetic-rgbd-camera-model/synthetic-rgbd-camera-model
    """
    base_dir = Path(main_dir).expanduser().resolve() / "synthetic-rgbd-camera-model"
    # Prefer the current single-folder repository layout first.
    candidates = [base_dir, base_dir / "synthetic-rgbd-camera-model"]

    for camera_model_root in candidates:
        params_path = camera_model_root / "params" / "femto_mega.json"
        src_dir = camera_model_root / "src"
        if params_path.exists() and src_dir.is_dir():
            return str(camera_model_root), str(params_path)

    raise FileNotFoundError(
        "Could not find camera model files. Expected either "
        f"{base_dir}/(params/femto_mega.json, src/) or "
        f"{base_dir}/synthetic-rgbd-camera-model/(params/femto_mega.json, src/)."
    )


def add_camera_model_to_path(camera_model_root: str) -> str:
    """Add the camera model root directory to sys.path and return it."""
    camera_model_path = str(Path(camera_model_root).resolve())
    if camera_model_path not in sys.path:
        sys.path.insert(0, camera_model_path)
    return camera_model_path


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


def collect_sample_jobs(root_dir: str) -> list[SampleJob]:
    """Collect all depth/RGB pairs under every run directory."""
    jobs = []
    for run_dir in find_run_dirs(root_dir):
        raw_dir = os.path.join(run_dir, "raw")
        run_name = os.path.basename(run_dir)

        depth_files = sorted(
            [
                f
                for f in os.listdir(raw_dir)
                if (f.startswith("depth_gt_") or f.startswith("depth_")) and f.endswith(".png")
            ]
        )

        for depth_file in depth_files:
            depth_stem = depth_file[:-4]

            if depth_stem.startswith("depth_gt_"):
                sample_id = depth_stem[len("depth_gt_"):]
            elif depth_stem.startswith("depth_"):
                sample_id = depth_stem[len("depth_"):]
            else:
                continue

            jobs.append(
                SampleJob(
                    run_dir=run_dir,
                    run_name=run_name,
                    sample_id=sample_id,
                    depth_file=depth_file,
                )
            )

    return jobs


def build_worker_manifests(sample_jobs: list[SampleJob], workers: int, manifest_dir: str) -> list[str]:
    """Split samples across workers and write one text manifest per worker."""
    os.makedirs(manifest_dir, exist_ok=True)
    worker_buckets = [[] for _ in range(workers)]

    for index, sample_job in enumerate(sample_jobs):
        worker_buckets[index % workers].append(sample_job)

    manifest_paths = []
    for worker_index, bucket in enumerate(worker_buckets):
        manifest_path = os.path.join(manifest_dir, f"worker_{worker_index:03d}.txt")
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            for sample_job in bucket:
                manifest_file.write(
                    f"{sample_job.run_dir}\t{sample_job.run_name}\t{sample_job.sample_id}\t{sample_job.depth_file}\n"
                )
        manifest_paths.append(manifest_path)

    return manifest_paths


def process_worker_manifest(manifest_path: str, camera_model_root: str, params_path: str, global_processed_dir: str):
    """Process the samples assigned to one worker."""
    add_camera_model_to_path(camera_model_root)
    from src.processor import ImageProcessor  # type: ignore[import-not-found]

    rgb_out_dir = os.path.join(global_processed_dir, "rgb")
    depth_perf_out_dir = os.path.join(global_processed_dir, "depth_perfect")
    depth_noisy_out_dir = os.path.join(global_processed_dir, "depth_noisy")

    processor = ImageProcessor(params_path=params_path)

    processed = 0
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        lines = [line.strip() for line in manifest_file if line.strip()]

    if not lines:
        print(f"  [SKIP] Empty manifest: {manifest_path}")
        return 0

    print(f"[Worker] {os.path.basename(manifest_path)} -> {len(lines)} samples")

    for line in lines:
        try:
            run_dir, run_name, sample_id, depth_file = line.split("\t", 3)
        except ValueError:
            print(f"  [WARN] Malformed manifest line: {line}")
            continue

        raw_dir = os.path.join(run_dir, "raw")
        rgb_file = f"rgb_{sample_id}.png"
        unique_id = f"{run_name}_{sample_id}"

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

            out_rgb_path = os.path.join(rgb_out_dir, f"{unique_id}.png")
            if os.path.exists(out_rgb_path):
                continue

            cv2.imwrite(out_rgb_path, rgb_img)

            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            depth_perfect_aligned = processor.projection.get_aligned_depth_img(
                depth_img, rgb_img
            )
            depth_perfect_aligned = fill_missing_depth_pixels(depth_perfect_aligned)
            cv2.imwrite(os.path.join(depth_perf_out_dir, f"{unique_id}.png"), depth_perfect_aligned)

            depth_noisy = processor.preprocessing.get_processed_image(
                depth_img, rgb_img
            )
            depth_noisy_aligned = processor.projection.get_aligned_depth_img(
                depth_noisy, rgb_img
            )
            cv2.imwrite(os.path.join(depth_noisy_out_dir, f"{unique_id}.png"), depth_noisy_aligned)

            processed += 1
            print(f"  [OK] Processed sample {unique_id}")

        except Exception as e:
            print(f"  [ERROR] {depth_file}: {e}")

    return processed


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential processing",
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

    try:
        camera_model_path, params_path = resolve_camera_model_paths(main_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    add_camera_model_to_path(camera_model_path)
    try:
        from src.processor import ImageProcessor  # type: ignore[import-not-found]
    except ImportError as e:
        print(f"[ERROR] Could not import ImageProcessor: {e}")
        sys.exit(1)

    if args.workers < 1:
        print("[ERROR] --workers must be >= 1")
        sys.exit(1)

    workers = 1 if args.sequential else args.workers

    print("[INFO] Preparing camera model workers...")
    global_processed_dir = os.path.join(output_dir, "processed")
    print(f"[INFO] Global output directory for ML dataset: {global_processed_dir}")

    sample_jobs = collect_sample_jobs(output_dir)
    if not sample_jobs:
        print(f"[ERROR] No sample jobs found under {output_dir}")
        sys.exit(1)

    os.makedirs(os.path.join(global_processed_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(global_processed_dir, "depth_perfect"), exist_ok=True)
    os.makedirs(os.path.join(global_processed_dir, "depth_noisy"), exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="depth_postprocess_", dir=global_processed_dir) as temp_dir:
        manifest_paths = build_worker_manifests(sample_jobs, workers, temp_dir)

        if workers == 1:
            process_worker_manifest(manifest_paths[0], camera_model_path, params_path, global_processed_dir)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(process_worker_manifest, manifest_path, camera_model_path, params_path, global_processed_dir)
                    for manifest_path in manifest_paths
                ]

                for future in as_completed(futures):
                    future.result()

    print("\n[DONE] All processing complete.")


if __name__ == "__main__":
    main()

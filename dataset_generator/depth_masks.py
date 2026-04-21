#!/usr/bin/env python3
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _sort_key(value: str):
    parts = re.split(r"(\d+)", value)
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key)


def _resolve_processed_dir_from_config(config_path: Path) -> Path:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    main_folder = cfg.get("main_folder")
    output_rel = cfg.get("output")
    if not main_folder or not output_rel:
        raise ValueError(
            "Config must contain 'main_folder' and 'output' to resolve processed directory"
        )

    return (Path(str(main_folder)) / str(output_rel).lstrip("/") / "processed").resolve()


def _index_images(directory: Path) -> dict[str, Path]:
    if not directory.exists() or not directory.is_dir():
        return {}

    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXT
    ]

    index: dict[str, Path] = {}
    for path in sorted(files, key=lambda p: _sort_key(p.stem)):
        index.setdefault(path.stem, path)
    return index


def _collect_existing_output_ids(mask_dir: Path) -> set[str]:
    if not mask_dir.exists():
        return set()

    return {
        path.stem
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }


def _valid_depth_mask(depth_img: np.ndarray) -> np.ndarray:
    if np.issubdtype(depth_img.dtype, np.floating):
        return np.isfinite(depth_img) & (depth_img > 0)
    return depth_img > 0


def _build_mask_from_noisy(depth_noisy: np.ndarray) -> np.ndarray:
    return _valid_depth_mask(depth_noisy).astype(np.uint8)


@dataclass(frozen=True)
class MaskTaskResult:
    status: str
    sample_id: str
    error: str | None = None


def _recommended_worker_count() -> int:
    cpu_total = os.cpu_count() or 1
    return max(1, cpu_total - 2)


def _process_single_task(task: tuple[str, str, str, str]) -> MaskTaskResult:
    sample_id, perfect_path_str, noisy_path_str, out_path_str = task
    perfect_path = Path(perfect_path_str)
    noisy_path = Path(noisy_path_str)
    out_path = Path(out_path_str)

    try:
        depth_perfect = cv2.imread(str(perfect_path), cv2.IMREAD_UNCHANGED)
        depth_noisy = cv2.imread(str(noisy_path), cv2.IMREAD_UNCHANGED)

        if depth_perfect is None:
            raise FileNotFoundError(f"Failed to read depth_perfect image: {perfect_path}")
        if depth_noisy is None:
            raise FileNotFoundError(f"Failed to read depth_noisy image: {noisy_path}")

        if depth_perfect.shape != depth_noisy.shape:
            raise ValueError(
                f"Shape mismatch for ID {sample_id}: "
                f"depth_perfect={depth_perfect.shape}, depth_noisy={depth_noisy.shape}"
            )

        mask = _build_mask_from_noisy(depth_noisy)
        ok = cv2.imwrite(str(out_path), mask)
        if not ok:
            raise RuntimeError(f"Failed to write mask: {out_path}")

        return MaskTaskResult(status="processed", sample_id=sample_id)
    except Exception as exc:
        return MaskTaskResult(status="error", sample_id=sample_id, error=str(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate binary depth masks (0=no depth, 1=depth present) from depth_noisy."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to config.yaml used to resolve output/processed location",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Path to processed directory (contains depth_perfect/, depth_noisy/, depth_mask/)",
    )
    parser.add_argument(
        "--depth-perfect-dir",
        type=Path,
        default=None,
        help="Reference directory with depth_perfect images (used for shared ID discovery)",
    )
    parser.add_argument(
        "--depth-noisy-dir",
        type=Path,
        default=None,
        help="Input directory with depth_noisy images (mask source)",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Output directory for binary mask PNGs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already existing masks in output directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of IDs to process after filtering",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of CPU worker processes (default: max(1, cpu_count-2))",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Task chunksize per worker (default: auto)",
    )
    parser.add_argument(
        "--ordered-results",
        action="store_true",
        help="Preserve input order in progress reporting (slower than unordered mode)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = args.config.expanduser().resolve()

    if args.processed_dir is not None:
        processed_dir = args.processed_dir.expanduser().resolve()
    elif args.depth_perfect_dir is None or args.depth_noisy_dir is None or args.mask_dir is None:
        processed_dir = _resolve_processed_dir_from_config(config_path)
    else:
        processed_dir = None

    args.depth_perfect_dir = (
        args.depth_perfect_dir.expanduser().resolve()
        if args.depth_perfect_dir is not None
        else processed_dir / "depth_perfect"
    )
    args.depth_noisy_dir = (
        args.depth_noisy_dir.expanduser().resolve()
        if args.depth_noisy_dir is not None
        else processed_dir / "depth_noisy"
    )
    args.mask_dir = (
        args.mask_dir.expanduser().resolve()
        if args.mask_dir is not None
        else processed_dir / "depth_mask"
    )

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0")

    workers = args.workers if args.workers is not None else _recommended_worker_count()
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.chunksize is not None and args.chunksize < 1:
        raise ValueError("--chunksize must be >= 1")

    if not args.depth_perfect_dir.is_dir():
        raise FileNotFoundError(f"depth_perfect directory does not exist: {args.depth_perfect_dir}")
    if not args.depth_noisy_dir.is_dir():
        raise FileNotFoundError(f"depth_noisy directory does not exist: {args.depth_noisy_dir}")

    print(f"depth_perfect dir: {args.depth_perfect_dir}")
    print(f"depth_noisy dir:   {args.depth_noisy_dir}")
    print(f"depth_mask dir:    {args.mask_dir}")

    perfect_index = _index_images(args.depth_perfect_dir)
    noisy_index = _index_images(args.depth_noisy_dir)

    shared_ids = sorted(set(perfect_index.keys()) & set(noisy_index.keys()), key=_sort_key)
    missing_in_noisy = len(perfect_index) - len(shared_ids)
    missing_in_perfect = len(noisy_index) - len(shared_ids)

    if not shared_ids:
        raise FileNotFoundError(
            "No matching IDs between depth_perfect and depth_noisy directories."
        )

    args.mask_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = _collect_existing_output_ids(args.mask_dir)
    ids_to_process = shared_ids if args.overwrite else [sample_id for sample_id in shared_ids if sample_id not in existing_ids]

    if args.max_samples is not None:
        ids_to_process = ids_to_process[: args.max_samples]

    skipped_existing = len(shared_ids) - len(ids_to_process) if not args.overwrite else 0

    print(f"depth_perfect: {len(perfect_index)} file(s)")
    print(f"depth_noisy:   {len(noisy_index)} file(s)")
    print(f"matching IDs:  {len(shared_ids)}")
    if missing_in_noisy > 0:
        print(f"warning: {missing_in_noisy} ID(s) present in depth_perfect but missing in depth_noisy")
    if missing_in_perfect > 0:
        print(f"warning: {missing_in_perfect} ID(s) present in depth_noisy but missing in depth_perfect")

    if not args.overwrite:
        print(f"skip existing: {skipped_existing}")

    if not ids_to_process:
        print("Nothing to process. All matching IDs already have masks.")
        return

    workers = max(1, min(workers, len(ids_to_process)))
    chunksize = args.chunksize
    if chunksize is None:
        chunksize = max(1, len(ids_to_process) // max(1, workers * 4))

    print(f"workers:      {workers}")
    print(f"chunksize:    {chunksize}")
    print(f"ordered:      {args.ordered_results}")

    processed_count = 0
    error_count = 0
    errors: list[str] = []

    tasks = [
        (
            sample_id,
            str(perfect_index[sample_id]),
            str(noisy_index[sample_id]),
            str(args.mask_dir / f"{sample_id}.png"),
        )
        for sample_id in ids_to_process
    ]

    progress_step = max(1, len(tasks) // 20)

    if workers == 1:
        for idx, task in enumerate(tasks, start=1):
            result = _process_single_task(task)
            if result.status == "processed":
                processed_count += 1
                print(f"[{idx}/{len(tasks)}] OK -> {result.sample_id}")
            else:
                error_count += 1
                error_line = f"[{idx}/{len(tasks)}] ERROR {result.sample_id}: {result.error}"
                errors.append(error_line)
                print(error_line)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            if args.ordered_results:
                for idx, result in enumerate(executor.map(_process_single_task, tasks, chunksize=chunksize), start=1):
                    if result.status == "processed":
                        processed_count += 1
                    else:
                        error_count += 1
                        errors.append(f"ERROR {result.sample_id}: {result.error}")

                    if idx % progress_step == 0 or idx == len(tasks):
                        print(f"Progress: {idx}/{len(tasks)}")
            else:
                future_to_task = {
                    executor.submit(_process_single_task, task): task
                    for task in tasks
                }

                for idx, future in enumerate(as_completed(future_to_task), start=1):
                    sample_id = future_to_task[future][0]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = MaskTaskResult(status="error", sample_id=sample_id, error=f"Worker crashed: {exc}")

                    if result.status == "processed":
                        processed_count += 1
                    else:
                        error_count += 1
                        errors.append(f"ERROR {result.sample_id}: {result.error}")

                    if idx % progress_step == 0 or idx == len(tasks):
                        print(f"Progress: {idx}/{len(tasks)}")

    print("\n[SUMMARY]")
    print(f"Processed: {processed_count}")
    print(f"Errors:    {error_count}")
    print(f"Output:    {args.mask_dir}")
    if errors:
        print("\n[ERRORS]")
        for line in errors[:20]:
            print(line)
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more")


if __name__ == "__main__":
    main()

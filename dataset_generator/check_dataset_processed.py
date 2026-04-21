#!/usr/bin/env python3
"""Dataset integrity checker for dataset_processed.

Checks:
1. ID continuity across channels (rgb, depth_noisy, depth_mask, depth_perfect, depth_from_rgb).
2. Fully black images per channel.

For each check, user can choose to remove all affected sample IDs.
Removal deletes all channel files for a selected ID to keep dataset consistent.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import re
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
LEGACY_DATASET_DIR = PROJECT_ROOT / "dataset_processed"
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CHANNELS = ["rgb", "depth_noisy", "depth_mask", "depth_perfect", "depth_from_rgb"]


def resolve_processed_dir_from_config(config_path: Path) -> Path:
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


def natural_sort_key(value: str):
    parts = re.split(r"(\d+)", value)
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key)


def index_channel_files(channel_dir: Path) -> Dict[str, Path]:
    if not channel_dir.exists():
        return {}

    files = [
        path
        for path in channel_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXT
    ]

    index: Dict[str, Path] = {}
    for path in sorted(files, key=lambda p: natural_sort_key(p.stem)):
        index.setdefault(path.stem, path)
    return index


def union_ids(indexes: Dict[str, Dict[str, Path]]) -> List[str]:
    out = set()
    for channel_index in indexes.values():
        out.update(channel_index.keys())
    return sorted(out, key=natural_sort_key)


def find_missing_cases(indexes: Dict[str, Dict[str, Path]], channels: Sequence[str]) -> Dict[str, List[str]]:
    ids = union_ids(indexes)
    missing: Dict[str, List[str]] = {}

    for sample_id in ids:
        missing_channels = [channel for channel in channels if sample_id not in indexes[channel]]
        if missing_channels:
            missing[sample_id] = missing_channels

    return missing


def is_fully_black(img: np.ndarray) -> bool:
    if img is None:
        return True

    if np.issubdtype(img.dtype, np.floating):
        finite = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.max(finite)) <= 0.0

    return int(np.max(img)) == 0


def _black_scan_worker(task: tuple[str, str, str]) -> tuple[str, str] | None:
    channel, sample_id, path_str = task
    img = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
    if is_fully_black(img):
        return channel, sample_id
    return None


def black_pixel_ratio(img: np.ndarray | None) -> float:
    """Return ratio of black pixels in [0, 1]."""
    if img is None:
        return 1.0

    if img.ndim == 2:
        total = img.size
        if total == 0:
            return 1.0
        black = np.count_nonzero(img == 0)
        return float(black) / float(total)

    if img.ndim == 3:
        total = img.shape[0] * img.shape[1]
        if total == 0:
            return 1.0
        # Treat a pixel as black only if all channels are exactly zero.
        black = np.count_nonzero(np.all(img == 0, axis=2))
        return float(black) / float(total)

    return 1.0


def find_black_ratio_cases(
    indexes: Dict[str, Dict[str, Path]],
    channel: str,
    threshold_ratio: float,
) -> Dict[str, float]:
    """Find IDs where black-pixel ratio in selected channel exceeds threshold."""
    if channel not in indexes:
        return {}

    affected: Dict[str, float] = {}
    for sample_id, path in indexes[channel].items():
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        ratio = black_pixel_ratio(img)
        if ratio > threshold_ratio:
            affected[sample_id] = ratio

    return dict(sorted(affected.items(), key=lambda item: natural_sort_key(item[0])))


def print_black_ratio_report(channel: str, threshold_ratio: float, affected: Dict[str, float]) -> List[str]:
    threshold_pct = 100.0 * threshold_ratio
    print(f"\n[Black Ratio] Channel '{channel}', threshold > {threshold_pct:.4f}%")

    if not affected:
        print("No IDs exceed black-pixel threshold.")
        return []

    print(f"IDs above threshold: {len(affected)}")
    for sample_id, ratio in affected.items():
        print(f"  {sample_id}: {100.0 * ratio:.4f}% black pixels")
    return list(affected.keys())


def find_black_cases(
    indexes: Dict[str, Dict[str, Path]],
    channels: Sequence[str],
    workers: int,
    chunksize: int,
) -> Dict[str, List[str]]:
    black_per_channel: Dict[str, List[str]] = {channel: [] for channel in channels}

    tasks: List[tuple[str, str, str]] = []
    for channel in channels:
        for sample_id, path in indexes[channel].items():
            tasks.append((channel, sample_id, str(path)))

    if workers <= 1:
        for task in tasks:
            hit = _black_scan_worker(task)
            if hit is not None:
                channel, sample_id = hit
                black_per_channel[channel].append(sample_id)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for hit in executor.map(_black_scan_worker, tasks, chunksize=chunksize):
                if hit is None:
                    continue
                channel, sample_id = hit
                black_per_channel[channel].append(sample_id)

    for channel in channels:
        black_per_channel[channel].sort(key=natural_sort_key)

    return black_per_channel


def ask_yes_no(question: str, auto_yes: bool = False) -> bool:
    if auto_yes:
        print(f"{question} -> yes (auto)")
        return True

    answer = input(f"{question} [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def delete_sample_ids(dataset_dir: Path, channels: Sequence[str], sample_ids: Iterable[str], dry_run: bool = False) -> int:
    sample_ids = list(sample_ids)
    deleted_files = 0

    for sample_id in sample_ids:
        for channel in channels:
            channel_dir = dataset_dir / channel
            if not channel_dir.exists():
                continue

            for ext in SUPPORTED_EXT:
                path = channel_dir / f"{sample_id}{ext}"
                if path.exists():
                    if dry_run:
                        print(f"[DRY-RUN] delete {path}")
                    else:
                        path.unlink()
                    deleted_files += 1

    return deleted_files


def print_missing_report(missing_cases: Dict[str, List[str]], channels: Sequence[str]) -> None:
    if not missing_cases:
        print("\n[Continuity] No missing counterparts across channels.")
        return

    print("\n[Continuity] Missing counterpart report")
    print(f"IDs with at least one missing channel: {len(missing_cases)}")

    missing_by_channel = {channel: 0 for channel in channels}
    for missing_channels in missing_cases.values():
        for channel in missing_channels:
            missing_by_channel[channel] += 1

    for channel in channels:
        print(f"  Missing in {channel}: {missing_by_channel[channel]} ID(s)")

    print("Affected IDs and missing channels:")
    for sample_id in sorted(missing_cases.keys(), key=natural_sort_key):
        print(f"  {sample_id}: missing -> {', '.join(missing_cases[sample_id])}")


def print_black_report(black_per_channel: Dict[str, List[str]], channels: Sequence[str]) -> List[str]:
    black_union = sorted({sid for channel in channels for sid in black_per_channel[channel]}, key=natural_sort_key)

    print("\n[Black Images] Fully black image report")
    for channel in channels:
        ids = black_per_channel[channel]
        print(f"  {channel}: {len(ids)} fully black image(s)")
        if ids:
            print(f"    IDs: {', '.join(ids)}")

    print(f"IDs with at least one fully black image: {len(black_union)}")
    return black_union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check and optionally clean dataset_processed integrity.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to config.yaml used to resolve output/processed location",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Path to processed dataset root (default: resolved from config.yaml)",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Channels to check (default: auto-detect existing channels from standard set).",
    )
    parser.add_argument("--yes-delete-missing", action="store_true", help="Auto-confirm deletion of IDs with missing channels.")
    parser.add_argument("--yes-delete-black", action="store_true", help="Auto-confirm deletion of IDs with fully black images.")
    parser.add_argument(
        "--black-pct-channel",
        type=str,
        default=None,
        help="Optional channel name for black-pixel percentage filtering (e.g. depth_perfect).",
    )
    parser.add_argument(
        "--black-pct-threshold",
        type=float,
        default=None,
        help="Optional threshold percentage (0-100). IDs with black%% > threshold are selected.",
    )
    parser.add_argument(
        "--yes-delete-black-pct",
        action="store_true",
        help="Auto-confirm deletion of IDs selected by --black-pct-channel/--black-pct-threshold.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without removing files.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="CPU worker processes for black-image scan (use 1 to disable multiprocessing).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=64,
        help="Task chunk size passed to ProcessPoolExecutor.map during black-image scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir.expanduser().resolve()
    else:
        try:
            dataset_dir = resolve_processed_dir_from_config(args.config.expanduser().resolve())
        except Exception as exc:
            dataset_dir = LEGACY_DATASET_DIR.expanduser().resolve()
            print(
                f"WARNING: could not resolve processed directory from config ({exc}). "
                f"Falling back to: {dataset_dir}"
            )

    if args.channels:
        channels = list(dict.fromkeys(args.channels))
    else:
        channels = [channel for channel in DEFAULT_CHANNELS if (dataset_dir / channel).exists()]
        if not channels:
            channels = DEFAULT_CHANNELS.copy()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    workers = max(1, int(args.workers))
    chunksize = max(1, int(args.chunksize))

    if (args.black_pct_channel is None) != (args.black_pct_threshold is None):
        raise ValueError("Use --black-pct-channel and --black-pct-threshold together.")

    if args.black_pct_threshold is not None and not (0.0 <= args.black_pct_threshold <= 100.0):
        raise ValueError("--black-pct-threshold must be in range [0, 100].")

    print(f"Dataset root: {dataset_dir}")
    print(f"Channels: {', '.join(channels)}")
    print(f"Black scan workers: {workers}, chunksize: {chunksize}")

    for channel in channels:
        channel_dir = dataset_dir / channel
        if not channel_dir.exists():
            print(f"WARNING: channel directory missing: {channel_dir}")

    # Pass 1: continuity check
    indexes = {channel: index_channel_files(dataset_dir / channel) for channel in channels}
    missing_cases = find_missing_cases(indexes, channels)
    print_missing_report(missing_cases, channels)

    if missing_cases:
        should_delete_missing = ask_yes_no(
            "Delete all IDs with missing channel counterparts?",
            auto_yes=args.yes_delete_missing,
        )
        if should_delete_missing:
            removed = delete_sample_ids(
                dataset_dir=dataset_dir,
                channels=channels,
                sample_ids=missing_cases.keys(),
                dry_run=args.dry_run,
            )
            print(f"Removed files from missing-ID cleanup: {removed}")

    # Refresh index after potential deletion
    indexes = {channel: index_channel_files(dataset_dir / channel) for channel in channels}

    # Pass 2: fully black images check
    black_per_channel = find_black_cases(
        indexes=indexes,
        channels=channels,
        workers=workers,
        chunksize=chunksize,
    )
    black_union = print_black_report(black_per_channel, channels)

    if black_union:
        should_delete_black = ask_yes_no(
            "Delete all IDs that contain at least one fully black image?",
            auto_yes=args.yes_delete_black,
        )
        if should_delete_black:
            removed = delete_sample_ids(
                dataset_dir=dataset_dir,
                channels=channels,
                sample_ids=black_union,
                dry_run=args.dry_run,
            )
            print(f"Removed files from black-image cleanup: {removed}")

    # Pass 3: black-pixel percentage threshold in selected channel
    if args.black_pct_channel is not None and args.black_pct_threshold is not None:
        if args.black_pct_channel not in channels:
            raise ValueError(
                f"--black-pct-channel '{args.black_pct_channel}' is not in selected channels: {channels}"
            )

        # Refresh index in case previous pass removed files.
        indexes = {channel: index_channel_files(dataset_dir / channel) for channel in channels}

        threshold_ratio = float(args.black_pct_threshold) / 100.0
        affected_ratio_cases = find_black_ratio_cases(
            indexes=indexes,
            channel=args.black_pct_channel,
            threshold_ratio=threshold_ratio,
        )
        affected_ids = print_black_ratio_report(
            channel=args.black_pct_channel,
            threshold_ratio=threshold_ratio,
            affected=affected_ratio_cases,
        )

        if affected_ids:
            should_delete_black_pct = ask_yes_no(
                f"Delete all IDs with black-pixel ratio above {args.black_pct_threshold}% in {args.black_pct_channel}?",
                auto_yes=args.yes_delete_black_pct,
            )
            if should_delete_black_pct:
                removed = delete_sample_ids(
                    dataset_dir=dataset_dir,
                    channels=channels,
                    sample_ids=affected_ids,
                    dry_run=args.dry_run,
                )
                print(f"Removed files from black-ratio cleanup: {removed}")

    print("\nDone.")


if __name__ == "__main__":
    main()

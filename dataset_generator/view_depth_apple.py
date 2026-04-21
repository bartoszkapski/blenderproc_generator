"""Dataset viewer for data_generator_B.

Synthetic layout:
    - default: four synchronized channels (2x2 grid)
            rgb, depth_noisy, depth_perfect, depth_from_rgb
    - with --depth: depth-only triplet
            depth_perfect, depth_from_rgb, depth_from_rgb_apple

Missing channels/files are shown as black panels with a "missing" label.

Usage:
  python view_depth.py
  python view_depth.py output/processed
  python view_depth.py output/processed --colormap
    python view_depth.py output/processed --depth
"""

import argparse
from pathlib import Path
import re
import shutil

import cv2
import numpy as np


SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

PANEL_WIDTH = 640
PANEL_HEIGHT = 480

# Hardcoded source for the "Depth From RGB" panel in five-channel mode.
# Allowed: "depth_from_rgb" or "depth_from_rgb_apple"
DEPTH_FROM_RGB_CHANNEL_SOURCE = "depth_from_rgb_apple"


def resolve_channel_defs(use_depth_triplet: bool) -> list[tuple[str, str, str]]:
    """Return channel definitions as tuples: (label, display_key, source_key)."""
    if DEPTH_FROM_RGB_CHANNEL_SOURCE not in {"depth_from_rgb", "depth_from_rgb_apple"}:
        raise ValueError(
            "DEPTH_FROM_RGB_CHANNEL_SOURCE must be 'depth_from_rgb' or 'depth_from_rgb_apple'"
        )

    if not use_depth_triplet:
        return [
            ("RGB", "rgb", "rgb"),
            ("Depth Noisy", "depth_noisy", "depth_noisy"),
            ("Depth Perfect", "depth_perfect", "depth_perfect"),
            (
                f"Depth From RGB [{DEPTH_FROM_RGB_CHANNEL_SOURCE}]",
                "depth_from_rgb_display",
                DEPTH_FROM_RGB_CHANNEL_SOURCE,
            ),
        ]

    return [
        ("Depth Perfect", "depth_perfect", "depth_perfect"),
        ("Depth From RGB", "depth_from_rgb", "depth_from_rgb"),
        ("Depth From RGB Apple", "depth_from_rgb_apple", "depth_from_rgb_apple"),
    ]


def natural_sort_key(value: str):
    parts = re.split(r"(\d+)", value)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def index_images(directory: Path) -> dict[str, Path]:
    if not directory.exists() or not directory.is_dir():
        return {}

    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXT
    ]

    index: dict[str, Path] = {}
    for path in sorted(candidates, key=lambda p: natural_sort_key(p.stem)):
        index.setdefault(path.stem, path)

    return index


def recompute_sample_ids(channel_indexes: dict[str, dict[str, Path]]) -> list[str]:
    all_ids = set()
    for channel_index in channel_indexes.values():
        all_ids.update(channel_index.keys())
    return sorted(all_ids, key=natural_sort_key)


def delete_current_sample_id(sample_id: str, channel_indexes: dict[str, dict[str, Path]]) -> int:
    deleted_count = 0
    for channel_key in channel_indexes:
        path = channel_indexes[channel_key].pop(sample_id, None)
        if path is None:
            continue
        if path.exists():
            path.unlink()
            deleted_count += 1
    return deleted_count


def move_current_sample_to_correction(
    sample_id: str,
    channel_indexes: dict[str, dict[str, Path]],
    channel_sources: dict[str, str],
    correction_root: Path,
) -> int:
    moved_count = 0

    for channel_key in channel_indexes:
        src_path = channel_indexes[channel_key].get(sample_id)
        if src_path is None or not src_path.exists():
            channel_indexes[channel_key].pop(sample_id, None)
            continue

        source_key = channel_sources[channel_key]
        dest_dir = correction_root / source_key
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / src_path.name
        if dest_path.exists():
            dest_path.unlink()

        shutil.move(str(src_path), str(dest_path))
        channel_indexes[channel_key].pop(sample_id, None)
        moved_count += 1

    return moved_count


def safe_imread(path: Path | None, flags: int):
    if path is None:
        return None
    if not path.exists():
        return None
    return cv2.imread(str(path), flags)


def _channel_read_flags(channel_key: str) -> int:
    return cv2.IMREAD_COLOR if channel_key == "rgb" else cv2.IMREAD_UNCHANGED


def load_sample(sample_id: str, channel_indexes: dict[str, dict[str, Path]]) -> dict[str, np.ndarray | None]:
    return {
        channel_key: safe_imread(channel_index.get(sample_id), _channel_read_flags(channel_key))
        for channel_key, channel_index in channel_indexes.items()
    }


def resize_panel(image, width=PANEL_WIDTH, height=PANEL_HEIGHT):
    if image is None:
        return np.zeros((height, width, 3), dtype=np.uint8)

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    aspect = w / h
    if aspect > width / height:
        new_w = width
        new_h = int(width / aspect)
    else:
        new_h = height
        new_w = int(height * aspect)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    panel = np.zeros((height, width, 3), dtype=np.uint8)
    y_offset = (height - new_h) // 2
    x_offset = (width - new_w) // 2
    panel[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return panel


def _metric_depth_from_raw(depth: np.ndarray | None) -> np.ndarray | None:
    """Return metric depth in meters for float/uint16 maps, otherwise None."""
    if depth is None or depth.ndim != 2:
        return None

    if np.issubdtype(depth.dtype, np.floating):
        return np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if depth.dtype == np.uint8:
        # Legacy relative-depth exports are not metric; keep dedicated display path.
        return None

    return depth.astype(np.float32) / 1000.0


def _shared_depth_range_for_sample(raw_channels: dict[str, np.ndarray | None]) -> tuple[float | None, float | None]:
    """Compute shared near/far for all metric depth channels in current sample."""
    near_m: float | None = None
    far_m: float | None = None

    for channel_key, image in raw_channels.items():
        if not channel_key.startswith("depth_") or channel_key == "depth_mask":
            continue

        depth_m = _metric_depth_from_raw(image)
        if depth_m is None:
            continue

        valid = np.isfinite(depth_m) & (depth_m > 0)
        if not np.any(valid):
            continue

        channel_near = float(np.min(depth_m[valid]))
        channel_far = float(np.max(depth_m[valid]))

        near_m = channel_near if near_m is None else min(near_m, channel_near)
        far_m = channel_far if far_m is None else max(far_m, channel_far)

    return near_m, far_m


def normalize_depth_for_display(depth, use_colormap=False, near_m=None, far_m=None):
    if depth is None:
        return None
    if depth.ndim != 2:
        return None

    if depth.dtype == np.uint8:
        # Legacy relative-depth exports were saved as 8-bit; keep direct display.
        depth_display = depth.astype(np.uint8)
        if use_colormap:
            return cv2.applyColorMap(depth_display, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)

    depth_float = _metric_depth_from_raw(depth)
    if depth_float is None:
        return None

    valid = np.isfinite(depth_float) & (depth_float > 0)
    if not np.any(valid):
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    if near_m is None or far_m is None:
        near_m = float(np.min(depth_float[valid]))
        far_m = float(np.max(depth_float[valid]))

    if far_m <= near_m:
        depth_display = np.zeros(depth_float.shape, dtype=np.uint8)
        depth_display[valid] = 255
    else:
        depth_norm = np.zeros(depth_float.shape, dtype=np.float32)
        depth_norm[valid] = (depth_float[valid] - float(near_m)) / (float(far_m) - float(near_m))
        depth_norm = np.clip(depth_norm, 0.0, 1.0)
        depth_display = (255 * depth_norm).astype(np.uint8)

    depth_display[~valid] = 0

    if use_colormap:
        colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        return colored

    return cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)


def mask_to_display(mask):
    if mask is None:
        return None

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask_bin = (mask > 0).astype(np.uint8) * 255
    return cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)


def grayscale_to_display(image):
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def prepare_display_channels(raw_channels: dict[str, np.ndarray | None], use_colormap=False) -> dict[str, np.ndarray | None]:
    near_m, far_m = _shared_depth_range_for_sample(raw_channels)

    display_channels: dict[str, np.ndarray | None] = {}
    for channel_key, image in raw_channels.items():
        if channel_key.startswith("depth_") and channel_key != "depth_mask":
            display_channels[channel_key] = normalize_depth_for_display(
                image,
                use_colormap=use_colormap,
                near_m=near_m,
                far_m=far_m,
            )
        elif channel_key == "depth_mask":
            display_channels[channel_key] = mask_to_display(image)
        elif channel_key == "rgb":
            display_channels[channel_key] = image
        else:
            display_channels[channel_key] = grayscale_to_display(image)
    return display_channels


def render_labeled_panel(image: np.ndarray | None, label: str) -> np.ndarray:
    panel = resize_panel(image)
    cv2.putText(panel, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    if image is None:
        cv2.putText(panel, "missing", (10, PANEL_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    return panel


def render_control_hint(display: np.ndarray) -> None:
    hint_text = "Arrows: next/prev | X: del | V: move to corr. | Esc"
    text_size, _ = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    x = max(10, display.shape[1] - text_size[0] - 10)
    y = display.shape[0] - 20
    top = max(0, y - text_size[1] - 10)
    cv2.rectangle(display, (x - 8, top), (x + text_size[0] + 8, y + 8), (0, 0, 0), -1)
    cv2.putText(display, hint_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 255), 1)


def compose_grid(panels: list[np.ndarray], columns: int = 2) -> np.ndarray:
    if not panels:
        return np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)

    columns = max(1, columns)
    rows = (len(panels) + columns - 1) // columns
    blank = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)

    row_images: list[np.ndarray] = []
    for row_idx in range(rows):
        start = row_idx * columns
        row_panels = panels[start:start + columns]
        if len(row_panels) < columns:
            row_panels = row_panels + [blank.copy() for _ in range(columns - len(row_panels))]
        row_images.append(np.hstack(row_panels))

    return np.vstack(row_images)


def prompt_sample_id(window_name: str = "Find sample ID") -> str | None:
    text = ""
    width, height = 720, 160

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    while True:
        dialog = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(dialog, "Enter sample ID or index and press Enter", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)
        cv2.rectangle(dialog, (20, 80), (width - 20, 132), (100, 100, 100), 2)

        shown = text if text else "<type ID>"
        shown_color = (255, 255, 255) if text else (140, 140, 140)
        cv2.putText(dialog, shown, (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, shown_color, 2)

        cv2.putText(dialog, "Enter=confirm  Esc=cancel  Backspace=delete", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.imshow(window_name, dialog)

        key = cv2.waitKeyEx(0)

        if key in (10, 13):
            value = text.strip()
            cv2.destroyWindow(window_name)
            return value if value else None

        if key == 27:
            cv2.destroyWindow(window_name)
            return None

        if key in (8, 127, 65288):
            text = text[:-1]
            continue

        if 32 <= key <= 126:
            text += chr(key)


def _find_numeric_like_matches(sample_ids: list[str], number_value: int) -> list[int]:
    """Return indexes of IDs containing a numeric token equal to number_value."""
    matches: list[int] = []
    for idx, sample_id in enumerate(sample_ids):
        for token in re.findall(r"\d+", sample_id):
            if int(token) == number_value:
                matches.append(idx)
                break
    return matches


def find_sample_index(query: str, sample_ids: list[str]) -> tuple[int | None, str | None]:
    """Find sample index by exact ID, 1-based ordinal number, or numeric token in ID."""
    value = query.strip()
    if not value:
        return None, "Empty input"

    try:
        return sample_ids.index(value), None
    except ValueError:
        pass

    if value.isdigit():
        number_value = int(value)

        # Primary behavior: numeric input means 1-based sample position.
        if 1 <= number_value <= len(sample_ids):
            return number_value - 1, f"Interpreted '{value}' as sample index #{number_value}"

        matches = _find_numeric_like_matches(sample_ids, number_value)
        if len(matches) == 1:
            return matches[0], f"Interpreted '{value}' as numeric token in sample ID"
        if len(matches) > 1:
            return None, (
                f"Ambiguous numeric match for '{value}' ({len(matches)} IDs). "
                "Try exact ID or sample index."
            )

    return None, f"ID/index not found: {value}"


def load_sample_ids_file(path: Path) -> tuple[list[str], dict[str, tuple[int, int | None]]]:
    """Load sample IDs from file.

    Each non-empty line may contain:
      sample_id
      sample_id <missing_px>
      sample_id <missing_px> <total_px>
    Delimiters: tab, comma, semicolon, or whitespace.
    """
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Sample IDs file not found: {path}")

    sample_ids: list[str] = []
    stats: dict[str, tuple[int, int | None]] = {}
    seen: set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part for part in re.split(r"[\t,;\s]+", line) if part]
            if not parts:
                continue

            sample_id = parts[0]
            if sample_id not in seen:
                sample_ids.append(sample_id)
                seen.add(sample_id)

            missing_px: int | None = None
            total_px: int | None = None
            if len(parts) >= 2:
                try:
                    missing_px = int(parts[1])
                except ValueError:
                    missing_px = None
            if len(parts) >= 3:
                try:
                    total_px = int(parts[2])
                except ValueError:
                    total_px = None

            if missing_px is not None:
                stats[sample_id] = (missing_px, total_px)

    return sample_ids, stats


def parse_args():
    parser = argparse.ArgumentParser(description="View RGB-D dataset channels side-by-side")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to the 'processed' dataset directory (default: current directory)",
    )
    parser.add_argument(
        "--colormap",
        action="store_true",
        help="Use turbo colormap for depth channels",
    )
    parser.add_argument(
        "--depth",
        action="store_true",
        help="Show depth triplet only (depth_perfect, depth_from_rgb, depth_from_rgb_apple)",
    )
    parser.add_argument(
        "--sample-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional file with sample IDs to display only. "
            "Optional columns with missing/total pixels are used for console reporting."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    processed_dir = Path(args.path).expanduser().resolve()
    channel_defs = resolve_channel_defs(use_depth_triplet=args.depth)

    if not processed_dir.exists() or not processed_dir.is_dir():
        print(f"Error: Processed directory not found: {processed_dir}")
        return

    channel_dirs = {
        "rgb": processed_dir / "rgb",
        "depth_noisy": processed_dir / "depth_noisy",
        "depth_perfect": processed_dir / "depth_perfect",
        "depth_from_rgb": processed_dir / "depth_from_rgb",
        "depth_from_rgb_apple": processed_dir / "depth_from_rgb_apple",
    }

    channel_indexes: dict[str, dict[str, Path]] = {}
    channel_sources: dict[str, str] = {}
    for _, channel_key, source_key in channel_defs:
        folder = channel_dirs[source_key]
        channel_indexes[channel_key] = index_images(folder)
        channel_sources[channel_key] = source_key
        status = "OK" if folder.exists() else "MISSING"
        count = len(channel_indexes[channel_key])
        print(f"Channel {channel_key} (source={source_key}): {count} file(s) [{status}] -> {folder}")

    correction_root = processed_dir.parent / "for_correction"

    sample_ids = recompute_sample_ids(channel_indexes)
    if not sample_ids:
        print(f"Error: No sample IDs found in {processed_dir}")
        return

    missing_stats: dict[str, tuple[int, int | None]] = {}
    if args.sample_ids_file is not None:
        requested_ids, loaded_stats = load_sample_ids_file(args.sample_ids_file.expanduser().resolve())
        if not requested_ids:
            print(f"Error: No sample IDs found in filter file: {args.sample_ids_file}")
            return

        available_ids = set(sample_ids)
        filtered_ids = [sample_id for sample_id in requested_ids if sample_id in available_ids]
        skipped_ids = [sample_id for sample_id in requested_ids if sample_id not in available_ids]

        if skipped_ids:
            print(
                f"WARNING: {len(skipped_ids)} ID(s) from filter file are missing in dataset "
                "and will be skipped"
            )

        if not filtered_ids:
            print("Error: Filtered sample list is empty after matching against dataset IDs")
            return

        sample_ids = filtered_ids
        missing_stats = {sample_id: loaded_stats[sample_id] for sample_id in sample_ids if sample_id in loaded_stats}
        print(f"Filtered viewer to {len(sample_ids)} sample(s) from {args.sample_ids_file}")

    mode_name = "depth_triplet" if args.depth else "five_channel"
    print(f"Using dataset: {processed_dir} [mode={mode_name}]")
    print(f"Found {len(sample_ids)} sample(s): {sample_ids[0]} to {sample_ids[-1]}")
    print("Controls:")
    print("  RIGHT arrow: Next sample")
    print("  LEFT arrow: Previous sample")
    print("  RIGHT arrow: Next sample")
    print("  LEFT arrow: Previous sample")
    print("  F / I: Find and jump to sample ID or sample index (1-based)")
    print("  X: Delete current sample ID")
    print("  V: Move current sample to for_correction")
    print("  ESC: Exit")

    current_idx = 0
    last_reported_id: str | None = None
    window_name = "Dataset Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    columns = 2
    rows = (len(channel_defs) + columns - 1) // columns
    cv2.resizeWindow(window_name, PANEL_WIDTH * columns, PANEL_HEIGHT * rows)

    while True:
        sample_id = sample_ids[current_idx]
        if sample_id != last_reported_id:
            stats = missing_stats.get(sample_id)
            if stats is not None:
                missing_px, total_px = stats
                if total_px is not None and total_px > 0:
                    missing_pct = 100.0 * float(missing_px) / float(total_px)
                    print(
                        f"Sample {sample_id}: depth_perfect missing pixels "
                        f"{missing_px}/{total_px} ({missing_pct:.4f}%)"
                    )
                else:
                    print(f"Sample {sample_id}: depth_perfect missing pixels {missing_px}")
            last_reported_id = sample_id

        raw_channels = load_sample(sample_id, channel_indexes)
        display_channels = prepare_display_channels(raw_channels, use_colormap=args.colormap)

        panels = [
            render_labeled_panel(display_channels.get(channel_key), channel_label)
            for channel_label, channel_key, _ in channel_defs
        ]
        display = compose_grid(panels, columns=columns)

        info_text = f"Sample {sample_id} ({current_idx + 1}/{len(sample_ids)})"
        cv2.putText(
            display,
            info_text,
            (10, display.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        render_control_hint(display)

        cv2.imshow(window_name, display)
        key = cv2.waitKeyEx(0)

        if key == 27:
            print("Exiting...")
            break

        if key in (83, 2555904, 65363, ord("d"), ord("D")):
            current_idx = (current_idx + 1) % len(sample_ids)
            print(f"Next sample: {sample_ids[current_idx]}")

        elif key in (81, 2424832, 65361, ord("a"), ord("A")):
            current_idx = (current_idx - 1) % len(sample_ids)
            print(f"Previous sample: {sample_ids[current_idx]}")

        elif key in (ord("x"), ord("X")):
            deleted_count = delete_current_sample_id(sample_id, channel_indexes)
            sample_ids = recompute_sample_ids(channel_indexes)

            if not sample_ids:
                print(f"Deleted ID {sample_id} ({deleted_count} file(s)). No samples left.")
                break

            current_idx = min(current_idx, len(sample_ids) - 1)
            print(f"Deleted ID {sample_id} ({deleted_count} file(s)).")
            print(f"Now {len(sample_ids)} sample ID(s) remaining.")

        elif key in (ord("v"), ord("v")):
            moved_count = move_current_sample_to_correction(
                sample_id,
                channel_indexes,
                channel_sources,
                correction_root,
            )
            sample_ids = recompute_sample_ids(channel_indexes)

            if not sample_ids:
                print(
                    f"Moved ID {sample_id} ({moved_count} file(s)) to {correction_root}. "
                    "No samples left."
                )
                break

            current_idx = min(current_idx, len(sample_ids) - 1)
            print(f"Moved ID {sample_id} ({moved_count} file(s)) to {correction_root}.")
            print(f"Now {len(sample_ids)} sample ID(s) remaining.")

        elif key in (ord("f"), ord("F"), ord("i"), ord("I")):
            searched_id = prompt_sample_id()
            if searched_id is None:
                print("ID search canceled")
                continue

            found_idx, reason = find_sample_index(searched_id, sample_ids)
            if found_idx is None:
                print(reason if reason else f"ID/index not found: {searched_id}")
                continue

            current_idx = found_idx
            if reason:
                print(reason)
            print(f"Jumped to sample ID: {sample_ids[current_idx]}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

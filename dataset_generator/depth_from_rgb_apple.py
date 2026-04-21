import argparse
from dataclasses import dataclass, replace
import importlib
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

DEPTH_PRO_ROOT_CANDIDATES = [
    PROJECT_ROOT / "ml-depth-pro",
    PROJECT_ROOT.parent / "ml-depth-pro",
]
DEPTH_PRO_ROOT = next(
    (path for path in DEPTH_PRO_ROOT_CANDIDATES if path.is_dir()),
    DEPTH_PRO_ROOT_CANDIDATES[0],
)
DEPTH_PRO_SRC = DEPTH_PRO_ROOT / "src"
if DEPTH_PRO_SRC.is_dir() and str(DEPTH_PRO_SRC) not in sys.path:
    sys.path.insert(0, str(DEPTH_PRO_SRC))

try:
    depth_pro = importlib.import_module("depth_pro")
    depth_pro_module = importlib.import_module("depth_pro.depth_pro")
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Cannot import depth_pro. Install ml-depth-pro (pip install -e .) "
        "or ensure Magisterka/ml-depth-pro/src is available."
    ) from exc

create_model_and_transforms = depth_pro.create_model_and_transforms
load_rgb = depth_pro.load_rgb
DEFAULT_MONODEPTH_CONFIG_DICT = depth_pro_module.DEFAULT_MONODEPTH_CONFIG_DICT

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".heic"}


@dataclass(frozen=True)
class CameraFocalParams:
    fx_px: float
    width_px: int | None
    height_px: int | None


def _default_checkpoint_path() -> Path:
    candidates = [
        DEPTH_PRO_ROOT / "checkpoints" / "depth_pro.pt",
        Path("checkpoints") / "depth_pro.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _natural_sort_key(value: str):
    parts = re.split(r"(\d+)", value)
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key)


def _sort_key(path: Path):
    return _natural_sort_key(path.stem)


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


def _collect_images(input_dir: Path) -> list[str]:
    return [
        str(path)
        for path in sorted(input_dir.iterdir(), key=_sort_key)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXT
    ]


def _collect_existing_output_ids(depth_dir: Path) -> set[str]:
    if not depth_dir.exists():
        return set()

    return {
        path.stem
        for path in depth_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }


def _depth_m_to_uint16_mm(depth_map: np.ndarray) -> np.ndarray:
    """Convert metric depth in meters to uint16 millimeters."""
    depth = np.nan_to_num(depth_map.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(depth * 1000.0, 0.0, 65535.0).astype(np.uint16)


def _load_main_folder_from_config(config_path: Path) -> Path | None:
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return None

    main_folder = cfg.get("main_folder")
    if not main_folder:
        return None
    return Path(str(main_folder)).expanduser().resolve()


def _resolve_camera_params_path(config_path: Path, camera_params_arg: Path | None) -> Path | None:
    if camera_params_arg is not None:
        path = camera_params_arg.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Camera params file not found: {path}")
        return path

    main_folder = _load_main_folder_from_config(config_path)
    if main_folder is None:
        return None

    auto_path = main_folder / "synthetic-rgbd-camera-model" / "params" / "femto_mega.json"
    if auto_path.is_file():
        return auto_path
    return None


def _load_rgb_focal_params(camera_params_path: Path) -> CameraFocalParams:
    with open(camera_params_path, "r", encoding="utf-8") as f:
        params = json.load(f) or {}

    rgb = params.get("rgb") or {}
    fx = rgb.get("fx")
    if not isinstance(fx, (int, float)):
        raise ValueError(f"Missing numeric rgb.fx in camera params: {camera_params_path}")

    fx = float(fx)
    if fx <= 0.0:
        raise ValueError(f"rgb.fx must be > 0 in camera params: {camera_params_path}")

    width = rgb.get("width")
    height = rgb.get("height")

    resolution = rgb.get("resolution")
    if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
        if width is None:
            width = resolution[0]
        if height is None:
            height = resolution[1]

    width_px = int(width) if isinstance(width, (int, float)) and float(width) > 0 else None
    height_px = int(height) if isinstance(height, (int, float)) and float(height) > 0 else None

    return CameraFocalParams(fx_px=fx, width_px=width_px, height_px=height_px)


def _scaled_camera_focal_px_for_image(
    camera_focal: CameraFocalParams,
    image_shape: tuple[int, ...],
) -> tuple[float, str | None]:
    """Scale camera fx to current image resolution if camera resolution is known."""
    if len(image_shape) < 2:
        raise ValueError("Image shape must have at least 2 dimensions (H, W)")

    image_h = int(image_shape[0])
    image_w = int(image_shape[1])
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"Invalid image shape for focal scaling: {image_shape}")

    if camera_focal.width_px is None or camera_focal.height_px is None:
        return camera_focal.fx_px, None

    sx = float(image_w) / float(camera_focal.width_px)
    sy = float(image_h) / float(camera_focal.height_px)

    warning: str | None = None
    if max(sx, sy) > 0 and abs(sx - sy) / max(sx, sy) > 0.02:
        warning = (
            "camera/exif aspect scaling mismatch; using average scale "
            f"(sx={sx:.4f}, sy={sy:.4f})"
        )
        scale = 0.5 * (sx + sy)
    else:
        scale = sx

    return float(camera_focal.fx_px * scale), warning


def _validate_depth_output(
    depth_map_m: np.ndarray,
    depth_u16_mm: np.ndarray,
    image_path: str,
) -> tuple[int, int]:
    if depth_u16_mm.dtype != np.uint16:
        raise TypeError(f"Expected uint16 depth output for {image_path}, got {depth_u16_mm.dtype}")
    if depth_u16_mm.ndim != 2:
        raise ValueError(f"Expected 2D depth output for {image_path}, got shape={depth_u16_mm.shape}")

    finite_mask = np.isfinite(depth_map_m)
    if not np.any(finite_mask):
        print(f"WARNING: {Path(image_path).name}: inferred depth contains no finite values.")
    elif not np.all(finite_mask):
        invalid_count = int(depth_map_m.size - np.count_nonzero(finite_mask))
        print(f"WARNING: {Path(image_path).name}: inferred depth has {invalid_count} non-finite value(s).")

    min_mm = int(np.min(depth_u16_mm))
    max_mm = int(np.max(depth_u16_mm))
    if min_mm < 0 or max_mm > 65535:
        raise ValueError(
            f"Depth range out of uint16-mm bounds for {image_path}: min={min_mm}, max={max_mm}"
        )

    if max_mm == 0:
        print(f"WARNING: {Path(image_path).name}: output depth is fully zero after conversion.")

    return min_mm, max_mm


def _to_scalar_focal_px(value) -> float | None:
    """Normalize EXIF/camera focal values to a Python float when possible."""
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().reshape(-1)[0].item())

    array = np.asarray(value)
    if array.size == 0:
        return None

    return float(array.reshape(-1)[0])


def _prepare_focal_for_infer(f_px_scalar: float | None, input_tensor: torch.Tensor) -> torch.Tensor | None:
    """Build focal argument compatible with Depth Pro infer() implementation."""
    if f_px_scalar is None:
        return None
    return torch.tensor([f_px_scalar], dtype=torch.float32, device=input_tensor.device)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return torch.device("cuda:0")

    if device_arg == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device: {device_arg}")


def _precision_for_device(requested_precision: str, device: torch.device) -> torch.dtype:
    if requested_precision == "fp16" and device.type == "cpu":
        print("Requested fp16 on CPU; falling back to fp32.")
        return torch.float32
    return torch.half if requested_precision == "fp16" else torch.float32


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
    requested_precision: str,
):
    precision = _precision_for_device(requested_precision, device)
    config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(checkpoint_path))
    model, transform = create_model_and_transforms(config=config, device=device, precision=precision)
    model.eval()
    return model, transform, precision


def _is_cuda_oom(exc: RuntimeError) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch depth inference with Apple Depth Pro and PNG export.")
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
        help="Path to processed directory (contains rgb/, depth_noisy/, depth_perfect/)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="RGB input directory (default: <processed-dir>/rgb)",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=None,
        help="Output depth_from_rgb_apple directory (default: <processed-dir>/depth_from_rgb_apple)",
    )
    parser.add_argument("--batch", type=int, default=1, help="Images per processing chunk.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate output for all IDs even if destination PNG already exists.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_default_checkpoint_path(),
        help="Path to Depth Pro checkpoint file (depth_pro.pt).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Inference precision.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--ignore-exif-focal",
        action="store_true",
        help=(
            "Ignore focal length from EXIF. If --focal-length-px or --camera-params "
            "is available, those values are still used."
        ),
    )
    parser.add_argument(
        "--focal-length-px",
        type=float,
        default=None,
        help="Optional fixed focal length in pixels passed to Depth Pro infer().",
    )
    parser.add_argument(
        "--camera-params",
        type=Path,
        default=None,
        help=(
            "Optional camera params JSON with rgb.fx. If omitted, script auto-tries "
            "<main_folder>/synthetic-rgbd-camera-model/params/femto_mega.json from config."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch <= 0:
        raise ValueError("--batch must be > 0")

    config_path = args.config.expanduser().resolve()

    if args.processed_dir is not None:
        processed_dir = args.processed_dir.expanduser().resolve()
    elif args.input_dir is None or args.depth_dir is None:
        processed_dir = _resolve_processed_dir_from_config(config_path)
    else:
        processed_dir = None

    input_dir = args.input_dir.expanduser().resolve() if args.input_dir is not None else processed_dir / "rgb"
    depth_dir = (
        args.depth_dir.expanduser().resolve()
        if args.depth_dir is not None
        else processed_dir / "depth_from_rgb_apple"
    )
    checkpoint_path = args.checkpoint.expanduser().resolve()

    if args.focal_length_px is not None and args.focal_length_px <= 0:
        raise ValueError("--focal-length-px must be > 0")

    camera_params_path = _resolve_camera_params_path(config_path, args.camera_params)
    camera_params_focal: CameraFocalParams | None = None
    if camera_params_path is not None:
        try:
            camera_params_focal = _load_rgb_focal_params(camera_params_path)
        except Exception as exc:
            print(f"WARNING: could not read rgb.fx from camera params ({camera_params_path}): {exc}")

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Depth Pro checkpoint not found: {checkpoint_path}. "
            "Download it with ml-depth-pro/get_pretrained_models.sh"
        )

    images = _collect_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No input images found in: {input_dir}")

    depth_dir.mkdir(parents=True, exist_ok=True)
    existing_output_ids = _collect_existing_output_ids(depth_dir)
    if args.overwrite:
        pending_images = images
        skipped_existing = 0
    else:
        pending_images = [image_path for image_path in images if Path(image_path).stem not in existing_output_ids]
        skipped_existing = len(images) - len(pending_images)
    print(f"Input RGB dir: {input_dir}")
    print(f"Output depth dir: {depth_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Overwrite existing outputs: {'yes' if args.overwrite else 'no'}")
    if args.focal_length_px is not None:
        print(f"Using fixed focal length: {args.focal_length_px:.3f} px")
    elif camera_params_focal is not None:
        resolution_info = ""
        if camera_params_focal.width_px is not None and camera_params_focal.height_px is not None:
            resolution_info = f", camera_res={camera_params_focal.width_px}x{camera_params_focal.height_px}"
        print(
            "Using camera-params focal fallback when EXIF is missing: "
            f"{camera_params_focal.fx_px:.3f} px ({camera_params_path}{resolution_info})"
        )
    else:
        print("No fixed focal fallback configured. Missing EXIF -> model-estimated focal.")
    print(f"Found {len(images)} image(s) in {input_dir}.")
    if args.overwrite:
        print(f"Will regenerate all {len(images)} image(s) in {depth_dir}.")
    else:
        print(f"Skipping {skipped_existing} image(s) already present in {depth_dir}.")

    if not pending_images:
        print("Nothing to process. All IDs already exist in output.")
        return

    device = _resolve_device(args.device)
    try:
        model, transform, precision = _load_model(checkpoint_path, device, args.precision)
    except RuntimeError as exc:
        if args.device == "auto" and device.type == "cuda" and _is_cuda_oom(exc):
            print("CUDA OOM detected while loading model. Switching to CPU (fp32)...")
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            model, transform, precision = _load_model(checkpoint_path, device, "fp32")
        else:
            raise

    print(
        f"Processing {len(pending_images)} image(s) in chunks of {args.batch} "
        f"on device: {device} ({str(precision).replace('torch.', '')})."
    )

    for start in range(0, len(pending_images), args.batch):
        end = min(start + args.batch, len(pending_images))
        batch_images = pending_images[start:end]
        print(f"[chunk {start}:{end}] running inference for {len(batch_images)} image(s)...")

        for image_path in batch_images:
            image_np, _, f_px_exif = load_rgb(image_path)

            focal_source = "model_estimated"
            if args.focal_length_px is not None:
                f_px_scalar = float(args.focal_length_px)
                focal_source = "fixed"
            else:
                f_px_scalar = None if args.ignore_exif_focal else _to_scalar_focal_px(f_px_exif)
                if f_px_scalar is not None:
                    focal_source = "exif"
                elif camera_params_focal is not None:
                    f_px_scalar, scaling_warning = _scaled_camera_focal_px_for_image(
                        camera_params_focal,
                        image_np.shape,
                    )
                    if scaling_warning is not None:
                        print(f"WARNING: {Path(image_path).name}: {scaling_warning}")
                    if camera_params_focal.width_px is not None and camera_params_focal.height_px is not None:
                        focal_source = "camera_params_scaled"
                    else:
                        focal_source = "camera_params"

            if f_px_scalar is None:
                focal_log = "model-estimated"
            else:
                focal_log = f"{f_px_scalar:.3f}px ({focal_source})"

            input_tensor = transform(image_np)
            f_px_for_infer = _prepare_focal_for_infer(f_px_scalar, input_tensor)

            try:
                prediction = model.infer(input_tensor, f_px=f_px_for_infer)
            except RuntimeError as exc:
                if args.device == "auto" and device.type == "cuda" and _is_cuda_oom(exc):
                    print("CUDA OOM detected. Switching to CPU (fp32) and retrying current image...")
                    torch.cuda.empty_cache()
                    device = torch.device("cpu")
                    model, transform, precision = _load_model(checkpoint_path, device, "fp32")
                    print(
                        f"Continuing on device: {device} "
                        f"({str(precision).replace('torch.', '')})."
                    )
                    input_tensor = transform(image_np)
                    f_px_for_infer = _prepare_focal_for_infer(f_px_scalar, input_tensor)
                    prediction = model.infer(input_tensor, f_px=f_px_for_infer)
                else:
                    raise

            depth_map = prediction["depth"].detach().cpu().numpy().squeeze().astype(np.float32)
            depth_u16_mm = _depth_m_to_uint16_mm(depth_map)
            min_mm, max_mm = _validate_depth_output(depth_map, depth_u16_mm, image_path)

            out_path = depth_dir / f"{Path(image_path).stem}.png"
            ok = cv2.imwrite(str(out_path), depth_u16_mm)
            if not ok:
                raise RuntimeError(f"Failed to save depth PNG: {out_path}")

            print(
                f"  saved {out_path.name}: depth_mm=[{min_mm}, {max_mm}] "
                f"focal={focal_log}"
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Done. Depth PNG maps (uint16 mm) saved in: {depth_dir}")


if __name__ == "__main__":
    main()

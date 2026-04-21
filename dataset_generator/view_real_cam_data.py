"""
View real camera samples (RGB + aligned noisy depth) side-by-side.

Expected structure:
	raw/
	  depth2color_aligned/
		<sample_id>/
		  color/*.png
		  depth/*.png
		  timestamp.txt

Usage:
	python view_real_cam_data.py /path/to/output_real_camera/raw
	python view_real_cam_data.py /path/to/output_real_camera/raw --colormap
"""

import argparse
import os

import cv2
import numpy as np


def process_depth_img(path, use_colormap):
	"""Load and normalize depth image for visualization."""
	if not os.path.exists(path):
		return None

	img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	if img is None:
		return None

	valid = img[img > 0]
	if len(valid) == 0:
		return np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

	d_min, d_max = valid.min(), valid.max()
	vis = np.zeros(img.shape[:2], dtype=np.uint8)
	mask = img > 0
	vis[mask] = np.clip(
		(img[mask].astype(float) - d_min) / max(d_max - d_min, 1) * 255,
		0,
		255,
	).astype(np.uint8)

	if use_colormap:
		vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
		vis[~mask] = 0
	else:
		vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

	return vis


def first_png_in_dir(folder_path):
	"""Return first PNG path from folder, or None when folder is missing/empty."""
	if not os.path.isdir(folder_path):
		return None

	files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".png"))
	if not files:
		return None
	return os.path.join(folder_path, files[0])


def load_timestamp(sample_dir):
	"""Read timestamp file when present."""
	ts_path = os.path.join(sample_dir, "timestamp.txt")
	if not os.path.exists(ts_path):
		return "N/A"
	try:
		with open(ts_path, "r", encoding="utf-8") as f:
			line = f.readline().strip()
			return line if line else "N/A"
	except OSError:
		return "N/A"


def create_missing_image(width=1920, height=1080, text="MISSING"):
	"""Create fallback gray panel with red text."""
	img = np.ones((height, width, 3), dtype=np.uint8) * 127
	cv2.putText(img, text, (80, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
	return img


def safe_delete_file(path):
	"""Delete file if it exists; return True if deleted."""
	if os.path.exists(path):
		try:
			os.remove(path)
			print(f"[Deleted] {path}")
			return True
		except OSError as e:
			print(f"[Error] Could not delete {path}: {e}")
			return False
	print(f"[Missing] {path}")
	return False


def cleanup_empty_dirs(sample_dir):
	"""Remove empty color/depth/sample directories after deletion."""
	for rel in ("color", "depth", ""):
		target = os.path.join(sample_dir, rel) if rel else sample_dir
		try:
			if os.path.isdir(target) and not os.listdir(target):
				os.rmdir(target)
				print(f"[Removed empty dir] {target}")
		except OSError:
			# Directory still contains files or cannot be removed.
			pass


def view_pair(sample_dir, use_colormap, save_dir):
	"""Load RGB + Depth for one sample directory and show side-by-side preview."""
	sample_id = os.path.basename(sample_dir)
	color_path = first_png_in_dir(os.path.join(sample_dir, "color"))
	depth_path = first_png_in_dir(os.path.join(sample_dir, "depth"))
	timestamp = load_timestamp(sample_dir)

	if color_path:
		rgb_img = cv2.imread(color_path)
		if rgb_img is None:
			rgb_img = create_missing_image(text="BAD RGB FILE")
	else:
		rgb_img = create_missing_image(text="NO RGB")

	depth_vis = process_depth_img(depth_path, use_colormap) if depth_path else None
	if depth_vis is None:
		depth_vis = create_missing_image(width=rgb_img.shape[1], height=rgb_img.shape[0], text="NO DEPTH")

	target_h = rgb_img.shape[0]
	h, w = depth_vis.shape[:2]
	aspect = w / max(h, 1)
	depth_vis = cv2.resize(depth_vis, (max(int(target_h * aspect), 1), target_h))

	cv2.putText(rgb_img, "RGB", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
	cv2.putText(depth_vis, "Depth Noisy", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

	combined = cv2.hconcat([rgb_img, depth_vis])
	cv2.putText(
		combined,
		f"Sample: {sample_id} | Timestamp: {timestamp}",
		(20, combined.shape[0] - 20),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.8,
		(255, 255, 255),
		2,
	)

	max_w = 1800
	if combined.shape[1] > max_w:
		scale = max_w / combined.shape[1]
		combined = cv2.resize(
			combined,
			(int(combined.shape[1] * scale), int(combined.shape[0] * scale)),
		)

	pending_delete = False

	print(f"\n[Showing] Sample: {sample_id}")
	print("Controls: [Space] Next | [S] Save | [Backspace x2] Delete current sample files | [Q]/[Esc] Quit")

	while True:
		display_img = combined.copy()
		if pending_delete:
			h, w = display_img.shape[:2]
			cv2.line(display_img, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
			cv2.line(display_img, (w - 1, 0), (0, h - 1), (0, 0, 255), 10)

		cv2.imshow("Real Camera Viewer", display_img)
		key = cv2.waitKey(0) & 0xFF

		is_backspace = key in (8, 127)

		if key == ord(" "):
			return True
		if key == ord("s"):
			save_path = os.path.join(save_dir, f"{sample_id}.png")
			if cv2.imwrite(save_path, combined):
				print(f"[Saved] {save_path}")
			else:
				print(f"[Error] Could not save image to {save_path}")
			continue
		if is_backspace:
			if not pending_delete:
				pending_delete = True
				print("[ARMED] Press Backspace again to confirm delete. Press Esc to cancel delete mode.")
			else:
				deleted_any = False
				if color_path:
					deleted_any = safe_delete_file(color_path) or deleted_any
				if depth_path:
					deleted_any = safe_delete_file(depth_path) or deleted_any
				deleted_any = safe_delete_file(os.path.join(sample_dir, "timestamp.txt")) or deleted_any
				cleanup_empty_dirs(sample_dir)

				if deleted_any:
					print("[INFO] Current sample files deleted. Moving to next sample.")
				else:
					print("[INFO] Nothing deleted for current sample. Moving to next sample.")
				return True
			continue
		if key == ord("q") or key == 27:
			if key == 27 and pending_delete:
				pending_delete = False
				print("[CANCELLED] Delete mode cancelled. Press Esc again to quit.")
				continue
			return False


def resolve_aligned_root(input_path):
	"""Accept both raw/ and raw/depth2color_aligned/ as script input."""
	input_path = input_path.rstrip("/")
	if os.path.basename(input_path) == "depth2color_aligned":
		return input_path

	candidate = os.path.join(input_path, "depth2color_aligned")
	if os.path.isdir(candidate):
		return candidate
	return input_path


def main():
	parser = argparse.ArgumentParser(description="View real camera RGB + noisy depth side-by-side")
	parser.add_argument(
		"path",
		help="Path to raw/ or raw/depth2color_aligned directory",
	)
	parser.add_argument(
		"--colormap",
		action="store_true",
		help="Use turbo colormap for depth instead of grayscale",
	)
	args = parser.parse_args()

	aligned_root = resolve_aligned_root(args.path)
	save_dir = os.path.join(aligned_root, "saved_previews")
	os.makedirs(save_dir, exist_ok=True)

	if not os.path.isdir(aligned_root):
		print(f"Error: Directory not found: {aligned_root}")
		return

	sample_dirs = [
		os.path.join(aligned_root, name)
		for name in sorted(os.listdir(aligned_root))
		if os.path.isdir(os.path.join(aligned_root, name)) and name != "saved_previews"
	]

	if not sample_dirs:
		print(f"No sample subdirectories found in: {aligned_root}")
		return

	print(f"Found {len(sample_dirs)} samples in {aligned_root}")
	print("Press Space for next sample.")

	for sample_dir in sample_dirs:
		should_continue = view_pair(sample_dir, args.colormap, save_dir)
		if not should_continue:
			break

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

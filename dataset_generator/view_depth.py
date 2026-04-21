"""
View a uint16 depth PNG in a human-readable way.

Usage:
    python view_depth.py /ścieżka/do/folderu/processed
    --colormap  # use blue-to-red colormap
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
        0, 255
    ).astype(np.uint8)

    if use_colormap:
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
        vis[~mask] = 0
    else:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR) # convert to 3 ch for concat
        
    return vis

def view_triplet(processed_dir, rgb_filename, use_colormap, save_dir):
    """Load RGB, Perfect Depth, and Noisy Depth for a specific file and concatenate them."""
    rgb_path = os.path.join(processed_dir, "rgb", rgb_filename)
    depth_perfect_path = os.path.join(processed_dir, "depth_perfect", rgb_filename)
    depth_noisy_path = os.path.join(processed_dir, "depth_noisy", rgb_filename)
    
    # ── 1. Read RGB ──
    if os.path.exists(rgb_path):
        rgb_img = cv2.imread(rgb_path)
    else:
        # Dummy gray image if missing
        rgb_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 127
        cv2.putText(rgb_img, "NO RGB", (400, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

    # ── 2. Read Perfect Depth ──
    depth_perfect_vis = process_depth_img(depth_perfect_path, use_colormap)
    if depth_perfect_vis is None:
        depth_perfect_vis = np.ones_like(rgb_img) * 127
        cv2.putText(depth_perfect_vis, "NO DEPTH PERFECT", (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        
    # Resize depth to match RGB height (since sensors often have different res, though here they are all 1920x1080)
    target_h = rgb_img.shape[0]
    h, w = depth_perfect_vis.shape[:2]
    aspect = w / h
    depth_perfect_vis = cv2.resize(depth_perfect_vis, (int(target_h * aspect), target_h))
    
    # ── 3. Read Noisy Depth ──
    depth_noisy_vis = process_depth_img(depth_noisy_path, use_colormap)
    if depth_noisy_vis is None:
        depth_noisy_vis = np.ones_like(depth_perfect_vis) * 127
        cv2.putText(depth_noisy_vis, "NO NOISY", (350, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
    else:
        depth_noisy_vis = cv2.resize(depth_noisy_vis, (depth_perfect_vis.shape[1], target_h))

    # Add labels
    cv2.putText(rgb_img, "RGB", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(depth_perfect_vis, "Depth Perfect", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(depth_noisy_vis, "Depth Noisy", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Concatenate horizontally (RGB | Depth Perfect | Depth Noisy)
    combined = cv2.hconcat([rgb_img, depth_perfect_vis, depth_noisy_vis])

    # Resize window down to fit screen if needed
    max_w = 1800
    if combined.shape[1] > max_w:
        scale = max_w / combined.shape[1]
        combined = cv2.resize(combined, (int(combined.shape[1]*scale), int(combined.shape[0]*scale)))

    pending_delete = False

    print(f"\n[Showing] File: {rgb_filename}")
    print("Controls: [Space] = Next image | [B] = Previous image | [S] = Save | [Backspace x2] = Delete triplet | [Q] or [Esc] = Quit")
    
    while True:
        display_img = combined.copy()
        if pending_delete:
            # Deletion safety indicator: draw a red X over the preview.
            h, w = display_img.shape[:2]
            cv2.line(display_img, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
            cv2.line(display_img, (w - 1, 0), (0, h - 1), (0, 0, 255), 10)

        cv2.imshow("Dataset Viewer", display_img)
        key = cv2.waitKey(0) & 0xFF

        # Backspace can be 8 on many Linux setups and 127 in some terminals.
        is_backspace = key in (8, 127)

        if key == ord(' '):  # Space
            return True  # continue to next
        elif key == ord('b') or key == ord('B'):  # b or B
            return "back"  # go back to previous
        elif key == ord('s'):
            save_path = os.path.join(save_dir, rgb_filename)
            if cv2.imwrite(save_path, combined):
                print(f"[Saved] {save_path}")
            else:
                print(f"[Error] Could not save image to {save_path}")
        elif is_backspace:
            if not pending_delete:
                pending_delete = True
                print("[ARMED] Deletion armed for current triplet. Press Backspace again to confirm, Esc to cancel.")
            else:
                deleted_any = False
                for path in (rgb_path, depth_perfect_path, depth_noisy_path):
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                            print(f"[Deleted] {path}")
                            deleted_any = True
                        except OSError as e:
                            print(f"[Error] Could not delete {path}: {e}")
                    else:
                        print(f"[Missing] {path}")

                if deleted_any:
                    print("[INFO] Current triplet deleted. Moving to next image.")
                else:
                    print("[INFO] Nothing was deleted for current triplet. Moving to next image.")
                return True
        elif key == ord('q') or key == 27:  # q or Escape
            if key == 27 and pending_delete:
                pending_delete = False
                print("[CANCELLED] Deletion cancelled. Press Esc again to quit viewer.")
                continue
            return False # quit


def main():
    parser = argparse.ArgumentParser(description="View RGB + Depth Perfect + Depth Noisy side-by-side")
    parser.add_argument("path", help="Path to the 'processed' dataset directory (containing rgb/, depth_perfect/ folders)")
    parser.add_argument("--colormap", action="store_true",
                        help="Use turbo colormap instead of grayscale")
    args = parser.parse_args()

    processed_dir = args.path.rstrip('/')
    save_dir = os.path.join(processed_dir, "saved")
    os.makedirs(save_dir, exist_ok=True)
    
    if not os.path.isdir(processed_dir):
        print(f"Error: Processed directory '{processed_dir}' not found.")
        return
        
    rgb_dir = os.path.join(processed_dir, "rgb")
    if not os.path.isdir(rgb_dir):
        print(f"Error: No 'rgb' subfolder found inside {processed_dir}. Is this the right dataset path?")
        return

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    if not rgb_files:
        print(f"No RGB PNG files found in {rgb_dir}")
        return
    
    print(f"Found {len(rgb_files)} samples in {processed_dir}. Press Space to iterate.")
    
    idx = 0
    while idx < len(rgb_files):
        result = view_triplet(processed_dir, rgb_files[idx], args.colormap, save_dir)
        if result == "back":
            # Go to previous image
            idx = max(0, idx - 1)
        elif result is True:
            # Go to next image
            idx += 1
        else:
            # result is False (quit)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

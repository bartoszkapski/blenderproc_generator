import blenderproc as bproc
import bpy
# Importy blenderproc/bpy musza byc na poczatku skryptu


"""
Przykład uruchomienia:
    blenderproc run  generate_data_v3.py --seed 42 --num_samples 5 --num_repeats 3

Wymagane argumenty:
    --seed <int> : ziarno losowania
    --num_samples <int> : liczba kadrów dla każdej sceny
    --num_repeats <int> : liczba powtorzen całego procesu z nową sceną

Dodatkowe argumenty:
    --config <path> : sciezka do config.yaml
    --hdri : zaladowanie dodatkowego HDRI do sceny
    --physics : upuszczenie obiektow przez symulację fizyki (proszczona fizyka - Raycastingu)
    --post-process <num_workers> : po zakończeniu renderu uruchamia skrypt post-processingu (num_workers >= 1)
    --debug : tryb debug (buduje scenę, ale nic nie renderuje)
  
Tryb debug (buduje scenę, ale nic nie renderuje):
    blenderproc debug generate_data_v3.py --seed 42 --num_samples 2 --num_repeats 1 --debug
"""



import argparse
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from part_physics import simulate_loaded_objects_physics
from part_render import (
    compute_camera_params,
    configure_bproc_optix_renderer,
    render_depth_pass,
    render_rgb_pass,
    sample_camera_poses,
    setup_cameras,
)
from part_scenes import choose_and_load_scene, load_objects_into_scene, place_objects_in_xy_bounds

VERSION = "1.3 - Fixed Depth duplication"
print(f"\n[BOOT] {__file__} | Version: {VERSION}")
POST_PROCESS_SCRIPT = SCRIPT_DIR.parent / "generate_real_depth_thread.py"




def load_config(config_path: str) -> dict:
    """Wczytuje config YAML i rozwiazuje sciezki datasetow."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    main = cfg["main_folder"]
    resolved = {}
    for key, rel in cfg["datasets"].items():
        resolved[key] = os.path.join(main, rel.lstrip("/"))
    cfg["datasets"] = resolved
    cfg["output"] = os.path.join(main, cfg["output"].lstrip("/"))
    return cfg




def create_output_dirs(base_output: str, scene_name: str, seed: int) -> dict:
    """Tworzy strukture katalogow wyjsciowych dla jednego uruchomienia."""
    from datetime import datetime
    import random
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = str(random.randint(10000000, 99999999))
    run_name = f"{random_id}_{timestamp}"
    run_dir = os.path.join(base_output, scene_name, run_name)

    raw_dir = os.path.join(run_dir, "raw")

    dirs = {
        "root": run_dir,
        "run_name": run_name,
        "raw": raw_dir,
    }
    for k, d in dirs.items():
        if k != "run_name":
            os.makedirs(d, exist_ok=True)

    return dirs




def run_single_repeat(cfg: dict, cam_params: dict, repeat_idx: int,
                      seed: int, num_samples: int, debug: bool = False,
                      physics: bool = False, hdri: bool = False):
    """Wykonuje jeden pelny przebieg generowania sceny i renderu."""
    datasets = cfg["datasets"]
    import random

    print(f"\n{'='*60}")
    print(f"  REPEAT {repeat_idx} | seed={seed}")
    print(f"{'='*60}\n")

    # Czyszczenie sceny
    bproc.clean_up()
    
    for obj in bpy.data.objects:
        if obj.type in ['LIGHT', 'CAMERA']:
            bpy.data.objects.remove(obj, do_unlink=True)

    scene_name, scene_type, scene_objs = choose_and_load_scene(cfg, hdri=hdri)

    # Wczytanie i rozmieszczenie obiektow
    loaded_objects = load_objects_into_scene(datasets, cfg["settings"])

    if loaded_objects:
        place_objects_in_xy_bounds(scene_objs, loaded_objects)

    # Symulacja fizyki tylko dla nowo wczytanych obiektow
    if physics and loaded_objects:
        print("[INFO] Running rigid-body physics for loaded objects...")
        simulate_loaded_objects_physics(scene_objs, loaded_objects)
        print("[INFO] Physics simulation complete.")

    # Ustawienie kamer
    depth_cam, rgb_cam = setup_cameras(cam_params)

    # Losowanie poz kamer
    poses = sample_camera_poses(
        scene_objs, loaded_objects, num_samples, depth_cam, rgb_cam,
        settings=cfg["settings"],
    )

    if not poses:
        msg = (
            "No valid camera poses sampled. "
            "Try lowering camera constraints (camera_proximity / camera_sampling) "
            "or reducing scene complexity."
        )
        print(f"[ERROR] {msg}")
        raise RuntimeError(msg)

    # Tryb debug: zatrzymaj po przygotowaniu sceny
    if debug:
        print("\n" + "=" * 60)
        print("  DEBUG MODE — Scene built successfully!")
        print(f"  {len(poses)} camera poses sampled.")

        # Weryfikacja, czy obiekty sa w scenie
        print(f"\n  Loaded objects ({len(loaded_objects)}):")
        bpy.ops.object.select_all(action='DESELECT')
        found_count = 0
        for obj in loaded_objects:
            try:
                blender_obj = obj.blender_obj
                name = blender_obj.name
                loc = blender_obj.location
                in_scene = name in bpy.data.objects
                if in_scene:
                    blender_obj.select_set(True)
                    found_count += 1
                visible = not blender_obj.hide_viewport
                print(f"    {name:30s} loc=({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f}) "
                      f"in_scene={in_scene} visible={visible}")
            except Exception as e:
                print(f"    [ERROR] {e}")

        print(f"\n  {found_count}/{len(loaded_objects)} objects found in scene")
        if found_count > 0:
            # Ustaw fokus widoku na zaladowane obiekty
            try:
                bpy.context.view_layer.objects.active = loaded_objects[0].blender_obj
            except Exception:
                pass
            print("  → Loaded objects are SELECTED (highlighted) in viewport")

        print("  Scene is ready for inspection in Blender.")
        print("  No rendering will be performed.")
        print("=" * 60 + "\n")
        return

    # Tworzenie katalogow wyjsciowych
    out_dirs = create_output_dirs(cfg["output"], scene_name, seed)
    print(f"[INFO] Output directory: {out_dirs['root']}")

    # Konfiguracja renderera BlenderProc (OPTIX)
    configure_bproc_optix_renderer()

    # Render przebiegu RGB
    print("[INFO] Rendering RGB...")
    render_rgb_pass(depth_cam, rgb_cam, cam_params, out_dirs["raw"], poses, out_dirs["run_name"], cfg)

    # Render przebiegu Depth GT
    print("[INFO] Rendering Depth Ground Truth...")
    render_depth_pass(depth_cam, rgb_cam, cam_params,
                      out_dirs["raw"], poses, out_dirs["run_name"])

    # Informacja o post-processingu depth
    processed_dir = os.path.join(out_dirs['root'], 'processed')
    print("\n[INFO] Depth distortion should be run separately (or via wrapper) on this raw folder:")
    print(f"  python generate_real_depth.py --input '{out_dirs['raw']}' --output '{processed_dir}'")

    print(f"\n[DONE] Repeat {repeat_idx} complete → {out_dirs['root']}\n")
    return out_dirs['root']


def main():
    """Parsuje argumenty i uruchamia generator wraz z opcjonalnym post-processem."""
    parser = argparse.ArgumentParser(description="BlenderProc Data Generator")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of camera poses per scene")
    parser.add_argument("--num_repeats", type=int, default=1,
                        help="Number of scene generation repeats")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: build scene only, skip rendering")
    parser.add_argument("--physics", action="store_true",
                        help="Enable physics simulation for object placement")
    parser.add_argument("--hdri", action="store_true",
                        help="Load HDRI environment lighting (requires HDRI dataset path in config.yaml)")
    parser.add_argument("--post-process", type=int, default=0,
                        help="If >=1, run generate_real_depth_thread.py after generation with this worker count")
    parser.add_argument("--post-process-python", type=str, default=None,
                        help="Optional Python interpreter used for post-processing")
    # BlenderProc kopiuje skrypt do katalogu tymczasowego, wiec __file__
    # nie wskazuje oryginalnej lokalizacji. Uzyj sciezki absolutnej.
    default_config = "/home/hampthamanta/code_workspace/magisterka/dataset_generator/optimalized_generator/config.yaml"
    parser.add_argument("--config", type=str,
                        default=default_config,
                        help="Path to config.yaml")
    args = parser.parse_args()
    if args.post_process < 0:
        raise ValueError("--post-process must be >= 0")

    config_path_resolved = str(Path(args.config).expanduser().resolve())

    # Wczytanie konfiguracji
    cfg = load_config(config_path_resolved)
    cam_params = compute_camera_params()

    # Inicjalizacja BlenderProc
    try:
        bproc.init()
    except Exception:
        pass

    # Ustawienie ziarna losowania
    random.seed(args.seed)
    np.random.seed(args.seed)

    mode_str = "DEBUG (scene only)" if args.debug else "GENERATE"
    print(f"\n{'='*60}")
    print(f"  DATA GENERATOR v{VERSION}")
    print(f"{'='*60}")
    print(f"[CONFIG] mode={mode_str}, seed={args.seed}, "
          f"num_samples={args.num_samples}, num_repeats={args.num_repeats}")
    print(f"[CONFIG] output={cfg['output']}")

    # Wykonanie kolejnych powtorzen
    successful_runs = []
    for i in range(args.num_repeats):
        try:
            out_path = run_single_repeat(cfg, cam_params, i, args.seed, args.num_samples,
                                         debug=args.debug, physics=args.physics, hdri=args.hdri)
        except Exception as e:
            print(f"[FATAL] Repeat {i} failed: {e}")
            raise
        if out_path:
            successful_runs.append(out_path)

    if successful_runs:
        print("\n" + "=" * 60)
        print("  GENERATION SUMMARY")
        print("=" * 60)
        print(f"Successfully generated {len(successful_runs)} runs:")
        for path in successful_runs:
            print(f"  - {path}")
        print("=" * 60 + "\n")

    if args.post_process >= 1:
        if not POST_PROCESS_SCRIPT.exists():
            raise FileNotFoundError(f"Post-process script not found: {POST_PROCESS_SCRIPT}")

        post_process_python = args.post_process_python
        if not post_process_python:
            venv_path = os.environ.get("VIRTUAL_ENV")
            if venv_path:
                venv_python = Path(venv_path) / "bin" / "python"
                if venv_python.exists():
                    post_process_python = str(venv_python)
        if not post_process_python:
            post_process_python = shutil.which("python3") or shutil.which("python")
        if not post_process_python:
            post_process_python = sys.executable

        cmd = [
            post_process_python,
            str(POST_PROCESS_SCRIPT),
            "--config",
            config_path_resolved,
            "--path",
            cfg["output"],
            "--workers",
            str(args.post_process),
        ]
        print(
            f"[POST] Running post-process with {args.post_process} workers "
            f"using interpreter: {post_process_python}"
        )
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Post-process failed with code {result.returncode}")
        print("[POST] Post-process finished successfully.")


if __name__ == "__main__":
    main()

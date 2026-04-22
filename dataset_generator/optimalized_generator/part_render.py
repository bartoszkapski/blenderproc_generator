import blenderproc as bproc
import bpy
# Importy blenderproc/bpy musza byc na poczatku skryptu

import math
import os
import numpy as np

import mathutils
# Biblioteka wewnątrz blendera


DEPTH_CALIB = {
	"resolution": (1024, 1024),
	"fx": 504.752167,
	"fy": 504.695465,
	"cx": 517.601746,
	"cy": 508.529358,
	"px": 0.0035,
}
RGB_CALIB = {
	"resolution": (1920, 1080),
	"fx": 1121.560547,
	"fy": 1121.922607,
	"cx": 939.114746,
	"cy": 534.709290,
	"px": 0.00125,
}

T_DC_OPT = mathutils.Matrix(
	(
		(0.994052, 0.002774, 0.005608, -32.665543 / 1000.0),
		(-0.003367, 0.994064, 0.108743, -0.986931 / 1000.0),
		(-0.005723, -0.108760, 0.994054, 2.863724 / 1000.0),
		(0.0, 0.0, 0.0, 1.0),
	)
)
OPT2BL = mathutils.Matrix(((-1, 0, 0, 0), (0, 1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))


def compute_camera_params():
	"""Wylicza parametry kamer na podstawie kalibracji."""
	w_d, h_d = DEPTH_CALIB["resolution"]
	px_d = DEPTH_CALIB["px"]
	sensor_w_d = w_d * px_d
	sensor_h_d = h_d * px_d
	f_x_mm_d = DEPTH_CALIB["fx"] * px_d
	shift_x_d = (DEPTH_CALIB["cx"] - w_d / 2) / w_d
	shift_y_d = (DEPTH_CALIB["cy"] - h_d / 2) / h_d

	w_r, h_r = RGB_CALIB["resolution"]
	px_r = RGB_CALIB["px"]
	sensor_w_r = w_r * px_r
	sensor_h_r = h_r * px_r
	fov_x_r = 2 * math.atan(w_r / (2 * RGB_CALIB["fx"]))
	fov_y_r = 2 * math.atan(h_r / (2 * RGB_CALIB["fy"]))
	shift_x_r = (RGB_CALIB["cx"] - w_r / 2) / w_r
	shift_y_r = (RGB_CALIB["cy"] - h_r / 2) / h_r

	return {
		"depth": {
			"sensor_w": sensor_w_d,
			"sensor_h": sensor_h_d,
			"f_mm": f_x_mm_d,
			"shift_x": shift_x_d,
			"shift_y": shift_y_d,
			"res": (w_d, h_d),
		},
		"rgb": {
			"sensor_w": sensor_w_r,
			"sensor_h": sensor_h_r,
			"fov_x": fov_x_r,
			"fov_y": fov_y_r,
			"shift_x": shift_x_r,
			"shift_y": shift_y_r,
			"res": (w_r, h_r),
		},
	}


def setup_cameras(cam_params: dict) -> tuple:
	"""Tworzy kamery RGB i Depth w Blenderze z poprawnymi intrinsics."""
	dp = cam_params["depth"]
	rp = cam_params["rgb"]

	depth_cam_data = bpy.data.cameras.new(name="Depth_camera")
	depth_cam = bpy.data.objects.new("Depth_camera", depth_cam_data)
	bpy.context.scene.collection.objects.link(depth_cam)

	depth_cam.data.type = "PANO"
	depth_cam.data.panorama_type = "FISHEYE_EQUISOLID"
	depth_cam.data.fisheye_fov = math.radians(120)
	depth_cam.data.fisheye_lens = dp["f_mm"]
	depth_cam.data.sensor_width = dp["sensor_w"]
	depth_cam.data.sensor_height = dp["sensor_h"]
	depth_cam.data.shift_x = -dp["shift_x"]
	depth_cam.data.shift_y = dp["shift_y"]
	depth_cam.data.clip_start = 0.1
	depth_cam.data.clip_end = 10.0

	rgb_cam_data = bpy.data.cameras.new(name="RGB_camera")
	rgb_cam = bpy.data.objects.new("RGB_camera", rgb_cam_data)
	bpy.context.scene.collection.objects.link(rgb_cam)

	rgb_cam.data.type = "PERSP"
	rgb_cam.data.sensor_width = rp["sensor_w"]
	rgb_cam.data.sensor_height = rp["sensor_h"]
	rgb_cam.data.lens_unit = "FOV"
	rgb_cam.data.angle_x = rp["fov_x"]
	rgb_cam.data.angle_y = rp["fov_y"]
	rgb_cam.data.shift_x = -rp["shift_x"]
	rgb_cam.data.shift_y = rp["shift_y"]
	rgb_cam.data.clip_start = 0.01
	rgb_cam.data.clip_end = 1000.0

	t_bl = OPT2BL @ T_DC_OPT @ OPT2BL
	rgb_cam.parent = None
	rgb_cam.matrix_world = depth_cam.matrix_world @ t_bl

	bpy.context.scene.camera = depth_cam
	return depth_cam, rgb_cam


def move_cameras_to_pose(depth_cam, rgb_cam, location, rotation_matrix):
	"""Ustawia poze kamery depth, a kamera RGB podaza transformacja zewnetrzna."""
	cam_pose = bproc.math.build_transformation_mat(location, rotation_matrix)
	depth_cam.matrix_world = mathutils.Matrix(cam_pose.tolist())
	t_bl = OPT2BL @ T_DC_OPT @ OPT2BL
	rgb_cam.matrix_world = depth_cam.matrix_world @ t_bl


def _apply_pose_to_cameras(depth_cam, rgb_cam, pose_mat: np.ndarray):
	"""Ustawia poze obu kamer i odswieza widok sceny."""
	depth_cam.matrix_world = mathutils.Matrix(pose_mat.tolist())
	t_bl = OPT2BL @ T_DC_OPT @ OPT2BL
	rgb_cam.matrix_world = depth_cam.matrix_world @ t_bl
	bpy.context.view_layer.update()


def sample_camera_poses(scene_objs: list, loaded_objects: list, num_samples: int, depth_cam, rgb_cam, settings: dict = None) -> list:
	"""Losuje poprawne pozy kamer i zwraca macierze poz."""
	all_world_points = []
	for obj in scene_objs:
		try:
			bl_obj = obj.blender_obj if hasattr(obj, "blender_obj") else obj
			if bl_obj is None or not hasattr(bl_obj, "bound_box"):
				continue
			corners_world = np.array(
				[(bl_obj.matrix_world @ mathutils.Vector(corner)).to_tuple() for corner in bl_obj.bound_box],
				dtype=np.float64,
			)
			if corners_world.shape == (8, 3):
				all_world_points.append(corners_world)
		except Exception:
			pass

	if not all_world_points:
		print("[WARN] Could not compute scene bounding box")
		return []

	world_pts = np.concatenate(all_world_points, axis=0)
	raw_min = np.min(world_pts, axis=0)
	raw_max = np.max(world_pts, axis=0)
	q_low = np.percentile(world_pts, 2.0, axis=0)
	q_high = np.percentile(world_pts, 98.0, axis=0)
	if np.all(q_high > q_low):
		scene_min = q_low
		scene_max = q_high
	else:
		scene_min = raw_min
		scene_max = raw_max

	scene_center = (scene_min + scene_max) / 2.0
	scene_size = np.maximum(scene_max - scene_min, np.array([1e-3, 1e-3, 1e-3]))

	print(f"[INFO] Scene bounds: min={scene_min}, max={scene_max}")
	print(f"[INFO] Raw scene bounds: min={raw_min}, max={raw_max}")
	print(f"[INFO] Scene center: {scene_center}, size: {scene_size}")

	cfg_settings = settings or {}
	floor_z = scene_min[2]

	cs_cfg = cfg_settings.get("camera_sampling", {})
	cam_z_min_pct = float(cs_cfg.get("cam_z_min_percent", 0.10))
	cam_z_max_pct = float(cs_cfg.get("cam_z_max_percent", 0.75))
	cam_z_min = floor_z + scene_size[2] * cam_z_min_pct
	cam_z_max = floor_z + scene_size[2] * cam_z_max_pct

	print(f"[INFO] Camera height range: {cam_z_min:.2f} - {cam_z_max:.2f} (floor={floor_z:.2f}, ceiling={scene_max[2]:.2f})")

	sampled_poses = []
	obj_targets = []
	for obj in scene_objs + loaded_objects:
		try:
			loc = obj.get_location()
			z = loc[2]
			if z < scene_min[2] + scene_size[2] * 0.7:
				obj_targets.append(np.array(loc))
		except Exception:
			pass
	if not obj_targets:
		obj_targets = [scene_center.copy()]

	print(f"[INFO] {len(obj_targets)} potential look-at targets")

	scene_diag = np.linalg.norm(scene_size)
	scale_factor = min(1.0, 5.0 / scene_diag)
	min_dist = scene_diag * 0.05 * scale_factor
	scene_scale_multiplier = max(1.0, scene_diag / 8.0)

	base_max_background_dist_m = float(cs_cfg.get("max_background_dist_m", 6.0))
	preferred_target_dist = base_max_background_dist_m * scene_scale_multiplier

	def cast_rays(origin_vec, forward_vec, depsgraph, grid_size=5):
		"""Rzuca siatke promieni i zbiera trafienia geometrii."""
		hits = []
		fwd = forward_vec.normalized()
		up = mathutils.Vector((0, 0, 1))
		if abs(fwd.dot(up)) > 0.9:
			up = mathutils.Vector((1, 0, 0))
		right = fwd.cross(up).normalized()
		up = right.cross(fwd).normalized()

		spread = 0.35
		directions = []
		for dx in np.linspace(-spread, spread, grid_size):
			for dy in np.linspace(-spread, spread, grid_size):
				d = (fwd + right * dx + up * dy).normalized()
				directions.append(d)

		for d in directions:
			hit, hit_loc, _, _, obj, _ = bpy.context.scene.ray_cast(depsgraph, origin_vec, d)
			if hit:
				hits.append({"dist": (hit_loc - origin_vec).length, "name": obj.name if obj else ""})
		return hits

	def sample_one(force_accept=False, relax_level=0):
		"""Probkuje jedna poze kamery i dodaje ja, jesli spelnia warunki."""
		height = np.random.uniform(cam_z_min, cam_z_max)
		margin_normal = float(cs_cfg.get("margin_normal", 0.15))
		margin_force = float(cs_cfg.get("margin_force_accept", 0.05))
		margin = margin_force if force_accept else margin_normal

		x = np.random.uniform(scene_min[0] + scene_size[0] * margin, scene_max[0] - scene_size[0] * margin)
		y = np.random.uniform(scene_min[1] + scene_size[1] * margin, scene_max[1] - scene_size[1] * margin)
		location = np.array([x, y, height])

		min_inter_pose_dist = scene_diag / (num_samples + 1) * 0.5
		for prev_pose in sampled_poses:
			prev_loc = prev_pose[:3, 3]
			if np.linalg.norm(location - prev_loc) < min_inter_pose_dist:
				return

		nearby_targets = [t for t in obj_targets if np.linalg.norm(t - location) <= preferred_target_dist]
		if nearby_targets:
			target = nearby_targets[np.random.randint(0, len(nearby_targets))].copy()
		else:
			target = min(obj_targets, key=lambda t: np.linalg.norm(t - location)).copy()

		jitter_xy = min(scene_size[0] * 0.05, preferred_target_dist * 0.15)
		jitter_yy = min(scene_size[1] * 0.05, preferred_target_dist * 0.15)
		jitter_zy = min(scene_size[2] * 0.05, preferred_target_dist * 0.10)
		target += np.array(
			[
				np.random.uniform(-jitter_xy, jitter_xy),
				np.random.uniform(-jitter_yy, jitter_yy),
				np.random.uniform(-jitter_zy, jitter_zy),
			]
		)

		rotation_matrix = bproc.camera.rotation_from_forward_vec(target - location)
		cam_pose_mat = bproc.math.build_transformation_mat(location, rotation_matrix)

		perturb_cfg = (settings or {}).get("camera_perturbation", {})
		p = perturb_cfg.get("probability", 0)
		if force_accept:
			p = max(p, 0.6)
		if p > 0 and np.random.random() < p:
			pitch_range = perturb_cfg.get("pitch_range", [-15, 15])
			roll_range = perturb_cfg.get("roll_range", [-10, 10])
			pitch = math.radians(np.random.uniform(*pitch_range))
			roll = math.radians(np.random.uniform(*roll_range))
			rot_perturb = mathutils.Euler((pitch, roll, 0), "XYZ").to_matrix().to_4x4()
			cam_pose_mat_m = mathutils.Matrix(cam_pose_mat.tolist()) @ rot_perturb
			cam_pose_mat = np.array(cam_pose_mat_m)

		forward = np.array(cam_pose_mat[:3, 2]) * (-1)
		depsgraph = bpy.context.evaluated_depsgraph_get()
		origin = mathutils.Vector(location.tolist())
		fwd_vec = mathutils.Vector(forward.tolist())

		hit_info = cast_rays(origin, fwd_vec, depsgraph)
		hit_dists = [h["dist"] for h in hit_info]
		num_rays_total = 25

		ray_hit_threshold = 0.4 if relax_level == 0 else (0.3 if relax_level == 1 else 0.2)
		if len(hit_dists) < num_rays_total * ray_hit_threshold:
			return

		if all(d < min_dist for d in hit_dists):
			return

		if len(hit_dists) >= 5:
			dist_std = np.std(hit_dists)
			dist_mean = np.mean(hit_dists)
			if dist_mean > 0 and dist_std / dist_mean < 0.05:
				return

		prox_cfg = cfg_settings.get("camera_proximity", {})
		prox_min_obj = int(prox_cfg.get("min_objects", 3))

		base_min_dist = float(prox_cfg.get("min_dist", 0.5))
		base_max_dist = float(prox_cfg.get("max_dist", 2.0))
		prox_min_dist = base_min_dist * scene_scale_multiplier
		prox_max_dist = base_max_dist * scene_scale_multiplier

		if relax_level == 1:
			prox_min_dist *= 0.8
			prox_max_dist *= 1.5
			prox_min_obj = max(1, prox_min_obj - 1)
		elif relax_level >= 2:
			prox_min_dist *= 0.5
			prox_max_dist *= 2.5
			prox_min_obj = 1

		close_objects = set()
		for h in hit_info:
			d = h["dist"]
			if prox_min_dist <= d <= prox_max_dist:
				name_lower = h["name"].lower()
				if not any(w in name_lower for w in ["wall", "floor", "ceiling", "room", "plane", "baseboard"]):
					if h["name"]:
						close_objects.add(h["name"])

		min_obj_requirement = 1 if force_accept else prox_min_obj
		if len(close_objects) < min_obj_requirement:
			return

		sampled_poses.append(cam_pose_mat)

	prox_cfg = cfg_settings.get("camera_proximity", {})
	max_attempts = int(prox_cfg.get("max_attempts", 400))
	attempts = 0

	for stage, relax_level in enumerate([0, 1, 2]):
		stage_attempts = 0
		while len(sampled_poses) < num_samples and stage_attempts < max_attempts:
			try:
				force_accept = attempts > 0 and attempts % 10 == 0
				sample_one(force_accept=force_accept, relax_level=relax_level)
			except Exception as e:
				print(f"[WARN] Camera pose sampling failed: {e}")
			attempts += 1
			stage_attempts += 1

		if len(sampled_poses) >= num_samples:
			break

		print(f"[WARN] Sampling stage {stage + 1} failed ({len(sampled_poses)}/{num_samples}). Relaxing constraints...")

	if sampled_poses:
		_apply_pose_to_cameras(depth_cam, rgb_cam, sampled_poses[-1])

	print(f"[INFO] Registered {len(sampled_poses)}/{num_samples} camera poses (after {attempts} attempts)")
	return sampled_poses


def _ensure_single_composite_node(scene):
	"""Pilnuje, aby w kompozytorze byl dokladnie jeden node Composite."""
	scene.use_nodes = True
	tree = scene.node_tree

	composites = [n for n in tree.nodes if n.bl_idname == "CompositorNodeComposite"]
	if len(composites) == 0:
		comp = tree.nodes.new(type="CompositorNodeComposite")
		comp.location = (300, 0)
	elif len(composites) > 1:
		for node in composites[1:]:
			tree.nodes.remove(node)

	rlayers = [n for n in tree.nodes if n.bl_idname == "CompositorNodeRLayers"]
	if not rlayers:
		rl = tree.nodes.new(type="CompositorNodeRLayers")
		rl.location = (0, 0)
	else:
		rl = rlayers[0]

	comp = [n for n in tree.nodes if n.bl_idname == "CompositorNodeComposite"][0]
	if not comp.inputs[0].is_linked and "Image" in rl.outputs:
		tree.links.new(rl.outputs["Image"], comp.inputs[0])


def configure_bproc_optix_renderer():
	"""Konfiguruje renderer BlenderProc pod denoising OPTIX."""
	_ensure_single_composite_node(bpy.context.scene)
	bproc.renderer.enable_normals_output()
	bproc.renderer.set_max_amount_of_samples(100)
	bproc.renderer.set_noise_threshold(0.05)
	bproc.renderer.set_render_devices(desired_gpu_device_type="OPTIX")
	bproc.renderer.set_denoiser("OPTIX")
	print("[INFO] Configured bproc renderer: 100 samples, noise threshold 0.05, OPTIX denoiser")


def _reset_bproc_camera_keyframes():
	"""Czyści keyframe'y kamer w sposob zgodny z BlenderProc."""
	try:
		bproc.utility.reset_keyframes()
		return
	except Exception:
		pass

	for obj in bpy.data.objects:
		if obj.type == "CAMERA" and obj.animation_data is not None:
			obj.animation_data_clear()


def render_rgb_pass(depth_cam, rgb_cam, cam_params: dict, output_dir: str, poses: list, run_name: str, cfg: dict):
	"""Renderuje obrazy RGB i zapisuje je jako rgb_*.png."""
	import cv2

	rp = cam_params["rgb"]
	settings = cfg.get("settings", {})
	scene = bpy.context.scene
	rgb_res_cfg = settings.get("resolution_RGB", rp["res"])
	rgb_w = int(rgb_res_cfg[0]) if isinstance(rgb_res_cfg, (list, tuple)) and len(rgb_res_cfg) == 2 else int(rp["res"][0])
	rgb_h = int(rgb_res_cfg[1]) if isinstance(rgb_res_cfg, (list, tuple)) and len(rgb_res_cfg) == 2 else int(rp["res"][1])
	rgb_w = max(64, rgb_w)
	rgb_h = max(64, rgb_h)

	scene.render.resolution_x = rgb_w
	scene.render.resolution_y = rgb_h
	scene.render.resolution_percentage = 100
	scene.camera = rgb_cam

	_reset_bproc_camera_keyframes()
	for pose in poses:
		_apply_pose_to_cameras(depth_cam, rgb_cam, pose)
		rgb_pose = np.array(rgb_cam.matrix_world, dtype=np.float64)
		bproc.camera.add_camera_pose(rgb_pose)

	print(f"[INFO] RGB output resolution: {rgb_w}x{rgb_h}")
	data = bproc.renderer.render()
	colors = data.get("colors", [])

	for i, img in enumerate(colors):
		if img is None:
			continue
		rgb_img = np.array(img)
		if rgb_img.ndim == 3 and rgb_img.shape[2] == 4:
			rgb_img = rgb_img[:, :, :3]
		if np.issubdtype(rgb_img.dtype, np.floating):
			rgb_img = np.clip(rgb_img, 0.0, 1.0) * 255.0
		rgb_img = rgb_img.astype(np.uint8)
		out_path = os.path.join(output_dir, f"rgb_{i}.png")
		cv2.imwrite(out_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))


def render_depth_pass(depth_cam, rgb_cam, cam_params: dict, output_dir: str, poses: list, run_name: str):
	"""Renderuje depth ground-truth i zapisuje jako uint16 mm depth_*.png."""
	import cv2

	dp = cam_params["depth"]
	scene = bpy.context.scene
	scene.camera = depth_cam
	scene.render.resolution_x = dp["res"][0]
	scene.render.resolution_y = dp["res"][1]
	scene.render.resolution_percentage = 100

	_reset_bproc_camera_keyframes()
	for pose in poses:
		_apply_pose_to_cameras(depth_cam, rgb_cam, pose)
		depth_pose = np.array(depth_cam.matrix_world, dtype=np.float64)
		bproc.camera.add_camera_pose(depth_pose)

	try:
		bproc.renderer.enable_depth_output(activate_antialiasing=False)
	except Exception:
		pass

	print(f"  [Depth Rendering] {len(poses)} samples")
	data = bproc.renderer.render()
	depth_maps = data.get("depth", None)
	if depth_maps is None:
		depth_maps = data.get("distance", [])

	if len(depth_maps) != len(poses):
		raise RuntimeError(
			f"Depth render returned {len(depth_maps)} maps for {len(poses)} poses. "
			"Aborting to avoid inconsistent or corrupted depth outputs."
		)

	missing = []
	for i, depth_m in enumerate(depth_maps):
		try:
			depth_m = np.array(depth_m, dtype=np.float32)
			depth_m[~np.isfinite(depth_m)] = 0.0
			depth_m[depth_m > 10.0] = 0.0
			depth_m[depth_m < 0.0] = 0.0
			depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
			png_path = os.path.join(output_dir, f"depth_{i}.png")
			if not cv2.imwrite(png_path, depth_mm):
				missing.append(f"depth_{i}.png")
		except Exception:
			missing.append(f"depth_{i}.png")

	if missing:
		print(f"[ERROR] Failed to save {len(missing)} depth PNG files: {missing}")
	else:
		print(f"[SUCCESS] All {len(depth_maps)} depth PNG files created successfully!")

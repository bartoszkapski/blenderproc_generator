import blenderproc as bproc
import bpy
# Importy blenderproc/bpy musza byc na poczatku skryptu

import math
import os
import random
import numpy as np

import mathutils
# Biblioteka wewnątrz blendera



def discover_scenenet_scenes(scenenet_path: str) -> list:
	"""Zwraca liste scen SceneNet jako pary (kategoria_pokoju, sciezka_obj)."""
	scenes = []
	for room_dir in sorted(os.listdir(scenenet_path)):
		room_path = os.path.join(scenenet_path, room_dir)
		if not os.path.isdir(room_path):
			continue
		for fname in os.listdir(room_path):
			if fname.endswith(".obj"):
				scenes.append((room_dir, os.path.join(room_path, fname)))
	return scenes


def discover_blenderkit_scenes(blenderkit_path: str) -> list:
	"""Zwraca liste scen BlenderKit jako pary (folder_sceny, sciezka_blend)."""
	scenes = []
	for scene_dir in sorted(os.listdir(blenderkit_path)):
		scene_path = os.path.join(blenderkit_path, scene_dir)
		if not os.path.isdir(scene_path):
			continue
		for fname in os.listdir(scene_path):
			if fname.endswith(".blend"):
				scenes.append((scene_dir, os.path.join(scene_path, fname)))
	return scenes


def load_scenenet_scene(obj_path: str, cctextures_path: str) -> list:
	"""Wczytuje scene SceneNet i naklada losowe materialy CCTextures."""
	print(f"[INFO] Loading SceneNet scene: {obj_path}")
	objs = bproc.loader.load_obj(obj_path)

	cc_materials = bproc.loader.load_ccmaterials(cctextures_path)
	for obj in objs:
		random_mat = random.choice(cc_materials)
		obj.replace_materials(random_mat)
	print(f"[INFO] Applied CCTextures to {len(objs)} objects")

	return objs


def load_blenderkit_scene(blend_path: str) -> list:
	"""Wczytuje scene BlenderKit z pliku .blend i odtwarza oswietlenie."""
	print(f"[INFO] Loading BlenderKit scene: {blend_path}")
	objs = bproc.loader.load_blend(blend_path)

	lights_loaded = 0
	try:
		with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
			data_to.objects = data_from.objects

		for obj in data_to.objects:
			if obj is None:
				continue
			if obj.type == "LIGHT":
				if obj.name not in bpy.context.scene.collection.objects:
					bpy.context.scene.collection.objects.link(obj)
				if hasattr(obj.data, "energy"):
					obj.data.energy *= round(random.uniform(1.2, 4.0),2)
					# losowe wzmocnienie oświetlenia
				lights_loaded += 1
			else:
				try:
					bpy.data.objects.remove(obj, do_unlink=True)
				except Exception:
					pass
	except Exception as e:
		print(f"[WARN] Failed to extract lights from {blend_path}: {e}")

	print(f"[INFO] Loaded {len(objs)} meshes and {lights_loaded} lights from .blend")
	return objs


def discover_ycb_objects(ycb_path: str) -> list:
	"""Wyszukuje wszystkie sciezki obiektow YCB mozliwych do wczytania."""
	obj_paths = []
	if not os.path.isdir(ycb_path):
		print(f"[WARN] YCB path not found: {ycb_path}")
		return obj_paths

	for folder in sorted(os.listdir(ycb_path)):
		folder_path = os.path.join(ycb_path, folder)
		if not os.path.isdir(folder_path):
			continue

		candidates = [
			os.path.join(folder_path, "google_16k", "textured.obj"),
			os.path.join(folder_path, "tsdf", "textured.obj"),
			os.path.join(folder_path, "poisson", "textured.obj"),
		]
		for cand in candidates:
			if os.path.exists(cand):
				obj_paths.append((folder, cand))
				break
	return obj_paths


def discover_pix3d_objects(pix3d_path: str) -> list:
	"""Wyszukuje wszystkie pliki model.obj w zbiorze pix3d."""
	obj_paths = []
	if os.path.basename(pix3d_path.rstrip("/")) == "model":
		model_root = pix3d_path
	else:
		model_root = os.path.join(pix3d_path, "model")
	if not os.path.isdir(model_root):
		print(f"[WARN] pix3d model path not found: {model_root}")
		return obj_paths

	for category in sorted(os.listdir(model_root)):
		cat_path = os.path.join(model_root, category)
		if not os.path.isdir(cat_path):
			continue
		for model_dir in sorted(os.listdir(cat_path)):
			model_path = os.path.join(cat_path, model_dir)
			if not os.path.isdir(model_path):
				continue
			for fname in os.listdir(model_path):
				if fname == "model.obj":
					obj_paths.append((f"{category}/{model_dir}", os.path.join(model_path, fname)))
					break
	return obj_paths


def discover_ply_objects(dataset_path: str) -> list:
	"""Wyszukuje wszystkie pliki .ply w katalogach modeli."""
	obj_paths = []
	if not dataset_path or not os.path.exists(dataset_path):
		print(f"[WARN] PLY dataset path not found: {dataset_path}")
		return obj_paths

	for root, dirs, files in os.walk(dataset_path):
		folder_name = os.path.basename(root)
		if folder_name in ["models", "models_cad"]:
			for f in sorted(files):
				if f.endswith(".ply"):
					obj_paths.append((f, os.path.join(root, f)))
	return obj_paths


def load_objects_into_scene(datasets: dict, cfg_settings: dict) -> list:
	"""Wczytuje losowe obiekty ze zbiorow danych zgodnie z czestotliwosciami."""
	freqs = cfg_settings.get("obj_freq", {})

	all_objects = {}
	if "ycb" in datasets:
		all_objects["ycb"] = discover_ycb_objects(datasets["ycb"])
	if "pix3d" in datasets:
		all_objects["pix3d"] = discover_pix3d_objects(datasets["pix3d"])
	for key in ["handal", "hope", "ruapc", "tless"]:
		if key in datasets:
			all_objects[key] = discover_ply_objects(datasets[key])

	loaded_objects = []
	category_id_counter = 1

	for dataset_name, available_objs in all_objects.items():
		if not available_objs:
			continue

		freq = freqs.get(dataset_name, 0.0)
		n_sample = int(len(available_objs) * freq)
		if freq > 0 and n_sample == 0:
			n_sample = 1

		if n_sample > 0:
			print(f"[INFO] Sampling {n_sample}/{len(available_objs)} {dataset_name} objects")
			selected = random.sample(available_objs, min(n_sample, len(available_objs)))

			for name, obj_path in selected:
				try:
					objs = bproc.loader.load_obj(obj_path)
					if not objs:
						continue
					obj = objs[0]
					obj.set_cp("category_id", category_id_counter)
					loaded_objects.append(obj)

					bb = np.array(obj.get_bound_box())
					size = bb.max(axis=0) - bb.min(axis=0)

					if max(size) > 5.0:
						obj.set_scale([0.001, 0.001, 0.001])
						bb = np.array(obj.get_bound_box())
						size = bb.max(axis=0) - bb.min(axis=0)

					print(
						f"  [{dataset_name}] Loaded: {name} "
						f"(size: {size[0]:.3f}x{size[1]:.3f}x{size[2]:.3f})"
					)
				except Exception as e:
					print(f"  [{dataset_name}] Failed to load {name}: {e}")

		category_id_counter += 1

	print(f"[INFO] Loaded {len(loaded_objects)} objects total")
	return loaded_objects


def place_objects_on_surfaces(scene_objs: list, loaded_objects: list):
	"""Rozmieszcza obiekty metoda smart placement i random-drop."""
	if not loaded_objects:
		return

	random.shuffle(loaded_objects)
	n_smart = max(1, len(loaded_objects) // 2)
	smart_objects = loaded_objects[:n_smart]
	drop_objects = loaded_objects[n_smart:]

	print(f"[INFO] Object placement split: {len(smart_objects)} smart, {len(drop_objects)} random-drop")

	_place_objects_smart(scene_objs, smart_objects)
	if drop_objects:
		_place_objects_random_drop(scene_objs, drop_objects)


def _place_objects_random_drop(scene_objs: list, drop_objects: list, z_offset_m: float = 0.2, n_exclude_top: int = 8):
	"""Rozmieszcza obiekty nad losowymi, dopuszczalnymi powierzchniami sceny."""
	obj_info = []
	for obj in scene_objs:
		try:
			bb_arr = np.array(obj.get_bound_box())
			bb_min = bb_arr.min(axis=0)
			bb_max = bb_arr.max(axis=0)
			center_xy = (bb_min[:2] + bb_max[:2]) / 2.0
			extent = bb_max - bb_min
			obj_info.append(
				{
					"obj": obj,
					"z_max": bb_max[2],
					"center_xy": center_xy,
					"extent": extent,
				}
			)
		except Exception:
			pass

	if not obj_info:
		print("[WARN] No scene objects found for random-drop placement")
		return

	obj_info.sort(key=lambda x: x["z_max"], reverse=True)
	eligible = obj_info[n_exclude_top:]

	if not eligible:
		eligible = obj_info[1:] if len(obj_info) > 1 else obj_info
		print(f"[WARN] Not enough scene objects to exclude {n_exclude_top}, using {len(eligible)} eligible targets")

	print(f"[INFO] Random-drop: {len(eligible)} eligible scene targets (excluded top {min(n_exclude_top, len(obj_info))} by height)")

	for obj in drop_objects:
		try:
			obj_bb = np.array(obj.get_bound_box())
			obj_size = obj_bb.max(axis=0) - obj_bb.min(axis=0)
		except Exception:
			obj_size = np.array([0.1, 0.1, 0.1])

		target = random.choice(eligible)
		xy_margin = 0.8
		x = target["center_xy"][0] + random.uniform(-target["extent"][0] * xy_margin * 0.5, target["extent"][0] * xy_margin * 0.5)
		y = target["center_xy"][1] + random.uniform(-target["extent"][1] * xy_margin * 0.5, target["extent"][1] * xy_margin * 0.5)
		z = target["z_max"] + z_offset_m + obj_size[2] * 0.5

		obj.set_location([x, y, z])
		obj.set_rotation_euler([0, 0, random.uniform(0, 2 * math.pi)])

		try:
			target_name = target["obj"].get_name()
		except Exception:
			target_name = "unknown"
		print(f"  {obj.get_name():30s} -> RANDOM-DROP above {target_name} at ({x:.2f}, {y:.2f}, {z:.2f})")


def _place_objects_smart(scene_objs: list, loaded_objects: list):
	"""Rozmieszcza obiekty z unikaniem nakladania sie bryl."""
	all_z_vals = []
	for obj in scene_objs:
		try:
			bb_arr = np.array(obj.get_bound_box())
			all_z_vals.extend(bb_arr[:, 2].tolist())
		except Exception:
			pass

	if all_z_vals:
		scene_z_min = min(all_z_vals)
		scene_z_max = max(all_z_vals)
		scene_height = scene_z_max - scene_z_min
		z_ceiling_threshold = scene_z_min + scene_height * 0.6
	else:
		scene_z_min = 0.0
		z_ceiling_threshold = 2.5

	print(f"[INFO] Surface filtering: z_ceiling_threshold={z_ceiling_threshold:.2f}")

	surfaces = []
	for obj in scene_objs:
		try:
			bb_arr = np.array(obj.get_bound_box())
			bb_min = bb_arr.min(axis=0)
			bb_max = bb_arr.max(axis=0)
			extent = bb_max - bb_min
			z_extent = extent[2]
			max_xy = max(extent[0], extent[1])
			surface_top_z = bb_max[2]

			if max_xy > 0.5 and z_extent < max_xy * 0.3 and surface_top_z < z_ceiling_threshold:
				surfaces.append(obj)
		except Exception:
			pass

	if not surfaces:
		print("[WARN] No suitable surfaces found - placing objects at scene center")
		center_z = scene_z_min + (scene_height * 0.3 if all_z_vals else 0.5)
		for obj in loaded_objects:
			obj.set_location([random.uniform(-2, 2), random.uniform(-2, 2), center_z])
		return

	print(f"[INFO] Found {len(surfaces)} placement surfaces (below z={z_ceiling_threshold:.2f})")

	placed_positions = []
	n_floor = max(1, int(len(loaded_objects) * 0.3))
	random.shuffle(loaded_objects)
	floor_objects = loaded_objects[:n_floor]
	surface_objects = loaded_objects[n_floor:]

	print(f"[INFO] Placing {n_floor} objects on floor, {len(surface_objects)} on surfaces")

	bb_list = []
	for o in scene_objs:
		try:
			bb_list.append(np.array(o.get_bound_box()))
		except Exception:
			pass
	if bb_list:
		scene_bb_min = np.min([bb.min(axis=0) for bb in bb_list], axis=0)
		scene_bb_max = np.max([bb.max(axis=0) for bb in bb_list], axis=0)
	else:
		scene_bb_min = np.array([-5, -5, 0])
		scene_bb_max = np.array([5, 5, 3])

	floor_z = scene_z_min
	floor_margin = 0.15

	def place_object(obj, x, y, z, label):
		"""Ustawia obiekt w pozycji i zapisuje go do listy juz rozmieszczonych."""
		try:
			obj_bb = np.array(obj.get_bound_box())
			obj_size = obj_bb.max(axis=0) - obj_bb.min(axis=0)
			obj_radius = max(obj_size[0], obj_size[1]) * 0.5
		except Exception:
			obj_radius = 0.1

		obj.set_location([x, y, z])
		obj.set_rotation_euler([0, 0, random.uniform(0, 2 * math.pi)])
		placed_positions.append((x, y, obj_radius))
		print(f"  {obj.get_name():30s} -> {label} at ({x:.2f}, {y:.2f}, {z:.2f})")

	for obj in floor_objects:
		try:
			obj_bb = np.array(obj.get_bound_box())
			obj_size = obj_bb.max(axis=0) - obj_bb.min(axis=0)
			obj_height = obj_size[2]
			obj_radius = max(obj_size[0], obj_size[1]) * 0.5
		except Exception:
			obj_height = 0.1
			obj_radius = 0.1

		best_pos = None
		best_min_dist = -1
		for _ in range(20):
			x = random.uniform(
				scene_bb_min[0] + (scene_bb_max[0] - scene_bb_min[0]) * floor_margin,
				scene_bb_max[0] - (scene_bb_max[0] - scene_bb_min[0]) * floor_margin,
			)
			y = random.uniform(
				scene_bb_min[1] + (scene_bb_max[1] - scene_bb_min[1]) * floor_margin,
				scene_bb_max[1] - (scene_bb_max[1] - scene_bb_min[1]) * floor_margin,
			)

			depsgraph = bpy.context.evaluated_depsgraph_get()
			ray_start_z = scene_bb_min[2] + (scene_bb_max[2] - scene_bb_min[2]) * 0.3
			ray_origin = mathutils.Vector((x, y, ray_start_z))
			hit, hit_loc, _, _, _, _ = bpy.context.scene.ray_cast(depsgraph, ray_origin, mathutils.Vector((0, 0, -1)))
			z = hit_loc.z + obj_height * 0.5 if hit else floor_z + obj_height * 0.5

			min_neighbor_dist = float("inf")
			for (px, py, pr) in placed_positions:
				dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
				required = pr + obj_radius + obj_radius * 0.1
				min_neighbor_dist = min(min_neighbor_dist, dist - required)

			if min_neighbor_dist > 0:
				best_pos = (x, y, z)
				break
			if min_neighbor_dist > best_min_dist:
				best_pos = (x, y, z)
				best_min_dist = min_neighbor_dist

		if best_pos is None:
			print(f"  [WARN] Failed to find non-overlapping floor space for {obj.get_name()}")
			continue

		x, y, z = best_pos
		place_object(obj, x, y, z, "FLOOR")

	for obj in surface_objects:
		try:
			obj_bb = np.array(obj.get_bound_box())
			obj_size = obj_bb.max(axis=0) - obj_bb.min(axis=0)
			obj_height = obj_size[2]
			obj_radius = max(obj_size[0], obj_size[1]) * 0.5
		except Exception:
			obj_height = 0.1
			obj_radius = 0.1

		best_pos = None
		best_min_dist = -1
		for _ in range(20):
			surface = random.choice(surfaces)
			bb_arr = np.array(surface.get_bound_box())
			s_min = bb_arr.min(axis=0)
			s_max = bb_arr.max(axis=0)
			s_center = (s_min + s_max) / 2.0
			s_extent = s_max - s_min

			x = s_center[0] + random.uniform(-s_extent[0] * 0.3, s_extent[0] * 0.3)
			y = s_center[1] + random.uniform(-s_extent[1] * 0.3, s_extent[1] * 0.3)
			z = s_max[2] + obj_height * 0.5

			min_neighbor_dist = float("inf")
			for (px, py, pr) in placed_positions:
				dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
				required = pr + obj_radius + obj_radius * 0.1
				min_neighbor_dist = min(min_neighbor_dist, dist - required)

			if min_neighbor_dist > 0:
				best_pos = (x, y, z, surface)
				break
			if min_neighbor_dist > best_min_dist:
				best_pos = (x, y, z, surface)
				best_min_dist = min_neighbor_dist

		if best_pos is None:
			print(f"  [WARN] Failed to find non-overlapping surface space for {obj.get_name()}")
			continue

		x, y, z, surface = best_pos
		place_object(obj, x, y, z, surface.get_name())


def setup_hdri_lighting(hdri_path: str):
	"""Ustawia losowe oswietlenie HDRI tla, jesli jest dostepne."""
	if not hdri_path or not os.path.exists(hdri_path):
		print(f"[WARN] HDRI path not found at {hdri_path}! Falling back to simple ambient.")

	if os.path.isdir(hdri_path):
		valid_files = [f for f in os.listdir(hdri_path) if f.lower().endswith((".hdr", ".exr"))]
		if not valid_files:
			print(f"[WARN] No .hdr or .exr files found in {hdri_path}! Falling back to ambient.")
			world = bpy.context.scene.world
			if world is None:
				bpy.ops.world.new()
				world = bpy.data.worlds[-1]
				bpy.context.scene.world = world
			world.use_nodes = True
			bg_node = world.node_tree.nodes.get("Background")
			if bg_node:
				bg_node.inputs["Color"].default_value = (0.05, 0.05, 0.06, 1.0)
				bg_node.inputs["Strength"].default_value = 0.3
			return
		chosen_hdri = os.path.join(hdri_path, random.choice(valid_files))
	else:
		chosen_hdri = hdri_path

	world = bpy.context.scene.world
	if world is None:
		bpy.ops.world.new()
		world = bpy.data.worlds[-1]
		bpy.context.scene.world = world

	strength = random.uniform(0.8, 1.5)
	bproc.world.set_world_background_hdr_img(chosen_hdri, strength=strength)
	print(f"[INFO] Applied HDRI lighting: {os.path.basename(chosen_hdri)} (strength: {strength:.2f})")


def add_soft_center_light_for_scenenet(scene_objs: list):
	"""Dodaje delikatne swiatlo centralne dla scen SceneNet."""
	points = []
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
				points.append(corners_world)
		except Exception:
			pass

	if points:
		world_pts = np.concatenate(points, axis=0)
		bb_min = world_pts.min(axis=0)
		bb_max = world_pts.max(axis=0)
		center = (bb_min + bb_max) * 0.5
		scene_diag = float(np.linalg.norm(bb_max - bb_min))
		scene_height = max(0.5, float(bb_max[2] - bb_min[2]))
		center[2] = bb_min[2] + scene_height * 0.55
	else:
		center = np.array([0.0, 0.0, 1.8], dtype=np.float64)
		scene_diag = 6.0

	light_data = bpy.data.lights.new(name="SceneNet_SoftCenterLight", type="AREA")
	light_data.shape = "SQUARE"
	light_data.size = max(0.8, min(3.0, scene_diag * 0.08))
	light_data.energy = max(120.0, min(420.0, scene_diag * 28.0))

	light_obj = bpy.data.objects.new("SceneNet_SoftCenterLight", light_data)
	bpy.context.scene.collection.objects.link(light_obj)
	light_obj.location = tuple(center.tolist())

	print(
		f"[INFO] Added weak SceneNet center light at "
		f"({light_obj.location.x:.2f}, {light_obj.location.y:.2f}, {light_obj.location.z:.2f}), "
		f"energy={light_data.energy:.1f}, size={light_data.size:.2f}"
	)


def choose_and_load_scene(cfg: dict, hdri: bool = False):
	"""Wybiera i wczytuje scene z konfiguracji oraz opcjonalnie ustawia HDRI."""
	datasets = cfg["datasets"]
	scene_sources = cfg.get("settings", {}).get("scene_sources", {})
	use_scenenet = scene_sources.get("scenenet", True)
	use_blenderkit = scene_sources.get("blenderkit", True)

	scenenet_scenes = discover_scenenet_scenes(datasets["scenenet"]) if use_scenenet else []
	blenderkit_scenes = discover_blenderkit_scenes(datasets["blenderkit"]) if use_blenderkit else []

	print(f"[INFO] SceneNet: {len(scenenet_scenes)} scenes" if use_scenenet else "[INFO] SceneNet: DISABLED")
	print(f"[INFO] BlenderKit: {len(blenderkit_scenes)} scenes" if use_blenderkit else "[INFO] BlenderKit: DISABLED")

	bk_weight = cfg.get("settings", {}).get("blenderkit_weight", 1)
	all_scenes = [(s, "scenenet") for s in scenenet_scenes] + [(s, "blenderkit") for s in blenderkit_scenes] * bk_weight

	if not all_scenes:
		msg = (
			"No scenes found. Check dataset paths and scene_sources in config.yaml "
			f"(SceneNet={len(scenenet_scenes)}, BlenderKit={len(blenderkit_scenes)})."
		)
		print(f"[ERROR] {msg}")
		raise RuntimeError(msg)

	chosen, scene_type = random.choice(all_scenes)
	scene_name, scene_path = chosen
	print(f"[INFO] Selected scene: {scene_name} (type: {scene_type})")

	if scene_type == "scenenet":
		scene_objs = load_scenenet_scene(scene_path, datasets["cctextures"])
		add_soft_center_light_for_scenenet(scene_objs)
	else:
		scene_objs = load_blenderkit_scene(scene_path)

	if hdri:
		setup_hdri_lighting(datasets.get("hdri"))
	else:
		print("[INFO] HDRI lighting DISABLED.")

	return scene_name, scene_type, scene_objs

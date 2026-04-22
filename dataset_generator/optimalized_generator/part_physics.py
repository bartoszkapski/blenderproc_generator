import bpy
import mathutils
# Biblioteka wewnątrz blendera

import numpy as np



def _get_blender_object(obj):
	"""Zwraca obiekt Blender z wrappera BlenderProc albo bezposrednio przekazany obiekt."""
	return obj.blender_obj if hasattr(obj, "blender_obj") else obj



def _collect_scene_z_bounds(scene_objs: list) -> tuple[float, float]:
	"""Wyznacza przyblizone granice Z sceny na podstawie bounding boxow."""
	all_z = []
	for obj in scene_objs:
		try:
			bb = np.array(obj.get_bound_box(), dtype=np.float64)
			all_z.extend(bb[:, 2].tolist())
		except Exception:
			pass

	if not all_z:
		return 0.0, 3.0

	return float(min(all_z)), float(max(all_z))



def simulate_loaded_objects_physics(scene_objs: list, loaded_objects: list):
	"""Szybko osadza nowe obiekty na najblizszej powierzchni pod nimi.

	To jest lekki zamiennik pelnej symulacji rigid body. Dziala tylko na
	nowych obiektach i nie modyfikuje calej sceny.
	"""
	if not loaded_objects:
		return

	scene_floor_z, scene_ceil_z = _collect_scene_z_bounds(scene_objs)
	depsgraph = bpy.context.evaluated_depsgraph_get()
	settled = 0
	fallback = 0

	for obj in loaded_objects:
		bl_obj = _get_blender_object(obj)
		if bl_obj is None:
			continue

		try:
			loc = obj.get_location() if hasattr(obj, "get_location") else list(bl_obj.location)
		except Exception:
			loc = list(bl_obj.location)

		try:
			obj_bb = np.array(obj.get_bound_box(), dtype=np.float64)
			obj_size = obj_bb.max(axis=0) - obj_bb.min(axis=0)
			obj_half_h = float(obj_size[2]) * 0.5
		except Exception:
			obj_half_h = 0.05

		# Startujemy tuz ponizej najwyzszej powierzchni sceny, aby nie trafic w dach,
		# a jednoczesnie zostawic raycastowi mozliwosc znalezienia powierzchni pod obiektem.
		origin_z = min(loc[2] + max(0.02, obj_half_h * 0.1), scene_ceil_z - 0.001)
		if origin_z <= scene_floor_z:
			origin_z = scene_floor_z + max(0.5, obj_half_h)

		ray_origin = mathutils.Vector((float(loc[0]), float(loc[1]), float(origin_z)))
		ray_dir = mathutils.Vector((0.0, 0.0, -1.0))

		hit, hit_loc, _, _, hit_obj, _ = bpy.context.scene.ray_cast(depsgraph, ray_origin, ray_dir)

		if hit and hit_obj is not None and hit_obj.name != "__physics_floor":
			new_z = float(hit_loc.z) + obj_half_h - 0.01
			bl_obj.location = (float(loc[0]), float(loc[1]), new_z)
			settled += 1
			try:
				obj_name = obj.get_name()
			except Exception:
				obj_name = getattr(bl_obj, "name", "?")
			print(f"  {obj_name:30s} -> settled on {hit_obj.name} at z={new_z:.3f}")
		else:
			new_z = scene_floor_z + obj_half_h
			bl_obj.location = (float(loc[0]), float(loc[1]), new_z)
			fallback += 1
			try:
				obj_name = obj.get_name()
			except Exception:
				obj_name = getattr(bl_obj, "name", "?")
			print(f"  {obj_name:30s} -> fallback to floor at z={new_z:.3f}")

	print(f"[INFO] Raycast settle: {settled} placed on surfaces, {fallback} on floor fallback")



def drop_objects_raycast(scene_objs: list, loaded_objects: list):
	"""Zachowany alias dla kompatybilnosci wstecznej."""
	simulate_loaded_objects_physics(scene_objs, loaded_objects)

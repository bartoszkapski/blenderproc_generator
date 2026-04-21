import bpy
import mathutils
# Biblioteka wewnątrz blendera

import numpy as np


def drop_objects_raycast(scene_objs: list, loaded_objects: list):
	"""Upuszcza obiekty na powierzchnie sceny za pomoca promieni w dol."""
	if not loaded_objects:
		return

	all_z = []
	for obj in scene_objs:
		try:
			bb = np.array(obj.get_bound_box())
			all_z.extend(bb[:, 2].tolist())
		except Exception:
			pass
	scene_floor_z = min(all_z) if all_z else 0.0
	scene_ceil_z = max(all_z) if all_z else 3.0

	depsgraph = bpy.context.evaluated_depsgraph_get()
	dropped = 0
	fallback = 0

	for obj in loaded_objects:
		try:
			bl_obj = obj.blender_obj if hasattr(obj, "blender_obj") else obj
			loc = obj.get_location() if hasattr(obj, "get_location") else list(bl_obj.location)
		except Exception:
			continue

		try:
			obj_bb = np.array(obj.get_bound_box())
			obj_size = obj_bb.max(axis=0) - obj_bb.min(axis=0)
			obj_half_h = obj_size[2] * 0.5
		except Exception:
			obj_half_h = 0.05

		ray_start_z = max(loc[2] + 0.5, scene_ceil_z + 0.5)
		ray_origin = mathutils.Vector((loc[0], loc[1], ray_start_z))
		ray_dir = mathutils.Vector((0, 0, -1))

		hit, hit_loc, hit_normal, hit_idx, hit_obj, _ = bpy.context.scene.ray_cast(
			depsgraph, ray_origin, ray_dir
		)

		if hit and hit_obj.name != "__physics_floor":
			new_z = hit_loc.z + obj_half_h
			obj.set_location([loc[0], loc[1], new_z])
			dropped += 1
			try:
				obj_name = obj.get_name()
			except Exception:
				obj_name = getattr(bl_obj, "name", "?")
			print(f"  {obj_name:30s} -> dropped onto {hit_obj.name} at z={new_z:.3f}")
		else:
			new_z = scene_floor_z + obj_half_h
			obj.set_location([loc[0], loc[1], new_z])
			fallback += 1
			try:
				obj_name = obj.get_name()
			except Exception:
				obj_name = getattr(bl_obj, "name", "?")
			print(f"  {obj_name:30s} -> fallback to floor at z={new_z:.3f}")

	print(f"[INFO] Raycast drop: {dropped} placed on surfaces, {fallback} on floor fallback")
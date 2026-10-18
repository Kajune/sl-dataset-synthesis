import bpy
import random, argparse, math, sys
import numpy as np
from mathutils import noise
from mathutils import Vector, Euler

def _clear_scene():
	# 既存オブジェクトを削除
	bpy.ops.object.select_all(action='SELECT')
	bpy.ops.object.delete(use_global=False)

	# 不要データブロックをパージ
	for datablocks in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.images, bpy.data.textures):
		for datablock in list(datablocks):
			if datablock.users == 0:
				datablocks.remove(datablock)

def _ensure_render_settings(engine="BLENDER_EEVEE"):
	scene = bpy.context.scene
	scene.render.engine = engine
	if engine == "CYCLES":
		scene.cycles.samples = 128
		scene.cycles.use_adaptive_sampling = True
		scene.cycles.device = 'GPU'
	else:
		scene.eevee.taa_render_samples = 64
		scene.eevee.use_gtao = True
		scene.eevee.use_bloom = True
		scene.eevee.use_ssr = True

def _random_color(hsv=False, min_val=0.0, max_val=1.0):
	if hsv:
		h = random.random()
		s = random.uniform(min_val, max_val)
		v = random.uniform(min_val, max_val)
		i = int(h * 6.0)
		f = h * 6.0 - i
		p = v * (1.0 - s)
		q = v * (1.0 - f * s)
		t = v * (1.0 - (1.0 - f) * s)
		i = i % 6
		if i == 0: r, g, b = v, t, p
		elif i == 1: r, g, b = q, v, p
		elif i == 2: r, g, b = p, v, t
		elif i == 3: r, g, b = p, q, v
		elif i == 4: r, g, b = t, p, v
		else: r, g, b = v, p, q
		return (r, g, b, 1.0)
	else:
		return (random.uniform(min_val, max_val),
				random.uniform(min_val, max_val),
				random.uniform(min_val, max_val), 1.0)

def _create_random_material(
	name_prefix="Mat",
	base_color_range=(0.0,1.0),
	metallic_range=(0.0,1.0),
	roughness_range=(0.0,1.0),
	specular_range=(0.0,1.0),
	subsurface_range=(0.0,0.2)
):
	mat = bpy.data.materials.new(name=f"{name_prefix}_{random.randrange(1_000_000)}")
	mat.use_nodes = True
	nt = mat.node_tree
	for n in nt.nodes:
		nt.nodes.remove(n)

	output = nt.nodes.new("ShaderNodeOutputMaterial")
	output.location = (400, 0)
	bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
	bsdf.location = (0, 0)

	base_color = tuple(random.uniform(*base_color_range) for _ in range(3)) + (1.0,)
	bsdf.inputs["Base Color"].default_value = base_color
	if "Metallic" in bsdf.inputs:
		bsdf.inputs["Metallic"].default_value = random.uniform(*metallic_range)
	if "Roughness" in bsdf.inputs:
		bsdf.inputs["Roughness"].default_value = random.uniform(*roughness_range)
	if "Specular" in bsdf.inputs:
		bsdf.inputs["Specular"].default_value = random.uniform(*specular_range)

	if "Subsurface" in bsdf.inputs and random.random() < 0.2:
		bsdf.inputs["Subsurface"].default_value = random.uniform(*subsurface_range)
		if "Subsurface Color" in bsdf.inputs:
			bsdf.inputs["Subsurface Color"].default_value = tuple(random.random() for _ in range(3)) + (1.0,)

	# ランダムテクスチャ
	if random.random() < 0.7:
		texture_types = ["Noise", "Voronoi", "Wave", "Checker", "Brick"]
		tex_type = random.choice(texture_types)

		if tex_type == "Noise":
			tex_node = nt.nodes.new("ShaderNodeTexNoise")
		elif tex_type == "Voronoi":
			tex_node = nt.nodes.new("ShaderNodeTexVoronoi")
		elif tex_type == "Wave":
			tex_node = nt.nodes.new("ShaderNodeTexWave")
		elif tex_type == "Checker":
			tex_node = nt.nodes.new("ShaderNodeTexChecker")
		elif tex_type == "Brick":
			tex_node = nt.nodes.new("ShaderNodeTexBrick")

		tex_node.location = (-400, 0)
		if "Scale" in tex_node.inputs:
			tex_node.inputs["Scale"].default_value = random.uniform(1.0, 10.0)
		if "Detail" in tex_node.inputs:
			tex_node.inputs["Detail"].default_value = random.uniform(0.5, 8.0)
		if "Distortion" in tex_node.inputs:
			tex_node.inputs["Distortion"].default_value = random.uniform(0.0, 5.0)

		mix_node = nt.nodes.new("ShaderNodeMixRGB")
		mix_node.blend_type = "MIX"
		mix_node.inputs["Fac"].default_value = random.uniform(0.3, 0.8)
		mix_node.location = (-200, 0)

		rgb_node = nt.nodes.new("ShaderNodeRGB")
		rgb_node.outputs[0].default_value = base_color
		rgb_node.location = (-600, 0)

		nt.links.new(rgb_node.outputs["Color"], mix_node.inputs[1])
		nt.links.new(tex_node.outputs["Color"], mix_node.inputs[2])
		nt.links.new(mix_node.outputs["Color"], bsdf.inputs["Base Color"])

	nt.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
	return mat

def smooth_random_normals(obj, normal_strength=0.2, noise_scale=1.0):
	if obj.type != 'MESH':
		return

	mesh = obj.data
	mesh.calc_normals_split()
	new_normals = []

	for loop in mesh.loops:
		orig = loop.normal
		vertex_co = mesh.vertices[loop.vertex_index].co

		# 位置を元にノイズベクトルを生成
		nx = noise.noise_vector(vertex_co * noise_scale)
		rand_vec = Vector((nx[0], nx[1], nx[2])) * normal_strength

		new_normals.append((orig + rand_vec).normalized())

	mesh.normals_split_custom_set(new_normals)
	mesh.use_auto_smooth = True
	mesh.update()

def _add_object(shape, loc, rot, scale, min_size=0.1, max_size=1.0, apply_sculpt=True, randomize_normals=False, normal_strength=10.0):
	if shape == "sphere":
		radius = random.uniform(min_size, max_size)
		bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=radius, location=loc, rotation=rot)
	elif shape == "icosphere":
		radius = random.uniform(min_size, max_size)
		bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=random.randint(3,6), radius=radius, location=loc, rotation=rot)
	elif shape == "cube":
		size = random.uniform(min_size, max_size)
		bpy.ops.mesh.primitive_cube_add(size=size, location=loc, rotation=rot)
	elif shape == "torus":
		major = random.uniform(min_size, max_size)
		minor = random.uniform(min_size/2, max_size/2)
		bpy.ops.mesh.primitive_torus_add(major_radius=major, minor_radius=minor, major_segments=64, minor_segments=32, location=loc, rotation=rot)
	elif shape == "cylinder":
		radius = random.uniform(min_size, max_size)
		depth = random.uniform(min_size, max_size)
		bpy.ops.mesh.primitive_cylinder_add(vertices=random.randint(12,32), radius=radius, depth=depth, location=loc, rotation=rot)
	elif shape == "cone":
		radius = random.uniform(min_size, max_size)
		depth = random.uniform(min_size, max_size)
		bpy.ops.mesh.primitive_cone_add(vertices=random.randint(12,32), radius1=radius, depth=depth, location=loc, rotation=rot)
	elif shape == "prism":
		radius = random.uniform(min_size, max_size)
		depth = random.uniform(min_size, max_size)
		bpy.ops.mesh.primitive_cylinder_add(vertices=random.randint(3,8), radius=radius, depth=depth, location=loc, rotation=rot)
	else:
		size = random.uniform(min_size, max_size)
		bpy.ops.mesh.primitive_cube_add(size=size, location=loc, rotation=rot)

	obj = bpy.context.active_object
	obj.scale = scale

	if apply_sculpt:
		mod = obj.modifiers.new("SculptDisplace", type='DISPLACE')
		tex = bpy.data.textures.new("SculptTex_"+str(random.randrange(1_000_000)), type='CLOUDS')
		tex.noise_scale = random.uniform(0.2,1.0)
		mod.texture = tex
		mod.strength = random.uniform(0.1,0.5)
		bpy.context.view_layer.objects.active = obj
		bpy.ops.object.modifier_apply(modifier=mod.name)

	# 法線ランダム化
	if randomize_normals and obj.type == 'MESH':
		smooth_random_normals(obj, normal_strength)

	return obj


def _random_light(obj_bounds=None):
	light_types = ["POINT", "SUN", "AREA"]
	lt = bpy.data.lights.new(name=f"Light_{random.randrange(1_000_000)}", type=random.choice(light_types))
	base_white = 0.95
	var = 0.1
	lt.color = (random.uniform(base_white-var, base_white+var),
				random.uniform(base_white-var, base_white+var),
				random.uniform(base_white-var, base_white+var))
	if lt.type == "POINT":
		lt.energy = random.uniform(300.0,800.0)
		lt.shadow_soft_size = random.uniform(0.05,1.0)
	elif lt.type == "SUN":
		lt.energy = random.uniform(1.0,10.0)
		lt.angle = random.uniform(math.radians(0.1), math.radians(5.0))
	elif lt.type == "AREA":
		lt.energy = random.uniform(500.0,1500.0)
		lt.shape = random.choice(["SQUARE","RECTANGLE"])
		if lt.shape=="RECTANGLE":
			lt.size = random.uniform(1.0,3.0)
			lt.size_y = random.uniform(1.0,3.0)
		else:
			lt.size = random.uniform(1.0,3.0)
	obj = bpy.data.objects.new(lt.name, lt)
	bpy.context.collection.objects.link(obj)

	if obj_bounds:
		x_min, x_max = obj_bounds[0]
		y_min, y_max = obj_bounds[1]
		z_min, z_max = obj_bounds[2]
		margin = 5.0
		obj.location = (
			random.choice([random.uniform(x_min-margin, x_min-2), random.uniform(x_max+2, x_max+margin)]),
			random.choice([random.uniform(y_min-margin, y_min-2), random.uniform(y_max+2, y_max+margin)]),
			random.uniform(z_max+2, z_max+5)
		)
	else:
		obj.location = (random.uniform(-8.0,8.0), random.uniform(-8.0,8.0), random.uniform(5.0,10.0))
	obj.rotation_euler = Euler((random.uniform(0,math.tau), random.uniform(0,math.tau), random.uniform(0,math.tau)), 'XYZ')
	return obj

def _random_world(world_randomness=1.0):
	world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
	bpy.context.scene.world = world
	world.use_nodes = True
	nt = world.node_tree

	bg = nt.nodes.get("Background") or nt.nodes.new("ShaderNodeBackground")
	bg.location = (-200,0)
	out = nt.nodes.get("World Output") or nt.nodes.new("ShaderNodeOutputWorld")
	out.location = (200,0)
	for l in list(nt.links):
		nt.links.remove(l)

	bg.inputs["Color"].default_value = _random_color(hsv=True, min_val=0.0, max_val=1.0*world_randomness)
	bg.inputs["Strength"].default_value = random.uniform(0.2,1.2)*world_randomness
	nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

def generate_scene(
	seed=None,
	num_frames=250,
	num_objects=50,
	fov_range=(35,90),
	light_range=(0,2),
	bounds=((-6.0,6.0),(-6.0,6.0),(0.5,4.0)),
	no_floor=False,
	engine="BLENDER_EEVEE",
	physics_simulation=False,
	save_path="//random_scene.blend",
	material_ranges=None,
	randomize_normals=False,
	camera_mode="random",
):
	if seed is not None:
		random.seed(seed)

#	_clear_scene()
#	_ensure_render_settings(engine=engine)
#	_random_world()

	shapes = ["sphere","icosphere","cube","torus","cylinder","cone","prism"]
	generated_objects = []

	t = max(0.0, min(1.0, num_objects / 50.0))

	center_bias = 0.7 + 0.3 * t
	scale_bias = 2.0 - 1.0 * t

	for _ in range(num_objects):
		loc = (
			random.uniform(bounds[0][0] * center_bias, bounds[0][1] * center_bias),
			random.uniform(bounds[1][0] * center_bias, bounds[1][1] * center_bias),
			random.uniform(bounds[2][0], bounds[2][1])
		)

		rot = (
			random.uniform(0, math.tau),
			random.uniform(0, math.tau),
			random.uniform(0, math.tau)
		)

		base_scale = random.gauss(1.0 * scale_bias, 0.5)
		scale = (base_scale, base_scale, base_scale)

		shape = random.choice(shapes)
		obj = _add_object(
			shape, loc, rot, scale,
			min_size=material_ranges.get("min_size", 0.1),
			max_size=material_ranges.get("max_size", 1.0) * scale_bias,
			randomize_normals=randomize_normals,
			normal_strength=random.uniform(0.5, 5.0),
		)
		generated_objects.append(obj)

		if obj.data and hasattr(obj.data,"materials"):
			mat = _create_random_material(
				name_prefix="Rand",
				base_color_range=material_ranges.get("base_color",(0.0,1.0)),
				metallic_range=material_ranges.get("metallic",(0.0,1.0)),
				roughness_range=material_ranges.get("roughness",(0.0,1.0)),
				specular_range=material_ranges.get("specular",(0.0,1.0)),
				subsurface_range=material_ranges.get("subsurface",(0.0,0.2))
			)
			if len(obj.data.materials)==0:
				obj.data.materials.append(mat)
			else:
				obj.data.materials[0] = mat

	# 床
	if random.random() > 0.5 and not no_floor:
		bpy.ops.mesh.primitive_plane_add(size=50, location=(0,0,0))
		floor = bpy.context.active_object
		floor.name="Floor"
		bpy.ops.rigidbody.object_add()
		floor.rigid_body.type='PASSIVE'
		floor.rigid_body.friction=0.5
		floor.rigid_body.restitution=0.2

	if physics_simulation:
		for obj in generated_objects:
			bpy.context.view_layer.objects.active=obj
			bpy.ops.rigidbody.object_add()
			obj.rigid_body.type='ACTIVE'
			obj.rigid_body.mass=random.uniform(0.5,2.0)
			obj.rigid_body.friction=0.5
			obj.rigid_body.restitution=0.3

		scene = bpy.context.scene
		scene.frame_start=1
		scene.frame_end=num_frames
		for frame in range(scene.frame_start, scene.frame_end+1):
			scene.frame_set(frame)

	num_lights = random.randint(*light_range)
	for _ in range(num_lights):
		_random_light(bounds)

	# カメラ（必要なら）
	# シーン内のカメラを検索
	cam_obj = None
	for obj in bpy.context.scene.objects:
		if obj.type == 'CAMERA':
			cam_obj = obj
			break

	if cam_obj is None:
		raise ValueError("シーン内にカメラが存在しません")

	cam_obj = bpy.context.scene.camera
	print(cam_obj)
	angle_deg = random.uniform(*fov_range)
	cam_obj.data.angle = math.radians(angle_deg)

	# ターゲットをオブジェクト群の中心に設定
	if generated_objects:
		avg_loc = [0.0, 0.0, 0.0]
		for obj in generated_objects:
			avg_loc[0] += obj.location[0]
			avg_loc[1] += obj.location[1]
			avg_loc[2] += obj.location[2]
		avg_loc = [c / len(generated_objects) for c in avg_loc]

	target = bpy.data.objects.new("CameraTarget", None)
	bpy.context.collection.objects.link(target)
	target.location = avg_loc

	# Track To 制約
	trk = cam_obj.constraints.new('TRACK_TO')
	trk.target = target
	trk.track_axis = 'TRACK_NEGATIVE_Z'
	trk.up_axis = 'UP_Y'

	# アニメーション設定
	scene = bpy.context.scene
	scene.frame_start = 1
	scene.frame_end = num_frames

	if camera_mode == "random":
		num_keypoints = random.randint(2, 5)
		for i in range(num_keypoints):
			frame = scene.frame_start + i * (scene.frame_end - scene.frame_start) / (num_keypoints - 1)
			loc = (
				avg_loc[0] + random.uniform(-12.0, 12.0),
				avg_loc[1] + random.uniform(-12.0, 12.0),
				avg_loc[2] + random.uniform(5.0, 10.0)
			)
			cam_obj.location = loc
			cam_obj.keyframe_insert(data_path="location", frame=frame)

	elif camera_mode == "round":
		num_keypoints = 12  # 円運動の分割数
		radius = random.uniform(bounds[0][1], bounds[0][1] * 3)       # 中心からの距離
		height = np.mean(bounds[2])
		for i in range(num_keypoints):
			frame = scene.frame_start + i * (scene.frame_end - scene.frame_start) / (num_keypoints - 1)
			angle = 2 * math.pi * i / num_keypoints
			loc = (
				avg_loc[0] + radius * math.cos(angle),
				avg_loc[1] + radius * math.sin(angle),
				height
			)
			cam_obj.location = loc
			cam_obj.keyframe_insert(data_path="location", frame=frame)

	bpy.ops.object.select_all(action='DESELECT')
	bpy.ops.wm.save_as_mainfile(filepath=save_path)
	return {"objects": num_objects, "lights": num_lights, "path": save_path}


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--template', type=str, default='scenes/template_scene.blend')
	parser.add_argument('--output', type=str, default='scenes/random_scene.blend')
	parser.add_argument('--max_num_objects', type=int, default=200)
	parser.add_argument('--num_objects', type=int, default=None)
	parser.add_argument('--num_frames', type=int, default=250)
	parser.add_argument('--fov_range', type=int, nargs=2, default=[35,90])
	parser.add_argument('--seed', type=int, default=None)
	parser.add_argument('--enable_physics', action='store_true')
	parser.add_argument('--min_size', type=float, default=0.1)
	parser.add_argument('--max_size', type=float, default=1.0)
	parser.add_argument('--light_range', type=int, nargs=2, default=[0,2])
	parser.add_argument('--no_floor', action='store_true')
	parser.add_argument('--world_randomness', type=float, default=1.0)
	parser.add_argument('--base_color_min', type=float, default=0.0)
	parser.add_argument('--base_color_max', type=float, default=1.0)
	parser.add_argument('--metallic_min', type=float, default=0.0)
	parser.add_argument('--metallic_max', type=float, default=1.0)
	parser.add_argument('--roughness_min', type=float, default=0.0)
	parser.add_argument('--roughness_max', type=float, default=1.0)
	parser.add_argument('--specular_min', type=float, default=0.0)
	parser.add_argument('--specular_max', type=float, default=1.0)
	parser.add_argument('--subsurface_min', type=float, default=0.0)
	parser.add_argument('--subsurface_max', type=float, default=0.2)
	parser.add_argument('--randomize_normals', action='store_true')
	parser.add_argument('--camera_mode', type=str, default='random',
						choices=['random', 'round'],
						help="Camera motion mode: 'random' or 'round'")
	args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])

	bpy.ops.wm.open_mainfile(filepath=args.template)

	if args.num_objects is not None:
		num_objects = args.num_objects
	else:
		low, high = 1, args.max_num_objects
		log_low, log_high = np.log(low), np.log(high)
		r = random.uniform(log_low, log_high)
		num_objects = int(round(np.exp(r)))

	material_ranges = {
		"min_size": args.min_size,
		"max_size": args.max_size,
		"base_color": (args.base_color_min, args.base_color_max),
		"metallic": (args.metallic_min, args.metallic_max),
		"roughness": (args.roughness_min, args.roughness_max),
		"specular": (args.specular_min, args.specular_max),
		"subsurface": (args.subsurface_min, args.subsurface_max)
	}

	res = generate_scene(
		seed=args.seed,
		num_frames=args.num_frames,
		num_objects=num_objects,
		fov_range=args.fov_range,
		light_range=args.light_range,
		no_floor=args.no_floor,
		engine="CYCLES",
		physics_simulation=args.enable_physics,
		save_path=args.output,
		material_ranges=material_ranges,
		randomize_normals=args.randomize_normals,
		camera_mode=args.camera_mode,
	)
	print("Scene generated:", res)

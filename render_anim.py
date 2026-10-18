# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.

import argparse, sys, os, random, json
import bpy
from math import radians
import mathutils
import numpy as np
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
	sys.path.append(script_dir)

from pattern_augmentation import augment_image


def enable_gpus(device_type, use_cpus=True):
	preferences = bpy.context.preferences
	cycles_preferences = preferences.addons["cycles"].preferences
	cycles_preferences.refresh_devices()
	devices = cycles_preferences.devices

	if not devices:
		raise RuntimeError("Unsupported device type")

	activated_gpus = []
	for device in devices:
		if device.type == "CPU":
			device.use = use_cpus
		else:
			device.use = True
			activated_gpus.append(device.name)
			print('activated gpu', device.name)

	cycles_preferences.compute_device_type = device_type
	bpy.context.scene.cycles.device = "GPU"

	return activated_gpus


def listify_matrix(matrix):
	matrix_list = []
	for row in matrix:
		matrix_list.append(list(row))
	return matrix_list


def parent_obj_to_camera(b_camera):
	origin = (0, 0, 0)
	b_empty = bpy.data.objects.new("Empty", None)
	b_empty.location = origin
	b_camera.parent = b_empty  # setup parenting

	scn = bpy.context.scene
	scn.collection.objects.link(b_empty)
	bpy.context.view_layer.objects.active = b_empty
	# scn.objects.active = b_empty
	return b_empty


def blender_to_cv(m):
	return m @ np.float32([
		[1,0,0,0],
		[0,-1,0,0],
		[0,0,-1,0],
		[0,0,0,1]
	])


def float_or_range(values):
	if len(values) == 1:
		return float(values[0])
	elif len(values) == 2:
		lo, hi = map(float, values)
		return random.uniform(lo, hi)
	else:
		raise argparse.ArgumentTypeError("1つか2つのfloat値を指定してください")


if __name__ == '__main__':
	enable_gpus("CUDA")

	parser = argparse.ArgumentParser()
	parser.add_argument('--results_path', type=str, required=True)
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--views', type=int, default=200)
	parser.add_argument('--width', type=int, default=960)
	parser.add_argument('--height', type=int, default=540)
	parser.add_argument('--format', type=str, default='PNG')
	parser.add_argument('--color_depth', type=int, default=8)
	parser.add_argument('--num_samples', type=int, default=330)
	parser.add_argument('--far', type=float, default=32)
	parser.add_argument('--near', type=float, default=0)
	parser.add_argument('--proj_roll', type=float)
	parser.add_argument('--out_depth_normal', action='store_true')

	parser.add_argument('--illum_intensity', nargs='+', type=float, default=[1.0])
	parser.add_argument('--proj_intensity', nargs='+', type=float, default=[1.0])
	parser.add_argument('--custom_pattern', type=str, nargs='+')
	parser.add_argument('--num_proj', type=int, default=4)

	parser.add_argument('--scatter_density', type=float, default=0.0)
	parser.add_argument('--volume_bounce', type=int, default=2)
	parser.add_argument('--flat_shading', action='store_true')
	parser.add_argument('--toon_shading', action='store_true')
	parser.add_argument('--enable_refraction', action='store_true')
	parser.add_argument('--enable_particle', action='store_true')
	parser.add_argument('--blur', nargs='+', type=float, default=[0.0])

	parser.add_argument('--begin', type=int, default=0)
	parser.add_argument('--end', type=int)
	parser.add_argument('--exist_skip', action='store_true')
	parser.add_argument('--scatter_type', type=str, default="hg", choices=["hg", "ff"])
	parser.add_argument('--co_moving_light', action='store_true')
	parser.add_argument('--distance_rendering', action='store_true')

	parser.add_argument('--randomize_projector_pose', action='store_true')
	parser.add_argument('--randomize_pattern', action='store_true')
	args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])

	args.illum_intensity = float_or_range(args.illum_intensity)
	args.proj_intensity = float_or_range(args.proj_intensity)
	args.blur = float_or_range(args.blur)

#	fp = bpy.path.abspath(f"//{args.results_path}")
	fp = os.path.join(os.path.dirname(__file__), args.results_path)


	if not os.path.exists(fp):
		os.makedirs(fp)


	scene = bpy.context.scene

	scene.cycles.samples = args.num_samples

	# Render Optimizations
	scene.render.use_persistent_data = True

	scene.render.image_settings.file_format = str(args.format)
	scene.render.image_settings.color_depth = str(args.color_depth)

	# Background
	scene.render.dither_intensity = 0.0
	scene.render.film_transparent = True

	# Create collection for objects not to render with background

	scene.render.resolution_x = args.width
	scene.render.resolution_y = args.height
	scene.render.resolution_percentage = 100

	if args.distance_rendering:
		scene.cycles.samples = 1
		scene.render.image_settings.file_format = 'PNG'
		scene.render.image_settings.color_depth = '16'
		scene.render.image_settings.color_mode = 'BW'
		scene.display_settings.display_device = 'sRGB'
		scene.view_settings.view_transform = 'Standard'
		scene.view_settings.exposure = 0
		scene.view_settings.gamma = 1
		scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0,0,0,1)

	# Projectors
	projectors = [obj for obj in bpy.data.objects if obj.type == 'LIGHT' and obj.data.type == 'SPOT']

	for obj in projectors:
		if args.flat_shading:
			light = obj.data
			light.cycles.cast_shadow = False
			light.use_shadow = False

		if args.proj_roll is not None:
			obj.rotation_euler.z = args.proj_roll
			bpy.context.view_layer.update()

		if args.randomize_projector_pose:
			pos_sigma = 0.1
			rot_sigma = 5.0

			obj.location.x += random.gauss(0.0, pos_sigma)
			obj.location.y += random.gauss(0.0, pos_sigma)
			obj.location.z += random.gauss(0.0, pos_sigma)

			random_scale = random.gauss(1.0, 1.0)
			if random.random() > 0.5:
				random_scale *= -1
			obj.location.x *= random_scale
			obj.location.y *= random_scale
			obj.location.z *= random_scale

			obj.rotation_euler.x += radians(random.gauss(0.0, rot_sigma))
			obj.rotation_euler.y += radians(random.gauss(0.0, rot_sigma))
			obj.rotation_euler.z += radians(random.gauss(0.0, rot_sigma))
			bpy.context.view_layer.update()

	# Intensity setting
	for obj in bpy.data.objects:
		for m in obj.material_slots:
			material = m.material
			if material is None:
				continue
			material.use_nodes = True
			tree = material.node_tree
			nodes = tree.nodes
			for node in nodes:
				if node.type == "EMISSION":
					if args.distance_rendering:
						node.inputs["Strength"].default_value = 0
					else:
						node.inputs["Strength"].default_value *= args.illum_intensity

	world = bpy.data.worlds['World']
	world.use_nodes = True
	bg = world.node_tree.nodes['Background']
	if args.distance_rendering:
		bg.inputs["Strength"].default_value = 0
	else:
		bg.inputs["Strength"].default_value *= args.illum_intensity

	# Scatter setting
	if args.scatter_density > 0:
		tree = world.node_tree
		links = tree.links

		if args.scatter_type == "hg":
			volume_node = tree.nodes.new('ShaderNodeVolumePrincipled')
		else:
			volume_node = tree.nodes.new('ShaderNodeVolumeScatter')
			volume_node.inputs['Anisotropy'].default_value = 0.05

		volume_node.inputs['Density'].default_value = args.scatter_density
		volume_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)

		links.new(volume_node.outputs["Volume"], world.node_tree.nodes["World Output"].inputs["Volume"])

		bpy.context.scene.cycles.volume_bounces = args.volume_bounce

	# Shading setting
	for obj in bpy.context.scene.objects:
		if obj.type == 'MESH':
			if 'Housing' in scene.objects and obj == scene.objects['Housing']:
				continue
			if args.flat_shading:
				if len(obj.data.materials) == 0:
					obj.data.materials.append(None)
				mat = bpy.data.materials.new(name="Diffuse_BSDF")
				mat.use_nodes = True
				nodes = mat.node_tree.nodes
				for node in nodes:
					nodes.remove(node)
				diffuse_node = nodes.new(type='ShaderNodeBsdfDiffuse')
				diffuse_node.location = (0, 0)
				output_node = nodes.new(type='ShaderNodeOutputMaterial')
				output_node.location = (200, 0)
				
				mat.node_tree.links.new(diffuse_node.outputs['BSDF'], output_node.inputs['Surface'])
				
				for i in range(len(obj.data.materials)):
					obj.data.materials[i] = mat

			elif args.distance_rendering:
				if len(obj.data.materials) == 0:
					obj.data.materials.append(None)

				mat = bpy.data.materials.new(name="Distance_Emission")
				mat.use_nodes = True
				nodes = mat.node_tree.nodes
				links = mat.node_tree.links

				for node in nodes:
					nodes.remove(node)

				output_node = nodes.new(type='ShaderNodeOutputMaterial')
				output_node.location = (400, 0)

				emission_node = nodes.new(type='ShaderNodeEmission')
				emission_node.location = (200, 0)
				emission_node.inputs['Strength'].default_value = 1.0

				light_path = nodes.new(type='ShaderNodeLightPath')
				light_path.location = (-400, 0)

				map_range = nodes.new(type='ShaderNodeMapRange')
				map_range.location = (-200, 0)
				map_range.inputs['From Min'].default_value = args.near
				map_range.inputs['From Max'].default_value = args.far
				map_range.inputs['To Min'].default_value = 1.0
				map_range.inputs['To Max'].default_value = 0.0

				links.new(light_path.outputs['Ray Length'], map_range.inputs[0])
				links.new(map_range.outputs[0], emission_node.inputs['Color'])
				links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

				for i in range(len(obj.data.materials)):
					obj.data.materials[i] = mat

			elif args.toon_shading:
				for obj in bpy.context.scene.objects:
					if obj.type != 'MESH':
						continue
					if 'Housing' in scene.objects and obj == scene.objects['Housing']:
						continue

					mat = bpy.data.materials.new(name="Toon_Material")
					mat.use_nodes = True
					nodes = mat.node_tree.nodes
					links = mat.node_tree.links

					for node in nodes:
						nodes.remove(node)

					toon_node = nodes.new(type='ShaderNodeBsdfToon')
					toon_node.location = (0, 0)
					toon_node.inputs['Size'].default_value = 0.5
					toon_node.inputs['Smooth'].default_value = 0.05

					output_node = nodes.new(type='ShaderNodeOutputMaterial')
					output_node.location = (200, 0)

					links.new(toon_node.outputs['BSDF'], output_node.inputs['Surface'])

					if len(obj.data.materials) == 0:
						obj.data.materials.append(mat)
					else:
						for i in range(len(obj.data.materials)):
							obj.data.materials[i] = mat

	# Blur setting
	if args.blur > 0:
		bpy.context.scene.cycles.motion_blur_position = 'CENTER'
		bpy.context.scene.render.motion_blur_shutter = args.blur
		bpy.context.scene.render.use_motion_blur = True

	pattern_objs = []
	if args.custom_pattern is not None:
		for pattern in args.custom_pattern:
			if args.randomize_pattern:
				img = Image.open(pattern)
				img = augment_image(img)
				tmp_path = os.path.join(os.path.dirname(__file__), "augmented_pattern.png")
				img.save(tmp_path)
				pattern_obj = bpy.data.images.load(tmp_path)

			else:
				pattern_obj = bpy.data.images.load(pattern)

			pattern_obj.save(filepath="test.png")	# This is necessary, but I don't know why
			pattern_objs.append(pattern_obj)

	for pi, proj in enumerate(projectors):
		if args.num_proj > pi and not args.distance_rendering:
			proj.data.energy *= args.proj_intensity
		else:
			proj.data.energy *= 0

		if len(pattern_objs) > 0:
			if len(pattern_objs) == 1:
				pattern_obj = pattern_objs[0]
			else:
				pattern_obj = pattern_objs[pi]

#			proj.data.use_nodes = True
			for node_group in proj.data.node_tree.nodes:
				for node in node_group.node_tree.nodes:
					if node.type != "TEX_IMAGE":
						continue
					node.image = pattern_obj

	tree = bpy.context.scene.node_tree
	links = tree.links

	if args.out_depth_normal:
		# Set up rendering of depth map.
		bpy.context.scene.use_nodes = True

		# Add passes for additionally dumping albedo and normals.
#		bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True

		if 'Custom Outputs' not in tree.nodes:
			# Create input render layer node.
			render_layers = tree.nodes.new('CompositorNodeRLayers')
			render_layers.label = 'Custom Outputs'
			render_layers.name = 'Custom Outputs'

			depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
			depth_file_output.label = 'Depth Output'
			depth_file_output.name = 'Depth Output'
			depth_file_output.format.file_format = 'PNG'
			depth_file_output.format.color_depth = '16'
			depth_file_output.format.color_mode = 'BW'

			if args.format == 'OPEN_EXR':
			  links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
			else:
			  # Remap as other types can not represent the full range of depth.
			  map = tree.nodes.new(type="CompositorNodeMapRange")
			  # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
			  map.inputs['From Min'].default_value = args.near
			  map.inputs['From Max'].default_value = args.far
			  map.inputs['To Min'].default_value = 1
			  map.inputs['To Max'].default_value = 0
			  links.new(render_layers.outputs['Depth'], map.inputs[0])

			  links.new(map.outputs[0], depth_file_output.inputs[0])

#			normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
#			normal_file_output.label = 'Normal Output'
#			normal_file_output.name = 'Normal Output'
#			links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

		if not args.debug:
			for output_node in [tree.nodes['Depth Output']]:
				if os.name == 'nt':  # Windows の場合
					output_node.base_path = ''
				else:  # Linux や macOS の場合
					output_node.base_path = '/'

	cam = scene.objects['Camera']
	if args.co_moving_light:
		light_data = bpy.data.lights.new(name="SpotLight", type='SPOT')
		spotlight_left = bpy.data.objects.new(name="SpotLight", object_data=light_data)
		spotlight_right = bpy.data.objects.new(name="SpotLight", object_data=light_data)
		spotlight_left.parent = cam
		spotlight_right.parent = cam
		spotlight_left.location = np.array(projectors[0].matrix_local)[:3,3]
		spotlight_right.location = np.array(projectors[-1].matrix_local)[:3,3]
		spotlight_left.data.energy = 30
		spotlight_right.data.energy = 30
		scene.collection.objects.link(spotlight_left)
		scene.collection.objects.link(spotlight_right)


	# Data to store in JSON file
	out_data = {
		'camera_angle_x': cam.data.angle_x,
		'camera_angle_y': cam.data.angle_y,
		'w': args.width,
		'h': args.height,
		'cx': args.width // 2,
		'cy': args.height // 2,
		'k1': 0,
		'k2': 0,
		'p1': 0,
		'p2': 0,
		'illum_intensity': args.illum_intensity,
		'proj_intensity': args.proj_intensity,
		'projectors': []
	}

	if 'Housing' in scene.objects:
		housing = scene.objects['Housing']

		if args.enable_refraction:
			d = (housing.scale[2] - housing.location[2]) / 2
			for m in housing.material_slots:
				material = m.material
				if material is None:
					continue
				material.use_nodes = True
				tree = material.node_tree
				nodes = tree.nodes
				for node in nodes:
					if node.type == "BSDF_REFRACTION":
						mu = 1 / node.inputs["IOR"].default_value
			out_data["refractive_interface"] = {"d": d, "mu": mu}
		else:
			housing.hide_render = True

	out_data['frames'] = []

	for pi, proj in enumerate(projectors[:args.num_proj]):
#		proj.data.use_nodes = True
		fov = None
		for node in proj.data.node_tree.nodes:
			fov = node.inputs['Degrees'].default_value
			break

		for node_group in proj.data.node_tree.nodes:
			for node in node_group.node_tree.nodes:
				if node.type != "TEX_IMAGE":
					continue
				path = os.path.join(os.path.dirname(__file__), args.results_path, "pattern_%d.png" % pi)
				node.image.filepath_raw = path
				node.image.file_format = 'PNG'
				node.image.save()

		out_data['projectors'].append({
			'proj_angle_x': radians(fov),
			'pattern': "pattern_%d.png" % pi,
			'transform_matrix': listify_matrix(proj.matrix_local),
		})

	frame_begin = bpy.context.scene.frame_start
	frame_end = bpy.context.scene.frame_end

	count = 0
	for i in range(args.begin, args.end if args.end is not None else args.views):
		fid = (frame_end - frame_begin) * i // args.views + frame_begin
		print(fid)
		bpy.context.scene.frame_set(fid)
		# Particle setting
		if not args.enable_particle:
			bpy.ops.object.select_all(action='DESELECT')
			for obj in bpy.data.objects:
				if obj.particle_systems:
					obj.select_set(True)
			bpy.ops.object.delete() 

		scene.render.filepath = fp + '/r_' + str(count)

		if args.out_depth_normal:
			bpy.context.scene.node_tree.nodes['Depth Output'].file_slots[0].path = scene.render.filepath + "_depth_"
#			bpy.context.scene.node_tree.nodes['Normal Output'].file_slots[0].path = scene.render.filepath + "_normal_"

		if args.debug:
			break
		else:
			if i >= args.begin and not (args.exist_skip and os.path.exists(scene.render.filepath + "." + args.format)):
				bpy.ops.render.render(write_still=True)  # render still

		print(scene.render.filepath)
		print(cam.matrix_world)
		frame_data = {
			'file_path': os.path.basename(scene.render.filepath),
			'depth_path': os.path.basename(scene.render.filepath + "_depth_%04d" % fid),
			'depth_near': args.near,
			'depth_far': args.far,
			'transform_matrix': listify_matrix(cam.matrix_world),
			'proj_transform_matrix': [listify_matrix(proj.matrix_local) for proj in projectors[:args.num_proj]],
		}
		out_data['frames'].append(frame_data)
		count += 1

	if not args.debug:
		with open(fp + '/' + 'transforms_train.json', 'w') as out_file:
			json.dump(out_data, out_file, indent=4)
		with open(fp + '/' + 'transforms_test.json', 'w') as out_file:
			json.dump(out_data, out_file, indent=4)
		with open(fp + '/' + 'transforms.json', 'w') as out_file:
			json.dump(out_data, out_file, indent=4)

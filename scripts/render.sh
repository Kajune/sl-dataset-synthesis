#!/bin/bash

choices=(
	"aruco"
	"aruco_sin"
	"colorful_grid"
	"colorful_lines"
	"colorful_stripes"
	"dotline"
	"hamming"
	"random_dot"
	"random_grid"
	"random_grid_sin"
	"random_lines_0"
	"random_shapes"
	"random_stripes"
	"sinusoidal"
)
count=${#choices[@]}


dataset_name=random

for i in $(seq 0 200); do
	rnd=$(( RANDOM % 10000 ))

	index=$(( rnd % count ))
	selected=${choices[$index]}

	echo "rnd=$rnd"
	echo "index=$index"
	echo "selected=$selected"

	blender --background --python generate_random_scene.py -- \
			--max_num_objects 100 \
			--camera_mode round \
			--num_frames 1000

	blender scenes/random_scene.blend --background --python render_anim.py -- \
		--results_path dataset/${dataset_name}/scene_$i \
		--views 20 --begin 0 --end 20 \
		--out_depth_normal \
		--num_proj 1 \
		--proj_roll 0 \
		--illum_intensity 0.0 0.1 \
		--proj_intensity 0.1 0.25 \
		--randomize_projector_pose \
		--randomize_pattern \
		--custom_pattern "./patterns/${selected}.png" \
		--blur 0.0 1.0
done

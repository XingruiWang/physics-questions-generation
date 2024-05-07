# python scripts/process_scene_files.py \
#     --data_dir /mnt/ccvl15/xingrui/output_v3_1k \
#     --output_train data/SuperCLEVR_physics_train_anno.json \
#     --output_val data/SuperCLEVR_physics_val_anno.json \
#     --train_scene_length 1000 \
#     --val_scene_length 100

# python scripts/generate_questions_video.py \
#     --input_scene_file data/SuperCLEVR_physics_val_anno.json \
#     --output_questions_file output/val/questions_physics_factual.json \
#     --template_dir templates/physics_factual \
#     --scene_start_idx 0 \
#     --num_scenes 100

MODE=val
python scripts/generate_questions_video.py \
    --input_scene_file data/SuperCLEVR_physics_${MODE}_anno.json \
    --output_questions_file output/${MODE}/questions_physics_predictive.json \
    --template_dir templates/physics_predictive \
    --scene_start_idx 0 \
    --num_scenes 1000 \
    --instances_per_template 3

python scripts/generate_questions_video.py \
    --input_scene_file data/SuperCLEVR_physics_${MODE}_anno.json \
    --output_questions_file output/${MODE}/questions_physics_factual.json \
    --template_dir templates/physics_factual \
    --scene_start_idx 0 \
    --num_scenes 1000 \
    --instances_per_template 1
    
#  templates/physics_factual_templates
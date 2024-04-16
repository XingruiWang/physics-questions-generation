"""
Process the generated scene annotation into the same format as the CLEVRER dataset
"""

import os
import json

def load_collisions(event_file):
    last_crash_frame = {}
    with open(event_file) as f:
        events = json.load(f)
    collisions = events["collisions"]
    filtered_collisions = []
    for c in collisions:
        instances = tuple(sorted(c['instances']))
        frame = int(c['frame'])+1
        if instances[0] >= 0:
            # if instances not in last_crash_frame:
            #     last_crash_frame[instances] = frame
            #     c['frame'] = frame
            #     filtered_collisions.append(c)
            if instances not in last_crash_frame or \
                frame - last_crash_frame[instances] > 5:
                last_crash_frame[instances] = frame
                c['frame'] = frame
                filtered_collisions.append(c)
    return filtered_collisions

def load_metadata(meta_file):
    with open(meta_file) as f:
        metadata = json.load(f)
    meta = metadata['metadata']
    camera = metadata["camera"]
    camera.pop("positions")
    camera.pop("quaternions")

    instances = metadata["instances"]

    # 1. Load objects 
    objects = []
    for instance in instances:
        # 'angular_velocities', 'asset_id', 'bboxes_3d', 'color', 'engine_on', 'floated', 'friction', 'id', 'image_positions', 'init_position', 'init_speed', 'mass', 'name', 'positions', 'quaternions', 'restitution', 'size', 'velocities', 'visibility'])

        obj = {
            "id": instance["id"],
            "name": instance["name"],
            "shape": instance["name"].split("_")[1],
            "color": instance["color"],
            "size": instance["size"],
            
            "mass": instance["mass"],
            "friction": instance["friction"],
            "restitution": instance["restitution"],
            
            "engine_on": instance["engine_on"],
            "floated": instance["floated"]
        }
        objects.append(obj)

    # 2. Load motions
    motion = {}
    frame_len = len(instances[0]["positions"])
    for instance in instances:
        for f in range(frame_len):
            if f not in motion:
                motion[f] = {}
                motion[f]['frame'] = f
                motion[f]['objects'] = []
            motion[f]['objects'].append({   
                "id": instance["id"],
                "positions": instance["positions"][f],
                "quaternions": instance["quaternions"][f],
                "_bboxes_3d": instance["bboxes_3d"][f],
                "_image_positions": instance["image_positions"][f],
                "velocities": instance["velocities"][f],
                "angular_velocities": instance["angular_velocities"][f],
                "visibility": instance["visibility"][f] > 100
            })
    motion = list(motion.values())
    

    for m in motion:
        visibility = [o['visibility'] for o in m['objects']]
        if not all(visibility):
            frame_len = m['frame']
            break

    motion = [m for m in motion if m['frame'] < frame_len]

    motion = sorted(motion, key=lambda x: x['frame'])

    return objects, motion, frame_len

def run_scene(scene_id):
    # scene_id = "super_clever_1098"
    event_file = f"/home/xingrui/physics_questions_generation/data/output/{scene_id}/events.json"
    meta_file = f"/home/xingrui/physics_questions_generation/data/output/{scene_id}/metadata.json"

    collisions = load_collisions(event_file)
    objects, motions, frame_len = load_metadata(meta_file)

    scene = {}
    scene['index'] = scene_id
    scene['frame_len'] = frame_len
    scene['objects'] = objects
    scene['motions'] = motions
    scene['collisions'] = collisions
    return scene

if __name__ == "__main__":
    from tqdm import tqdm

    all_scenes = {}
    all_scenes['info'] = {
        "name": "SuperCLEVR_physics",
        "version": "1.0",
        "date": "2024-04-15",
        "description": "SuperCLEVR_physics dataset",
        "contributor": "Xingrui Wang",
        "split": "train"
    }
    all_scenes['scenes'] = []
    for i in tqdm(range(1000)):
        scene_id = f"super_clever_{i}"
        scene = run_scene(scene_id)
        all_scenes['scenes'].append(scene)

    with open("data/SuperCLEVR_physics_train_anno.json", "w") as f:
        json.dump(all_scenes, f)
    print("Done")


    all_scenes = {}
    all_scenes['info'] = {
        "name": "SuperCLEVR_physics",
        "version": "1.0",
        "date": "2024-04-15",
        "description": "SuperCLEVR_physics dataset",
        "contributor": "Xingrui Wang",
        "split": "validation"
    }
    all_scenes['scenes'] = []
    for i in tqdm(range(1000,1100)):
        scene_id = f"super_clever_{i}"
        scene = run_scene(scene_id)
        all_scenes['scenes'].append(scene)
    with open("data/SuperCLEVR_physics_val_anno.json", "w") as f:
        json.dump(all_scenes, f)


#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # parent of scripts/
PY = sys.executable


mipnerf360_outdoor_scenes = ["treehill", "garden","stump","bicycle", "flowers"]
mipnerf360_indoor_scenes = ["counter", "room", "kitchen", "bonsai"]
# tanks_and_temples_scenes = ["truck", "train"]
# deep_blending_scenes = ["drjohnson", "playroom"] 

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="eval/mipnerf360")
parser.add_argument("--clean_pc", action="store_true", help="Apply hull removal filtering to point clouds")  

args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
# all_scenes.extend(tanks_and_temples_scenes)
# all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
    # parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    # parser.add_argument("--deepblending", "-db", required=True, type=str)
    args = parser.parse_args()

if not args.skip_training:
    seg_args = " --dataset_type nerf --run_segmentation --lambda_normal 0.00 --lambda_dist 0.00 --lambda_segment 0.00"
    common_args = " --quiet --eval --test_iterations -1" + seg_args

    if args.clean_pc:
        seg_args += " --clean"

    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system(f"{PY} {REPO_ROOT/'train.py'} -s {source} -i images -m {args.output_path}/{scene}{common_args}")

    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system(f"{PY} {REPO_ROOT/'train.py'} -s {source} -i images -m {args.output_path}/{scene}{common_args}")
        # source = args.tanksandtemples + "/" + scene
        # os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    # for scene in deep_blending_scenes:
        # source = args.deepblending + "/" + scene
        # os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
       all_sources.append(args.mipnerf360 + "/" + scene)
    # for scene in tanks_and_temples_scenes:
        # all_sources.append(args.tanksandtemples + "/" + scene)
    # for scene in deep_blending_scenes:
        # all_sources.append(args.deepblending + "/" + scene)

    common_args = " --quiet --eval --skip_train"
    for scene, source in zip(all_scenes, all_sources):
        os.system(f"{PY} {REPO_ROOT/'render.py'} --iteration 30000 -s {source} -m {args.output_path}/{scene}{common_args}")


if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "
    
    os.system(f"{PY} {REPO_ROOT/'metrics.py'} -m {scenes_string}")

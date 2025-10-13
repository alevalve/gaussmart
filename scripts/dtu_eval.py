#!/usr/bin/env python3
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import os
import sys
from pathlib import Path
from argparse import ArgumentParser

REPO_ROOT = Path(__file__).resolve().parents[1]  # parent of scripts/
PY = sys.executable

# DTU 15 scenes
dtu_scenes = [
    "scan24", "scan37", "scan40", "scan55", "scan63",
    "scan65", "scan69", "scan83", "scan97", "scan105",
    "scan106", "scan110", "scan114", "scan118", "scan122"
]

parser = ArgumentParser(description="Full evaluation script parameters (DTU)")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="eval/dtu")
parser.add_argument("--clean_pc", action="store_true", help="Apply hull removal filtering to point clouds")

args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(dtu_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument("--dtu", "-dtu", required=True, type=str)
    args = parser.parse_args()
else:
    args = parser.parse_args()

if not args.skip_training:
    seg_args = " --dataset_type dtu --run_segmentation --lambda_normal 0.00 --lambda_dist 0.00 --lambda_segment 0.00"
    if args.clean_pc:
        seg_args += " --clean"

    common_args = " --quiet --eval --test_iterations -1" + seg_args

    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        os.system(f"{PY} {REPO_ROOT/'train.py'} -s {source} -i images -m {args.output_path}/{scene}{common_args}")

if not args.skip_rendering:
    all_sources = []
    for scene in dtu_scenes:
        all_sources.append(args.dtu + "/" + scene)

    common_args = " --quiet --eval --skip_train"
    for scene, source in zip(all_scenes, all_sources):
        os.system(f"{PY} {REPO_ROOT/'render.py'} --iteration 30000 -s {source} -m {args.output_path}/{scene}{common_args}")

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system(f"{PY} {REPO_ROOT/'metrics.py'} -m {scenes_string}")

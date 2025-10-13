import os
from argparse import ArgumentParser
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # parent of scripts/
PY = sys.executable

# Tanks and Temples scenes (lowercase to match typical folder names)
tnt_360_scenes = ["barn", "caterpillar", "ignatius", "truck"]
tnt_large_scenes = ["meetingroom", "courthouse"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="eval/tnt")
parser.add_argument("--clean_pc", action="store_true",
                    help="Apply hull removal filtering to point clouds")

# NEW: pass-through DINO args
parser.add_argument("--dino_start_iter", type=int, default=3000,
                    help="Iteration after which DINO loss is enabled")
parser.add_argument("--lambda_dino", type=float, default=0.05,
                    help="Weight for DINO loss once enabled")

args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(tnt_360_scenes)
all_scenes.extend(tnt_large_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--tnt', "-tnt", required=True, type=str)  # dataset root
    args = parser.parse_args()

if not args.skip_training:
    # Base segmentation arguments
    seg_args = " --dataset_type tyt --run_segmentation --lambda_normal 0.00 --lambda_dist 0.00 --lambda_segment 0.00"
    
    # Add clean flag if specified
    if args.clean_pc:
        seg_args += " --clean"
    
    # Add dino args
    dino_args = f" --dino_start_iter {args.dino_start_iter} --lambda_dino {args.lambda_dino}"
    
    common_args = " --quiet --eval --test_iterations -1" + seg_args + dino_args

    for scene in tnt_360_scenes:
        source = args.tnt + "/" + scene
        os.system(f"{PY} {REPO_ROOT/'train.py'} -s {source} -i images -m {args.output_path}/{scene}{common_args}")

    for scene in tnt_large_scenes:
        source = args.tnt + "/" + scene
        os.system(f"{PY} {REPO_ROOT/'train.py'} -s {source} -i images -m {args.output_path}/{scene}{common_args}")

if not args.skip_rendering:
    all_sources = []
    for scene in tnt_360_scenes:
        all_sources.append(args.tnt + "/" + scene)
    for scene in tnt_large_scenes:
        all_sources.append(args.tnt + "/" + scene)

    common_args = " --quiet --eval --skip_train"
    for scene, source in zip(all_scenes, all_sources):
        os.system(f"{PY} {REPO_ROOT/'render.py'} --iteration 30000 -s {source} -m {args.output_path}/{scene}{common_args}")

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "
    os.system(f"{PY} {REPO_ROOT/'metrics.py'} -m {scenes_string}")

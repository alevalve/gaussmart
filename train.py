# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import os
import csv
import subprocess 
import json
import sys
import torch
import torchvision
import torch.nn.functional as F
from pathlib import Path
from random import randint
from utils.loss_utils import l1_loss, ssim, dino_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.embeds_utils import tensor_to_pil, load_gt_embeddings
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from identification.extraction.feature_extraction import DINOImageEncoder
from arguments import ModelParams, PipelineParams, OptimizationParams

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Import LPIPS
from lpipsPyTorch.modules.lpips import LPIPS

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe,
             testing_iterations,
             saving_iterations,
             checkpoint_iterations,
             checkpoint,
             render_indices=None,
             render_every=1000,
             use_dino_loss=True,
             lambda_dino=0.1,
             gt_embeddings_path=None):
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    dino_encoder = None
    gt_embeddings = {}
    
    if use_dino_loss:
        dino_encoder = DINOImageEncoder(batch_size=1)
        
        if gt_embeddings_path is None:
            raise ValueError("gt_embeddings_path must be provided when use_dino_loss=True")
        
        gt_embeddings = load_gt_embeddings(gt_embeddings_path, scene.getTrainCameras())

    if render_indices:
        with open(render_indices, 'r') as f:
             reader = csv.DictReader(f)
             selected_indices = []
             for row in reader:
                stem = row['stem']
                index = int(stem.split('_')[-1])
                selected_indices.append(index)
                 
        # Get the sorted image_name list (as 2DGS does internally)
        train_cams = scene.getTrainCameras()
        sorted_train_cams = sorted(train_cams, key=lambda c: Path(c.image_name).stem)

        # Map selected segmentation indices to the actual Camera objects
        render_list = [train_cams.index(sorted_train_cams[i]) for i in selected_indices]

    # Initialize LPIPS loss
    lpips_loss = LPIPS(net_type='alex', version='0.1').cuda()
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_dino_for_log = 0.0  # Add DINO loss tracking

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        dino_loss = dino_loss(
            image=image,
            viewpoint_cam=viewpoint_cam,
            scene=scene,
            gt_embeddings=gt_embeddings,
            dino_encoder=dino_encoder,
            lambda_dino=lambda_dino,
            iteration=iteration,
            render_every=render_every,
            use_dino_loss=use_dino_loss
        )

        # Regular regularization terms
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # Include DINO loss in total loss
        total_loss = loss + dist_loss + normal_loss + dino_loss
        
        total_loss.backward()

        iter_end.record()

        if render_indices and (iteration % render_every == 0):
            try:
                torch.cuda.empty_cache()
                with open(render_indices, 'r') as f:
                    reader = csv.DictReader(f)
                    selected_inds = []
                    for row in reader:
                        stem = row['stem']
                        index = int(stem.split('_')[-1])
                        selected_inds.append(index)
                
                if not selected_inds:
                    print(f"[Iter {iteration}] No valid indices found")
                    continue

                train_cams = scene.getTrainCameras()
                sorted_train_cams = sorted(train_cams, key=lambda c: Path(c.image_name).stem)
                
                out_dir = os.path.join(dataset.model_path, "renders", f"iter_{iteration}")
                os.makedirs(out_dir, exist_ok=True)
                
                successful_renders = 0
                for idx in selected_inds:
                    try:
                        if 0 <= idx < len(sorted_train_cams):
                            torch.cuda.empty_cache()
                            cam = sorted_train_cams[idx]
                            original_resolution = cam.image_width, cam.image_height
                            cam.image_width = cam.image_width // 2
                            cam.image_height = cam.image_height // 2
                            with torch.no_grad():
                                render_pkg = render(cam, gaussians, pipe, background)
                                image = render_pkg['render'].clamp(0, 1)
                            cam.image_width, cam.image_height = original_resolution
                            torchvision.utils.save_image(
                                image, 
                                os.path.join(out_dir, f"view_{idx:03d}.png")
                            )
                            successful_renders += 1
                    except Exception as e:
                        print(f"[Error] Failed to render view {idx}: {str(e)}")
                        continue

                print(f"[Iter {iteration}] Rendered {successful_renders}/{len(selected_inds)} views to {out_dir}")
            except Exception as e:
                print(f"[Error] During rendering block: {str(e)}")

        with torch.no_grad():
            # Progress bar - now including DINO loss
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_dino_for_log = 0.4 * dino_loss.item() + 0.6 * ema_dino_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "dino": f"{ema_dino_for_log:.{5}f}",  
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/dino_loss', ema_dino_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), lpips_loss)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # GUI update
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, lpips_loss):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                          {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0  # Add LPIPS evaluation
                
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).double()
                    
                    # Calculate LPIPS for evaluation
                    image_norm = image * 2.0 - 1.0
                    gt_image_norm = gt_image * 2.0 - 1.0
                    lpips_test += lpips_loss(image_norm, gt_image_norm).double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000,30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--render_indices", type=str, default=None, help="Path to JSON file listing train‐camera indices to render")
    parser.add_argument("--render_every", type=int, default=1000, help="How often (in iterations) to quick-render those views")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--run_segmentation", action="store_true")
    parser.add_argument("--segmentation_output", type=str, default="segmentation_results")
    parser.add_argument("--dataset_type", type=str, choices=['dtu', 'nerf', 'tyt'], default='tyt')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    
    if args.run_segmentation:
        print("\nRunning segmentation process...")
        try:
            seg_output_path = os.path.join("identification", "results")
            os.makedirs(seg_output_path, exist_ok=True)

            # Run segmentation
            subprocess.run([
                sys.executable,
                "-m", "identification.main",
                "-s", args.source_path,
                "-o", seg_output_path,
                "-t", args.dataset_type
            ], check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
            print("Segmentation completed successfully!")

        except subprocess.CalledProcessError as e:
            print(f"Segmentation failed with error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)

    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.render_indices, args.render_every)

    # All done
    print("\nTraining complete.")

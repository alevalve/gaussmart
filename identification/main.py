import os
import sys
import json
import argparse
import shutil
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from identification.extraction.image_encoder import process_images

from identification.process_selected_views import ProcessSelectedViews
from identification.sam import SAMSegmentation
from identification.pc_projection import (
    load_point_cloud, project_points_to_view, assign_segment_indices_simple
)



def setup_directories(output_base_path: str) -> Dict[str, str]:
    """Create directory structure for output files and return paths."""
    base_dir = os.path.join(output_base_path, 'segments')
    dirs = {
        'base': base_dir,
        'boxes': os.path.join(base_dir, 'boxes'),
        'masks': os.path.join(base_dir, 'masks'),
        'cameras': os.path.join(base_dir, 'cameras'),
        'segmented_images': os.path.join(base_dir, 'segmented_images'),
        'point_cloud': os.path.join(base_dir, 'point_cloud'),
        'images': os.path.join(base_dir, 'images'),
        'embeddings':os.path.join(base_dir, 'embeddings')
    }

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return dirs



def extract_embeddings(image_path, out_dir, batch_size):
    process_images(
        images_dir=image_path,
        out_dir=out_dir,
        batch_size=batch_size,
    )



def get_point_cloud_path(scan_path: str, dataset_type: str) -> str:
    """Return the path to the raw point cloud file based on dataset type."""
    if dataset_type.lower() == 'dtu':
        return os.path.join(scan_path, "points.ply")
    return os.path.join(scan_path, "sparse/0/points3D.ply")


def get_camera_path(scan_path: str, dataset_type: str) -> str:
    """Return the path to the camera/pose file based on dataset type."""
    return os.path.join(
        scan_path,
        'cameras.npz' if dataset_type.lower() == 'dtu' else 'poses_bounds.npy'
    )



def load_raw_point_cloud(scan_path: str, dataset_type: str) -> Optional[o3d.geometry.PointCloud]:
    """Load raw point cloud if present; otherwise return None."""
    pc_path = get_point_cloud_path(scan_path, dataset_type)
    if not os.path.exists(pc_path):
        print(f"Warning: Point cloud not found at {pc_path}")
        return None
    
    return o3d.io.read_point_cloud(pc_path)


def persist_raw_point_cloud(pcd: o3d.geometry.PointCloud, output_dir: str, name: str = "raw_pc.ply") -> None:
    """Save a point cloud to disk."""
    out_path = os.path.join(output_dir, name)
    o3d.io.write_point_cloud(out_path, pcd)



def run_sam_on_image(
    segmenter: SAMSegmentation,
    image_path: str,
    out_masks_path: str,
    out_vis_dir: str,
    tag: str
) -> List[Dict]:
    """
    Run SAM segmentation for a single image and save both raw masks and a visualization.
    Returns the masks list.
    """
    masks = segmenter.process_image(image_path)
    segmenter.save_segments_boxes(masks, out_masks_path)
    segmenter.visualize_masks(image_path, masks, tag, out_vis_dir)
    return masks


def process_and_copy_image(src_image_path: str, images_dir: str, index: int) -> str:
    """Copy source image to the output images folder with a consistent name."""
    dst = os.path.join(images_dir, f"{index:03d}_{os.path.basename(src_image_path)}")
    shutil.copy2(src_image_path, dst)
    return dst


def preselect_views(
    scan_path: str,
    output_base_path: str,
    dataset_type: str
) -> Tuple[ProcessSelectedViews, List[int], Dict]:
    """
    Build the view processor, select indices, and return:
      - processor
      - selected_indices
      - selected_data (image paths + camera parameters for those indices)
    """
    camera_path = get_camera_path(scan_path, dataset_type)
    image_root = os.path.join(scan_path, 'images')

    processor = ProcessSelectedViews(camera_path, image_root, output_base_path, dataset_type)
    sel_info = processor.process_views()
    selected_indices = sel_info['selected_indices']

    selected_data = processor.get_selected_data(selected_indices, already_mapped=True)
    return processor, selected_indices, selected_data


def build_camera_dict(processor: ProcessSelectedViews, selected_indices: List[int]) -> Dict[str, Dict]:
    """
    Build a dict of camera parameters keyed by 'camera_###' matching selected_indices order.
    """
    return {
        f'camera_{i:03d}': processor.analyzer.views[idx]
        for i, idx in enumerate(selected_indices)
    }


def save_selected_cameras(
    dirs: Dict[str, str],
    selected_indices: List[int],
    processor: ProcessSelectedViews
) -> None:
    """Persist selected camera metadata (dict and list) to disk."""
    cameras_dict = build_camera_dict(processor, selected_indices)
    camera_data = {
        'selected_indices': selected_indices,
        'cameras_dict': cameras_dict,
        'cameras_list': [processor.analyzer.views[idx] for idx in selected_indices],
    }
    np.savez(os.path.join(dirs['cameras'], 'selected_cameras.npz'), **camera_data)


def compute_mask_areas_for_view(segmentation_masks: List[np.ndarray]) -> Dict[int, int]:
    """Return a dict {mask_idx: area_pixels} for a single view."""
    areas = {}
    for mask_idx, mask in enumerate(segmentation_masks):
        areas[mask_idx] = int(np.sum(mask > 0))
    return areas


def merge_max_areas(global_areas: Dict[int, int], view_areas: Dict[int, int]) -> None:
    """Update global mask areas by taking the max across views."""
    for k, v in view_areas.items():
        if k not in global_areas:
            global_areas[k] = v
        else:
            global_areas[k] = max(global_areas[k], v)


def project_and_assign_for_view(
    points: np.ndarray,
    camera: Dict,
    dataset_type: str,
    segmentation_masks: List[np.ndarray],
    current_assignments: np.ndarray
) -> None:
    """
    Project 3D points into a single view and assign segment IDs for those
    still unassigned (-1) and visible in this view. Updates current_assignments in place.
    """
    h, w = segmentation_masks[0].shape
    points_2d, depths = project_points_to_view(points, camera, dataset_type)

    in_bounds = np.all((points_2d >= [0, 0]) & (points_2d < [w, h]), axis=1)
    front_of_camera = depths > 0
    visible = in_bounds & front_of_camera

    to_process = visible & (current_assignments == -1)
    if not np.any(to_process):
        return

    points_2d_clipped = np.clip(points_2d[to_process], [0, 0], [w - 1, h - 1])
    view_segments = assign_segment_indices_simple(points_2d_clipped, segmentation_masks)

    # Update only newly assigned points
    newly_assigned = view_segments != -1
    current_assignments[to_process] = np.where(newly_assigned, view_segments, -1)


def process_segments_with_areas(
    points: np.ndarray,
    all_masks: List[List[Dict]],
    cameras: Dict[str, Dict],
    dataset_type: str
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Process segments across all selected views and calculate per-segment max mask areas.
    Returns:
      - segment_indices: (N,) int array with segment IDs for each 3D point (or -1 if none)
      - mask_areas: dict {segment_id: max_pixel_area_across_views}
    """
    segment_indices = -np.ones(len(points), dtype=int)
    mask_areas: Dict[int, int] = {}

    for view_idx, masks_list in enumerate(all_masks):
        if not masks_list:
            continue

        camera = cameras[f'camera_{view_idx:03d}']
        segmentation_masks = [m['segmentation'] for m in masks_list]

        # areas for this view
        view_areas = compute_mask_areas_for_view(segmentation_masks)
        merge_max_areas(mask_areas, view_areas)

        # assign indices for visible/unassigned points
        project_and_assign_for_view(points, camera, dataset_type, segmentation_masks, segment_indices)

    return segment_indices, mask_areas


def save_results_with_areas(
    pcd: o3d.geometry.PointCloud,
    segment_indices: np.ndarray,
    mask_areas: Dict[int, int],
    output_dir: str
) -> None:
    """
    Save the point cloud with original colors, segment indices, and mask areas.
    """
    # Create output point cloud preserving original geometry and colors
    output_pcd = o3d.geometry.PointCloud()
    output_pcd.points = pcd.points
    if pcd.has_normals():
        output_pcd.normals = pcd.normals
    if pcd.has_colors():
        output_pcd.colors = pcd.colors

    pc_path = os.path.join(output_dir, "segmented_point_cloud.ply")
    indices_path = os.path.join(output_dir, "segment_indices.npy")
    areas_path = os.path.join(output_dir, "mask_areas.npy")

    o3d.io.write_point_cloud(pc_path, output_pcd)
    np.save(indices_path, segment_indices)
    np.save(areas_path, mask_areas)


# ------------------------------- ORCHESTRATOR ------------------------------- #

def run_segmentation(
    scan_path: str,
    output_base_path: str,
    dataset_type: str
) -> Tuple[Optional[np.ndarray], Optional[Dict[int, int]]]:
    """Main segmentation pipeline that computes segment indices and per-segment areas."""
    # Create output structure
    dirs = setup_directories(output_base_path)

    # View selection
    processor, selected_indices, selected_data = preselect_views(scan_path, output_base_path, dataset_type)
    save_selected_cameras(dirs, selected_indices, processor)

    # Segmenter
    weights_path = os.path.join(Path(__file__).resolve().parent, 'weights', 'sam_vit_h_4b8939.pth')
    segmenter = SAMSegmentation(weights_path)

    # Run SAM per selected image
    results = []
    for i, image_path in enumerate(selected_data['image_paths']):
        copied = process_and_copy_image(image_path, dirs['images'], i)

        seg_path = os.path.join(dirs['masks'], f"segments_{i:03d}.npz")
        masks = run_sam_on_image(
            segmenter=segmenter,
            image_path=image_path,
            out_masks_path=seg_path,
            out_vis_dir=dirs['segmented_images'],
            tag=f"{i:03d}"
        )

        results.append({
            "masks": masks,
            "camera_data": selected_data['camera_parameters'][f"camera_{i:03d}"]
        })
    
    # Called DinoV3
    extract_embeddings(dirs['images'],dirs['embeddings'],4)

    # Point cloud
    raw_pcd = load_raw_point_cloud(scan_path, dataset_type)
    if raw_pcd is None:
        return None, None

    persist_raw_point_cloud(raw_pcd, dirs['point_cloud'], "raw_pc.ply")
    points = np.asarray(raw_pcd.points)

    # Compute segment indices + areas
    cameras_dict = build_camera_dict(processor, selected_indices)
    segment_indices, mask_areas = process_segments_with_areas(
        points=points,
        all_masks=[r["masks"] for r in results],
        cameras=cameras_dict,
        dataset_type=dataset_type
    )

    # Save
    save_results_with_areas(raw_pcd, segment_indices, mask_areas, dirs['point_cloud'])
    return segment_indices, mask_areas



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process camera views and generate segmentations')
    parser.add_argument('-s', '--scan_path', required=True, help='Path to scan folder')
    parser.add_argument('-o', '--output_base_path', required=True, help='Output directory')
    parser.add_argument('-t', '--type', choices=['dtu', 'nerf', 'tyt'], required=True, help='Dataset type')
    return parser.parse_args()


def main():
    args = parse_args()
    run_segmentation(args.scan_path, args.output_base_path, args.type)


if __name__ == "__main__":
    main()
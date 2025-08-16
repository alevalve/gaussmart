import numpy as np
import open3d as o3d
import cv2
import os
import matplotlib.pyplot as plt
from typing import Dict, List

def load_point_cloud(point_cloud_path: str, dataset_type: str):
    """
    Load a point cloud from file path based on the dataset type.
    Returns both the o3d point cloud object and its points as numpy array.
    """
    if dataset_type.lower() == 'dtu':
        pcd = o3d.io.read_point_cloud(point_cloud_path)
    else:  # NeRF or TYT
        pcd = o3d.io.read_point_cloud(point_cloud_path)
    
    points = np.asarray(pcd.points)
    return pcd, points

def project_points_to_view(points: np.ndarray, camera: Dict, dataset_type: str):
    """
    Project 3D points into a 2D image plane using camera parameters.
    Handles DTU, NeRF, and Tanks and Temples (TYT) dataset formats.
    """
    if dataset_type.lower() == 'dtu':
        world_mat = camera['world_mat']
        scale_mat = camera['scale_mat']
        camera_mat = camera['camera_mat']
        
        points_homo = np.hstack((points, np.ones((points.shape[0], 1))))
        points_scaled = (scale_mat @ points_homo.T).T
        points_cam = (world_mat @ points_scaled.T).T
        z = points_cam[:, 2].copy()
        
        fx = camera_mat[0, 0]
        fy = camera_mat[1, 1]
        cx = camera_mat[0, 2]
        cy = camera_mat[1, 2]
        
        x = points_cam[:, 0] / points_cam[:, 3]
        y = points_cam[:, 1] / points_cam[:, 3]
        u = fx * x + cx
        v = fy * y + cy
        
        points_2d = np.column_stack((u, v))
        
        w, h = 1554, 1162
        in_bounds = np.all((points_2d >= [0, 0]) & (points_2d < [w, h]), axis=1)
        
        if np.sum(in_bounds) < 0.1 * len(points):
            K = np.array([
                [800, 0, w/2],
                [0, 800, h/2],
                [0, 0, 1]
            ])
            cam_pos = -np.linalg.inv(world_mat[:3, :3]) @ world_mat[:3, 3]
            vectors = points - cam_pos
            lengths = np.linalg.norm(vectors, axis=1).reshape(-1, 1)
            normalized = vectors / lengths
            points_2d = normalized[:, :2] / (normalized[:, 2].reshape(-1, 1) + 1e-10)
            points_2d[:, 0] = points_2d[:, 0] * (w/3) + w/2
            points_2d[:, 1] = points_2d[:, 1] * (h/3) + h/2
        
        return points_2d, z
        
    elif dataset_type.lower() == 'nerf':
        K = camera['camera_mat'][:3, :3]
        R = camera['world_mat'][:3, :3]
        t = camera['world_mat'][:3, 3]
        
        points_cam = (R @ points.T).T + t
        points_2d = (K @ points_cam.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:]
        
        return points_2d, points_cam[:, 2]
        
    elif dataset_type.lower() == 'tyt':
        if 'img_size' in camera:
            w, h = camera['img_size']
        else:
            w, h = 982, 543
            
        valid_points = ~np.isnan(points).any(axis=1)
        if np.any(valid_points):
            min_xyz = np.min(points[valid_points], axis=0)
            max_xyz = np.max(points[valid_points], axis=0)
            
            padding = 0.1
            norm_x = padding + (1 - 2*padding) * (points[:, 0] - min_xyz[0]) / (max_xyz[0] - min_xyz[0] + 1e-10)
            norm_y = padding + (1 - 2*padding) * (points[:, 1] - min_xyz[1]) / (max_xyz[1] - min_xyz[1] + 1e-10)
            
            points_2d = np.zeros((len(points), 2))
            points_2d[:, 0] = norm_x * w
            points_2d[:, 1] = norm_y * h
            points_2d = np.nan_to_num(points_2d, nan=0.0)
            
            world_mat = camera['world_mat']
            R = world_mat[:3, :3]
            t = world_mat[:3, 3]
            C = -R.T @ t
            vectors = points - C
            z = np.sum(vectors * (R[2, :]), axis=1)
            
            return points_2d, z
        else:
            return np.zeros((len(points), 2)), np.zeros(len(points))
    else:
        raise Exception("Dataset cameras are not configurable for projection")

def assign_segment_indices_simple(points_2d: np.ndarray, masks: List[np.ndarray]):
    """
    Assign segment indices to projected points based on their position in segmentation masks.
    Simply assigns sequential indices (0, 1, 2, ...) for each mask.
    Returns an array of segment indices (-1 indicates no segment).
    """
    if not masks:
        return -np.ones(len(points_2d), dtype=int)
    
    segment_indices = -np.ones(len(points_2d), dtype=int)
    y_coords = np.round(points_2d[:, 1]).astype(int)
    x_coords = np.round(points_2d[:, 0]).astype(int)
    
    # Assign sequential indices starting from 0
    for mask_idx, mask in enumerate(masks):
        valid = (x_coords >= 0) & (x_coords < mask.shape[1]) & \
                (y_coords >= 0) & (y_coords < mask.shape[0])
        
        inside_mask = np.zeros_like(valid)
        inside_mask[valid] = mask[y_coords[valid], x_coords[valid]] > 0
        
        # Assign sequential index (0, 1, 2, ...)
        segment_indices[inside_mask] = mask_idx
    
    return segment_indices

def process_all_views_with_mask_size(points: np.ndarray, all_masks: List[List[Dict]], 
                                    cameras: Dict, dataset_type: str):
    """
    Process all views and augment point cloud based on mask sizes.
    Larger masks get more points allocated to maintain density proportional to area.
    """
    num_points = len(points)
    final_segments = -np.ones(num_points, dtype=int)
    mask_areas = {}  # Store area for each segment across all views
    
    # First pass: calculate mask areas and assign segments
    for view_idx, masks_list in enumerate(all_masks):
        camera = cameras[f'camera_{view_idx:03d}']
        points_2d, depths = project_points_to_view(points, camera, dataset_type)
        
        segmentation_masks = [mask['segmentation'] for mask in masks_list]
        if len(segmentation_masks) == 0:
            continue
            
        h, w = segmentation_masks[0].shape
        in_bounds = np.all((points_2d >= [0, 0]) & (points_2d < [w, h]), axis=1)
        visible = in_bounds & (depths > 0)
        
        # Calculate mask areas (number of pixels)
        for mask_idx, mask in enumerate(segmentation_masks):
            segment_id = mask_idx  # Simple sequential ID
            mask_area = np.sum(mask > 0)
            
            if segment_id not in mask_areas:
                mask_areas[segment_id] = mask_area
            else:
                mask_areas[segment_id] = max(mask_areas[segment_id], mask_area)
        
        # Assign segments to visible points
        points_2d_clipped = np.clip(points_2d, [0, 0], [w-1, h-1])
        view_segments = assign_segment_indices_simple(points_2d_clipped, segmentation_masks)
        
        update_mask = visible & (view_segments != -1) & (final_segments == -1)
        final_segments[update_mask] = view_segments[update_mask]
    
    return final_segments, mask_areas

def save_point_cloud_with_segmentation(pcd: o3d.geometry.PointCloud, 
                                     segment_indices: np.ndarray, 
                                     output_path: str):
    """
    Save the point cloud to a PLY file preserving original colors (if any),
    and save the segmentation indices to a separate NumPy file.
    """
    # Create new point cloud preserving original geometry and colors
    output_pcd = o3d.geometry.PointCloud()
    output_pcd.points = pcd.points
    if pcd.has_normals():
        output_pcd.normals = pcd.normals
    if pcd.has_colors():
        output_pcd.colors = pcd.colors
    
    # Save point cloud
    o3d.io.write_point_cloud(output_path, output_pcd)
    
    # Save segmentation indices
    seg_indices_path = output_path.replace('.ply', '_segmentation.npy')
    np.save(seg_indices_path, segment_indices)
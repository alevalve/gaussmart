import numpy as np
import open3d as o3d
import os
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

def assign_segment_indices(points_2d: np.ndarray, masks: List[np.ndarray]):
    """
    Assign segment indices to projected points based on their position in segmentation masks.
    Returns an array of segment indices (-1 indicates no segment).
    """
    if not masks:
        return -np.ones(len(points_2d), dtype=int)
    
    segment_indices = -np.ones(len(points_2d), dtype=int)
    y_coords = np.round(points_2d[:, 1]).astype(int)
    x_coords = np.round(points_2d[:, 0]).astype(int)
    
    for idx, mask in enumerate(masks):
        valid = (x_coords >= 0) & (x_coords < mask.shape[1]) & \
                (y_coords >= 0) & (y_coords < mask.shape[0])
        inside_mask = np.zeros_like(valid)
        inside_mask[valid] = mask[y_coords[valid], x_coords[valid]] > 0
        segment_indices[inside_mask] = idx
    
    return segment_indices

def aggregate_segment_indices(points: np.ndarray, all_masks: List[List[Dict]], cameras: Dict, dataset_type: str):
    """
    Aggregate segment indices across multiple views with improved consistency.
    Uses segment overlap and IOU to match segments between views.
    """
    num_points = len(points)
    
    view_assignments = []
    view_point_visibility = []
    
    max_segments = 0
    
    for view_idx, masks_list in enumerate(all_masks):
        camera = cameras[f'camera_{view_idx:03d}']
        
        points_2d, depths = project_points_to_view(points, camera, dataset_type)
        
        segmentation_masks = [mask['segmentation'] for mask in masks_list]
        if len(segmentation_masks) == 0:
            continue
        
        h, w = segmentation_masks[0].shape
        
        in_bounds = np.all((points_2d >= [0, 0]) & (points_2d < [w, h]), axis=1)
        front_of_camera = depths > 0
        visible = in_bounds & front_of_camera
        
        points_2d_clipped = points_2d.copy()
        points_2d_clipped[visible] = np.clip(points_2d_clipped[visible], [0, 0], [w-1, h-1])
        
        view_segment_indices = assign_segment_indices(points_2d_clipped, segmentation_masks)
        
        if len(segmentation_masks) > max_segments:
            max_segments = len(segmentation_masks)
        
        view_assignments.append(view_segment_indices)
        view_point_visibility.append(visible & (view_segment_indices != -1))
    
    num_views = len(view_assignments)
    
    if num_views == 0:
        return -np.ones(num_points, dtype=int)
    
    global_segment_map = {}
    next_global_id = 0
    
    correspondence_matrices = []
    
    for i in range(num_views):
        for j in range(i+1, num_views):
            if not np.any(view_point_visibility[i]) or not np.any(view_point_visibility[j]):
                continue
                
            common_points = view_point_visibility[i] & view_point_visibility[j]
            
            if np.sum(common_points) < 10: 
                continue
            
            segments_i = view_assignments[i][common_points]
            segments_j = view_assignments[j][common_points]
            
            unique_i = np.unique(segments_i)
            unique_j = np.unique(segments_j)
            
            if len(unique_i) == 0 or len(unique_j) == 0:
                continue
                
            overlap_matrix = np.zeros((np.max(unique_i) + 1, np.max(unique_j) + 1), dtype=int)
            
            for idx in range(np.sum(common_points)):
                if segments_i[idx] != -1 and segments_j[idx] != -1:
                    overlap_matrix[segments_i[idx], segments_j[idx]] += 1
            
            segment_sizes_i = np.bincount(segments_i[segments_i != -1])
            segment_sizes_j = np.bincount(segments_j[segments_j != -1])
            
            if len(segment_sizes_i) < overlap_matrix.shape[0]:
                segment_sizes_i = np.pad(segment_sizes_i, (0, overlap_matrix.shape[0] - len(segment_sizes_i)))
            if len(segment_sizes_j) < overlap_matrix.shape[1]:
                segment_sizes_j = np.pad(segment_sizes_j, (0, overlap_matrix.shape[1] - len(segment_sizes_j)))
            
            sizes_i_expanded = segment_sizes_i[:, np.newaxis]
            sizes_j_expanded = segment_sizes_j[np.newaxis, :]
            
            min_sizes = np.minimum(sizes_i_expanded, sizes_j_expanded)
            min_sizes[min_sizes == 0] = 1
            
            iou_matrix = overlap_matrix / min_sizes
            
            correspondence_matrices.append({
                'view_i': i,
                'view_j': j,
                'matrix': iou_matrix,
                'overlap': overlap_matrix
            })
    
    if len(view_assignments) > 0:
        first_view = 0
        while first_view < len(view_assignments) and not np.any(view_assignments[first_view] != -1):
            first_view += 1
            
        if first_view < len(view_assignments):
            unique_segments = np.unique(view_assignments[first_view])
            unique_segments = unique_segments[unique_segments != -1]
            
            for seg_id in unique_segments:
                global_segment_map[(first_view, seg_id)] = next_global_id
                next_global_id += 1
            
            processed_views = {first_view}
            
            while len(processed_views) < num_views:
                made_progress = False
                
                for corr in correspondence_matrices:
                    view_i, view_j = corr['view_i'], corr['view_j']
                    
                    if (view_i in processed_views and view_j not in processed_views) or \
                       (view_j in processed_views and view_i not in processed_views):
                        
                        ref_view = view_i if view_i in processed_views else view_j
                        new_view = view_j if view_i in processed_views else view_i
                        
                        iou_matrix = corr['matrix']
                        if ref_view == view_j:  
                            iou_matrix = iou_matrix.T
                        
                        for new_seg in range(iou_matrix.shape[1]):
                            if np.any(iou_matrix[:, new_seg]):
                                best_match = np.argmax(iou_matrix[:, new_seg])
                                
                                if iou_matrix[best_match, new_seg] > 0.5:
                                    if (ref_view, best_match) in global_segment_map:
                                        global_id = global_segment_map[(ref_view, best_match)]
                                        global_segment_map[(new_view, new_seg)] = global_id
                                    else:
                                        global_segment_map[(new_view, new_seg)] = next_global_id
                                        next_global_id += 1
                                else:
                                    global_segment_map[(new_view, new_seg)] = next_global_id
                                    next_global_id += 1
                        
                        processed_views.add(new_view)
                        made_progress = True
                
                if not made_progress and len(processed_views) < num_views:
                    for view_idx in range(num_views):
                        if view_idx not in processed_views:
                            unique_segments = np.unique(view_assignments[view_idx])
                            unique_segments = unique_segments[unique_segments != -1]
                            
                            for seg_id in unique_segments:
                                global_segment_map[(view_idx, seg_id)] = next_global_id
                                next_global_id += 1
                            
                            processed_views.add(view_idx)
    
    point_segments = [[] for _ in range(num_points)]
    
    for view_idx, segment_indices in enumerate(view_assignments):
        for i in range(num_points):
            if view_point_visibility[view_idx][i] and segment_indices[i] != -1:
                global_id = global_segment_map.get((view_idx, segment_indices[i]), -1)
                if global_id != -1:
                    point_segments[i].append(global_id)
    
    final_segment_indices = -np.ones(num_points, dtype=int)
    
    for i in range(num_points):
        if len(point_segments[i]) > 0:
            counts = np.bincount(point_segments[i])
            final_segment_indices[i] = np.argmax(counts)
    
    return final_segment_indices

def save_point_cloud_with_segmentation(pcd: o3d.geometry.PointCloud, segment_indices: np.ndarray, output_path: str):
    """
    Save the point cloud to a PLY file and its segmentation indices to a separate NumPy file.
    """
    o3d.io.write_point_cloud(output_path, pcd)
    
    seg_indices_path = output_path.replace('.ply', '_segmentation_indices.npy')
    np.save(seg_indices_path, segment_indices)
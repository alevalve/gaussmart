import numpy as np
import open3d as o3d
import cv2
import os
import matplotlib.pyplot as plt
from typing import Dict, List
from identification.pc_projection import *


def visualize_projection_grid(points, camera, masks_list, dataset_type, view_idx, save_path=None, max_masks_per_row=5):
    """
    Visualize projections with a grid layout that handles many masks efficiently.
    """
    # Project points to 2D
    points_2d, depths = project_points_to_view(points, camera, dataset_type)
    
    # Get segmentation masks
    segmentation_masks = [mask['segmentation'] for mask in masks_list]
    if not segmentation_masks:
        print(f"No masks for view {view_idx}")
        return
    
    # Filter visible points
    h, w = segmentation_masks[0].shape
    in_bounds = np.all((points_2d >= [0, 0]) & (points_2d < [w, h]), axis=1)
    front_of_camera = depths > 0
    visible = in_bounds & front_of_camera
    
    # Assign segments
    segment_indices = assign_segment_indices(points_2d, segmentation_masks)
    
    # Calculate grid dimensions
    num_masks = len(segmentation_masks)
    rows = int(np.ceil((num_masks + 1) / max_masks_per_row))  # +1 for the overview plot
    cols = min(num_masks + 1, max_masks_per_row)
    
    # Create figure
    fig = plt.figure(figsize=(5*cols, 5*rows))
    
    # Plot 1: Overview of all projected points
    ax = plt.subplot(rows, cols, 1)
    ax.imshow(np.zeros((h, w)), cmap='gray', alpha=0.3)
    
    if np.any(visible):
        scatter = ax.scatter(points_2d[visible, 0], points_2d[visible, 1], 
                           c=depths[visible], s=1, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Depth')
    
    ax.set_title(f'View {view_idx}: All Points')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    
    # Plot masks in grid
    for mask_idx, mask in enumerate(segmentation_masks):
        plot_pos = mask_idx + 2  # +1 for 1-based index, +1 because first plot is position 1
        ax = plt.subplot(rows, cols, plot_pos)
        
        # Show mask as background
        ax.imshow(mask, cmap='Blues', alpha=0.3)
        
        # Show points in this segment
        points_in_segment = (segment_indices == mask_idx) & visible
        
        if np.any(points_in_segment):
            ax.scatter(points_2d[points_in_segment, 0], points_2d[points_in_segment, 1], 
                      c='red', s=2, alpha=0.8)
        
        ax.set_title(f'Segment {mask_idx}')
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}_view_{view_idx:03d}.png", dpi=150, bbox_inches='tight')
    
    plt.show()
    
    # Print stats
    print(f"View {view_idx}:")
    print(f"  Total points: {len(points)}")
    print(f"  Visible points: {np.sum(visible)}")
    print(f"  Points in segments: {np.sum(segment_indices[visible] != -1)}")
    print(f"  Visualization layout: {rows}x{cols} grid")

def visualize_all_views_grid(points, cameras, all_masks, dataset_type, max_views=5, max_masks_per_row=5):
    """
    Visualize projections for multiple views using grid layout.
    """
    num_views = min(len(all_masks), max_views)
    
    for view_idx in range(num_views):
        if f'camera_{view_idx:03d}' in cameras:
            camera = cameras[f'camera_{view_idx:03d}']
            masks_list = all_masks[view_idx]
            
            visualize_projection_grid(
                points, camera, masks_list, dataset_type, 
                view_idx, max_masks_per_row=max_masks_per_row
            )

def quick_check_projection_grid(points, cameras, all_masks, dataset_type, list_views):
    """
    Quick check with grid layout for a specific view.
    """
    for view_idx in list_views:
        camera_key = f'camera_{view_idx:03d}'
        if camera_key not in cameras:
            print(f"Camera {camera_key} not found")
            return
        
        camera = cameras[camera_key]
        masks_list = all_masks[view_idx] if view_idx < len(all_masks) else []
        
        if not masks_list:
            print(f"No masks for view {view_idx}")
            return
        
        # Project points
        points_2d, depths = project_points_to_view(points, camera, dataset_type)
        
        # Get first mask for size info
        mask = masks_list[0]['segmentation']
        h, w = mask.shape
        
        # Filter visible points
        in_bounds = np.all((points_2d >= [0, 0]) & (points_2d < [w, h]), axis=1)
        front_of_camera = depths > 0
        visible = in_bounds & front_of_camera
        
        print(f"=== Quick Check View {view_idx} ===")
        print(f"Total points: {len(points)}")
        print(f"Points in bounds: {np.sum(in_bounds)}")
        print(f"Points in front of camera: {np.sum(front_of_camera)}")
        print(f"Visible points: {np.sum(visible)}")
        print(f"Image size: {w}x{h}")
        print(f"Number of masks: {len(masks_list)}")
        
        # Grid layout visualization
        if np.any(visible):
            num_masks = len(masks_list)
            cols = 3  # Fixed number of columns for quick check
            rows = int(np.ceil((num_masks + 2) / cols))  # +2 for overview and mask
            
            fig = plt.figure(figsize=(5*cols, 5*rows))
            
            # Plot 1: Overview
            ax = plt.subplot(rows, cols, 1)
            ax.imshow(np.zeros((h, w)), cmap='gray', alpha=0.3)
            ax.scatter(points_2d[visible, 0], points_2d[visible, 1], 
                    c='red', s=1, alpha=0.7)
            ax.set_title(f'All Projected Points')
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            
            # Plot 2: First mask
            ax = plt.subplot(rows, cols, 2)
            ax.imshow(mask, cmap='Blues', alpha=0.3)
            ax.set_title('First Segmentation Mask')
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            
            # Plot points in segments
            segment_indices = assign_segment_indices(points_2d, [m['segmentation'] for m in masks_list])
            
            for mask_idx in range(min(num_masks, rows*cols - 2)):
                ax = plt.subplot(rows, cols, mask_idx + 3)
                current_mask = masks_list[mask_idx]['segmentation']
                ax.imshow(current_mask, cmap='Blues', alpha=0.3)
                
                points_in_segment = (segment_indices == mask_idx) & visible
                if np.any(points_in_segment):
                    ax.scatter(points_2d[points_in_segment, 0], points_2d[points_in_segment, 1], 
                            c='red', s=2, alpha=0.8)
                
                ax.set_title(f'Segment {mask_idx}')
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
            
            plt.tight_layout()
            plt.savefig(f"grid_view_{view_idx}.png")
            plt.show()
        else:
            print("No visible points to display!")
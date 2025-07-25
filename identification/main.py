import os
import torch
import numpy as np
import json
import argparse
import open3d as o3d
import sys
from process_selected_views import ProcessSelectedViews
from sam import SAMSegmentation
import shutil
from pc_projection import aggregate_segment_indices, load_point_cloud

def setup_directories(output_base_path: str) -> tuple:
    """
    Create the directory structure needed for the processing pipeline.
    Returns tuples of paths to the created directories.
    """
    base_dir = os.path.join(output_base_path, 'segments')
    boxes_dir = os.path.join(base_dir, 'boxes')
    masks_dir = os.path.join(base_dir, 'masks')
    cameras_dir = os.path.join(base_dir, 'cameras')
    segmented_images_dir = os.path.join(base_dir, 'segmented_images')
    point_cloud_dir = os.path.join(base_dir, 'point_cloud')
    raw_images_dir = os.path.join(base_dir, 'images') 
    indices_dir = os.path.join(base_dir, 'indices')
    
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    for directory in [boxes_dir, masks_dir, cameras_dir, segmented_images_dir, point_cloud_dir, raw_images_dir, indices_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return boxes_dir, masks_dir, cameras_dir, segmented_images_dir, point_cloud_dir, raw_images_dir, indices_dir

def get_point_cloud_path(scan_path: str, dataset_type: str) -> str:
    """Get point cloud path based on dataset type"""
    if dataset_type.lower() == 'dtu':
        return os.path.join(scan_path, "points.ply")
    else:
        return os.path.join(scan_path, "sparse/0/points3D.ply")

def run_segmentation(scan_path: str, output_base_path: str, dataset_type: str):
    """Run segmentation process with given parameters"""
    try:
        # Setup directories
        boxes_dir, masks_dir, cameras_dir, segmented_images_dir, point_cloud_dir, raw_images_dir, indices_dir = setup_directories(output_base_path)
        
        # Initialize components
        if dataset_type.lower() == 'dtu':
            camera_path = os.path.join(scan_path, 'cameras.npz')
        else:  
            camera_path = os.path.join(scan_path, 'poses_bounds.npy')
        
        images_dir = os.path.join(scan_path, 'images')
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'weights', 'sam_vit_h_4b8939.pth')
        
        processor = ProcessSelectedViews(camera_path, images_dir, output_base_path, dataset_type)
        selection_results = processor.process_views()
        selected_data = processor.get_selected_data(selection_results['selected_indices'])
        
        segmenter = SAMSegmentation(checkpoint_path)
        results = []
        
        # Process each selected view
        for i, image_path in enumerate(selected_data['image_paths']):
            image_filename = os.path.basename(image_path)
            raw_image_dest = os.path.join(raw_images_dir, f"{i:03d}_{image_filename}")
            shutil.copy2(image_path, raw_image_dest)
            
            result = segmenter.process_image(image_path)
            output_name = f"{i:03d}"
            segments_path = os.path.join(masks_dir, f"segments_{output_name}.npz")
            segmenter.save_segments_boxes(result, segments_path)
            
            segmenter.visualize_masks(
                image_path=image_path,
                result=result,
                output_name=output_name,
                output_dir=segmented_images_dir
            )
            
            results.append({
                "masks": result,
                "camera_data": selected_data['camera_parameters'][f"camera_{i:03d}"]
            })
        
        # Save camera data and indices
        selected_indices = selection_results['selected_indices']
        if isinstance(selected_indices, np.ndarray):
            str_indices = [str(int(x)) for x in selected_indices.tolist()]
        else:
            str_indices = [str(int(x)) for x in selected_indices]
        
        json_path = os.path.join(indices_dir, "selected_indices.json")
        with open(json_path, "w") as f:
            json.dump(str_indices, f, indent=2)

        camera_data = {
            'selected_indices': selected_indices,
            'cameras_dict': {},
            'cameras_list': []
        }

        for i, idx in enumerate(selected_indices):
            camera_id = f"camera_{i:03d}"
            view_data = processor.analyzer.views[idx]
            camera_data['cameras_dict'][camera_id] = view_data
            camera_data['cameras_list'].append(view_data)

        camera_params_path = os.path.join(cameras_dir, 'selected_cameras.npz')
        np.savez(camera_params_path, **camera_data)

        # Process point cloud
        point_cloud_path = get_point_cloud_path(scan_path, dataset_type)
        
        if os.path.exists(point_cloud_path):
            pcd, points = load_point_cloud(point_cloud_path, dataset_type)
            all_masks = [result["masks"] for result in results]
            
            segment_indices = aggregate_segment_indices(
                points=points,
                all_masks=all_masks,
                cameras=camera_data['cameras_dict'],
                dataset_type=dataset_type
            )
            
            output_point_cloud = os.path.join(point_cloud_dir, "segmented_point_cloud.ply")
            output_indices = os.path.join(point_cloud_dir, "segment_indices.npy")
            
            o3d.io.write_point_cloud(output_point_cloud, pcd)
            np.save(output_indices, segment_indices)
            
            return segment_indices
        else:
            print(f"Warning: Point cloud not found at {point_cloud_path}")
            return None
            
    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Process camera views and generate segmentations')
    parser.add_argument('-s', '--scan_path', type=str, required=True,
                       help='Base path to the scan directory')
    parser.add_argument('-o', '--output_base_path', type=str, required=True,
                       help='Base path where output files will be saved')
    parser.add_argument('-t', '--type', type=str, choices=['dtu', 'nerf', 'tyt'], required=True,
                       help='Type of dataset (dtu or nerf)')
    
    args = parser.parse_args()
    
    try:
        segment_indices = run_segmentation(args.scan_path, args.output_base_path, args.type)
        if segment_indices is not None:
            print("\nSegmentation completed successfully!")
            print(f"Segment indices saved to: {os.path.join(args.output_base_path, 'segments/results/point_cloud/segment_indices.npy')}")
        else:
            print("\nSegmentation completed with warnings (no point cloud found)")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
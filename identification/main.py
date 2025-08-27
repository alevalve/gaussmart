import os
import shutil
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from filter.hull_removal import HullRemoval
from identification.process_selected_views import ProcessSelectedViews
from identification.sam import SAMSegmentation
from identification.pc_projection import project_points_to_view, assign_segment_indices_simple


class Pipeline:
    """Clutering/Segmentation/Projection pipeline."""
    
    def __init__(self, args):
        self.scan_path = args.scan_path
        self.output_path = args.output_path
        self.dataset_type = args.type.lower()
        self.cluster_cameras = not args.skip_camera_clustering
        self.sam2 = args.sam2
        self.dirs = self._setup_directories()
        
    def _setup_directories(self) -> Dict[str, str]:
        """Create output directory structure."""
        base_dir = os.path.join(self.output_path, 'segments')
        dirs = {
            'base': base_dir,
            'images': os.path.join(base_dir, 'images'),
            'masks': os.path.join(base_dir, 'masks'),
            'point_cloud': os.path.join(base_dir, 'point_cloud'),
            'embeddings': os.path.join(base_dir, 'embeddings'),
            'cameras': os.path.join(base_dir, 'cameras'),
        }
        
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
            
        return dirs

    def _get_paths(self) -> Tuple[str, str]:
        """Get dataset-specific file paths."""
        if self.dataset_type == 'dtu':
            pc_path = os.path.join(self.scan_path, "points.ply")
            camera_path = os.path.join(self.scan_path, 'cameras.npz')
        else:
            pc_path = os.path.join(self.scan_path, "sparse/0/points3D.ply")
            camera_path = os.path.join(self.scan_path, 'poses_bounds.npy')
        return pc_path, camera_path

    def select_views(self) -> Tuple[List[int], Dict, ProcessSelectedViews]:
        """Select optimal views for segmentation."""
        _, camera_path = self._get_paths()
        image_root = os.path.join(self.scan_path, 'images')
        
        processor = ProcessSelectedViews(camera_path, image_root, self.output_path, self.dataset_type, self.cluster_cameras)
        sel_info = processor.process_views()
        selected_indices = sel_info['selected_indices']
        selected_data = processor.get_selected_data(selected_indices, already_mapped=True)
        
        # Save camera data
        cameras_dict = {f'camera_{i:03d}': processor.analyzer.views[idx] 
                       for i, idx in enumerate(selected_indices)}
        camera_data = {
            'selected_indices': selected_indices,
            'cameras_dict': cameras_dict,
        }
        np.savez(os.path.join(self.dirs['cameras'], 'selected_cameras.npz'), **camera_data)
        
        return selected_indices, selected_data, processor

    def run_sam_segmentation(self, selected_data: Dict) -> List[List[Dict]]:
        """Run SAM segmentation on selected views."""
        weights_path = os.path.join(Path(__file__).resolve().parent, 'weights', 'sam_vit_h_4b8939.pth')
        segmenter = SAMSegmentation(weights_path, sam2=self.sam2)
        
        all_masks = []
        for i, image_path in enumerate(selected_data['image_paths']):
            # Copy image
            dst = os.path.join(self.dirs['images'], f"{os.path.basename(image_path)}")
            shutil.copy2(image_path, dst)
            
            # Run segmentation
            masks = segmenter.process_image(image_path)
            seg_path = os.path.join(self.dirs['masks'], f"segments_{i:03d}.npz")
            segmenter.save_segments_boxes(masks, seg_path)
            
            all_masks.append(masks)
            
        return all_masks

    def load_point_cloud(self, clean: bool = True) -> Optional[o3d.geometry.PointCloud]:
       
        pc_path, _ = self._get_paths()
        if not os.path.exists(pc_path):
            print(f"Warning: Point cloud not found at {pc_path}")
            return None
            
        pcd = o3d.io.read_point_cloud(pc_path)

        if clean:
            print("Applying hull removal filtering...")
            hull_removal = HullRemoval(pcd)
            filtered_points, _, pcd_filtered = hull_removal.forward()
            o3d.io.write_point_cloud(os.path.join(self.dirs['point_cloud'], "raw_pc.ply"), pcd_filtered)
            return pcd_filtered
        else:
            o3d.io.write_point_cloud(os.path.join(self.dirs['point_cloud'], "raw_pc.ply"), pcd)
            return pcd

    def project_segments(self, points: np.ndarray, all_masks: List[List[Dict]], 
                        cameras_dict: Dict) -> Tuple[np.ndarray, Dict[int, int]]:
        """Project 3D points to 2D views and assign segment labels."""
        segment_indices = -np.ones(len(points), dtype=int)
        mask_areas = {}
        
        for view_idx, masks_list in enumerate(all_masks):
            if not masks_list:
                continue
                
            camera = cameras_dict[f'camera_{view_idx:03d}']
            segmentation_masks = [m['segmentation'] for m in masks_list]
            
            # Calculate mask areas
            for mask_idx, mask in enumerate(segmentation_masks):
                area = int(np.sum(mask > 0))
                mask_areas[mask_idx] = max(mask_areas.get(mask_idx, 0), area)
            
            # Project points and assign labels
            h, w = segmentation_masks[0].shape
            points_2d, depths = project_points_to_view(points, camera, self.dataset_type)
            
            # Find visible, unassigned points
            in_bounds = np.all((points_2d >= [0, 0]) & (points_2d < [w, h]), axis=1)
            visible = in_bounds & (depths > 0) & (segment_indices == -1)
            
            if not np.any(visible):
                continue
                
            # Assign segments
            points_2d_clipped = np.clip(points_2d[visible], [0, 0], [w - 1, h - 1])
            view_segments = assign_segment_indices_simple(points_2d_clipped, segmentation_masks)
            segment_indices[visible] = np.where(view_segments != -1, view_segments, -1)
            
        return segment_indices, mask_areas

    def save_results(self, pcd: o3d.geometry.PointCloud, segment_indices: np.ndarray, 
                    mask_areas: Dict[int, int]):
        """Save segmentation results."""
        # Save point cloud with original properties
        output_pcd = o3d.geometry.PointCloud()
        output_pcd.points = pcd.points
        if pcd.has_normals():
            output_pcd.normals = pcd.normals
        if pcd.has_colors():
            output_pcd.colors = pcd.colors
            
        o3d.io.write_point_cloud(os.path.join(self.dirs['point_cloud'], "segmented_point_cloud.ply"), output_pcd)
        np.save(os.path.join(self.dirs['point_cloud'], "segment_indices.npy"), segment_indices)
        np.save(os.path.join(self.dirs['point_cloud'], "mask_areas.npy"), mask_areas)

    def run(self, clean_pc: bool = True) -> Tuple[Optional[np.ndarray], Optional[Dict[int, int]]]:
        """Execute the complete segmentation pipeline."""

        print("1. Selecting optimal views...")
        selected_indices, selected_data, processor = self.select_views()
        
        print("2. Running SAM segmentation...")
        all_masks = self.run_sam_segmentation(selected_data)
        
        print("3. Loading point cloud...")
        pcd = self.load_point_cloud(clean=clean_pc)
        if pcd is None:
            return None, None
            
        print("4. Projecting segments to 3D...")
        cameras_dict = {f'camera_{i:03d}': processor.analyzer.views[idx] 
                    for i, idx in enumerate(selected_indices)}
        segment_indices, mask_areas = self.project_segments(
            np.asarray(pcd.points), all_masks, cameras_dict
        )
        
        print("5. Saving results...")
        self.save_results(pcd, segment_indices, mask_areas)
        
        return segment_indices, mask_areas


def main():
    import argparse
    parser = argparse.ArgumentParser(description='3D Point Cloud Segmentation Pipeline')
    parser.add_argument('-s', '--scan_path', required=True, help='Path to scan folder')
    parser.add_argument('-o', '--output_path', required=True, help='Output directory')
    parser.add_argument('-t', '--type', choices=['dtu', 'nerf', 'tyt'], required=True, help='Dataset type')
    parser.add_argument('--skip_camera_clustering', action='store_true', help='Skip camera clustering')
    parser.add_argument('--sam2', action='store_true', help='Use SAM2 instead of SAM1')
    parser.add_argument('--clean', action='store_true', help='Apply hull removal filtering to point cloud')
    
    args = parser.parse_args()
    
    pipeline = Pipeline(args)
    segment_indices, mask_areas = pipeline.run(clean_pc=args.clean)
    


if __name__ == "__main__":
    main()
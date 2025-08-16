import os
import shutil
import numpy as np
from typing import Dict, List
from pathlib import Path
try:
    from .analyze_cameras import AnalyzeCameras
    from .clustering_cameras import CameraClustering
except ImportError:
    from analyze_cameras import AnalyzeCameras
    from clustering_cameras import CameraClustering

class ProcessSelectedViews:
    def __init__(self, camera_path: str, images_dir: str, output_dir: str, dataset_type: str = None):
        self.camera_path = camera_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.dataset_type = dataset_type 
        self.analyzer = None
        self.clusterer = None
        self.setup()

    def setup(self):
        """Set up analyzer and clusterer"""
        self.analyzer = AnalyzeCameras(self.camera_path, self.images_dir)
        self.clusterer = CameraClustering(self.analyzer)

    def _filter_image_files(self, files: List[str]) -> List[str]:
        """Filter out hidden and system files"""
        filtered = [f for f in files if not f.startswith('.') and not f.startswith('._')]
        if len(files) != len(filtered):
            print(f"Filtered out {len(files) - len(filtered)} files")
            print("First few filtered out files:", [f for f in files if f not in filtered][:5])
        return filtered

    def _map_camera_to_image_index(self, camera_idx: int) -> int:
        """Map camera index to image index based on dataset type"""
        if self.dataset_type and self.dataset_type.lower() == 'tyt':
            # For Tanks and Temples, divide camera index by 2 and round down
            return camera_idx // 2
        return camera_idx

    def process_views(self) -> Dict:
        """Process and save selected views"""
        selection_results = self.clusterer.select_representative_cameras()
        selected_indices = selection_results['selected_indices']
        
        print("\nSelection Results:")
        print(f"Number of selected indices: {len(selected_indices)}")
        print(f"Selected indices: {selected_indices}")

        # Map camera indices to image indices for Tanks and Temples
        if self.dataset_type and self.dataset_type.lower() == 'tyt':
            mapped_indices = [self._map_camera_to_image_index(idx) for idx in selected_indices]
            print(f"Mapped indices for TanksAndTemples: {mapped_indices}")
        else:
            mapped_indices = selected_indices

        selected_indices = [int(idx) for idx in mapped_indices]

        return { 'selected_indices': selected_indices }

    def save_camera_parameters(self, selected_indices: List[int], camera_dir: str):
        """Save selected camera parameters with all formats in one file"""
        camera_data = {
            'selected_indices': selected_indices,
            'cameras_dict': {},
            'cameras_list': []
        }
        
        for i, idx in enumerate(selected_indices):
            camera_id = f"camera_{i:03d}"
            view_data = self.analyzer.views[idx]
            camera_data['cameras_dict'][camera_id] = view_data
            camera_data['cameras_list'].append(view_data)
        
        save_path = os.path.join(camera_dir, 'selected_cameras.npz')
        np.savez(save_path, **camera_data)

    def copy_selected_images(self, selected_indices: List[int], output_dir: str) -> List[str]:
        """Copy selected images to output directory"""
        all_files = sorted(os.listdir(self.images_dir))
        
        image_files = self._filter_image_files(all_files)
        
        copied_images = []
        
        for i, idx in enumerate(selected_indices):
            
            if idx < len(image_files):
                # Format the image filename according to TanksAndTemples convention if needed
                if self.dataset_type and self.dataset_type.lower() == 'tyt':
                    # Try both 5-digit and 6-digit formats
                    src_filename_5digit = f"{idx * 2:05d}.jpg"
                    src_filename_6digit = f"{idx * 2:06d}.jpg"
                    
                    src_path_5digit = os.path.join(self.images_dir, src_filename_5digit)
                    src_path_6digit = os.path.join(self.images_dir, src_filename_6digit)
                    
                    if os.path.exists(src_path_5digit):
                        src_path = src_path_5digit
                    elif os.path.exists(src_path_6digit):
                        src_path = src_path_6digit
                    else:
                        print(f"Warning: Source image not found for index {idx} in either format")
                        continue
                else:
                    src_filename = image_files[idx]
                    src_path = os.path.join(self.images_dir, src_filename)
                
                dst_path = os.path.join(output_dir, f"image_{i:03d}.jpg")
                
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    copied_images.append(dst_path)
                    print(f"Successfully copied image {idx} to {dst_path}")
                else:
                    print(f"Warning: Source image {src_path} not found")
            else:
                print(f"Warning: Index {idx} is out of range (max index: {len(image_files)-1})")
        
        print(f"\nTotal images copied: {len(copied_images)}")
        return copied_images

    def get_selected_data(self, selected_indices: List[int], already_mapped=False) -> Dict:
        """Get data for selected views"""
        all_files = sorted(os.listdir(self.images_dir))
        image_files = self._filter_image_files(all_files)
        camera_parameters = {}
        image_paths = []

        # Map indices if needed
        if self.dataset_type and self.dataset_type.lower() == 'tyt' and not already_mapped:
            mapped_indices = [idx // 2 for idx in selected_indices]
        else:
            mapped_indices = selected_indices

        for i, (cam_idx, img_idx) in enumerate(zip(selected_indices, mapped_indices)):
            # Add camera parameters using original camera index
            camera_id = f"camera_{i:03d}"
            camera_parameters[camera_id] = self.analyzer.views[cam_idx]

            # Get image path using mapped index
            if img_idx < len(image_files):
                if self.dataset_type and self.dataset_type.lower() == 'tyt':
                    # Try both 5-digit and 6-digit formats
                    image_filename_5digit = f"{img_idx:05d}.jpg"
                    image_filename_6digit = f"{img_idx:06d}.jpg"
                    
                    # Check which format exists
                    path_5digit = os.path.join(self.images_dir, image_filename_5digit)
                    path_6digit = os.path.join(self.images_dir, image_filename_6digit)
                    
                    if os.path.exists(path_5digit):
                        image_path = path_5digit
                    elif os.path.exists(path_6digit):
                        image_path = path_6digit
                    else:
                        print(f"Warning: Image not found for index {img_idx} in either format")
                        continue
                else:
                    image_filename = image_files[img_idx]
                    image_path = os.path.join(self.images_dir, image_filename)
                
                if os.path.exists(image_path):
                    image_paths.append(image_path)

        selected_indices = [int(idx) for idx in mapped_indices]

        return {
            'indices': selected_indices,
            'image_paths': image_paths,
            'camera_parameters': camera_parameters
        }
import numpy as np
from typing import Dict, Tuple, Optional
import os

class CameraLoader:
    """Handles loading camera data from different dataset formats (DTU, NeRF, and Tanks and Temples)"""
    @staticmethod
    def detect_format(camera_path: str) -> str:
        """
        Detect the dataset format based on file extension and content structure.
        Returns: 'dtu', 'nerf', or 'tyt' based on the identified format.
        """
        ext = os.path.splitext(camera_path)[1].lower()
        if ext == '.npz':
            return 'dtu'
        elif ext == '.npy':
            try:
                data = np.load(camera_path)
                if data.ndim == 2:
                    if data.shape[1] == 17:
                        return 'nerf'
                    elif data.shape[1] == 14:
                        return 'tyt'
            except Exception:
                pass
        raise ValueError(f"Unrecognized camera data format for file: {camera_path}")

    @staticmethod
    def load_dtu_cameras(camera_path: str) -> Dict:
        """
        Load camera data in DTU format from an NPZ file, organizing views by their indices.
        """
        npz_data = np.load(camera_path)
        views = {}
        for key in npz_data.files:
            if '_' in key:
                mat_type, view_num = key.rsplit('_', 1)
                view_num = int(view_num)
                if view_num not in views:
                    views[view_num] = {}
                views[view_num][mat_type] = npz_data[key]
        return views

    @staticmethod
    def load_nerf_cameras(camera_path: str) -> Dict:
        """
        Load camera data in NeRF format, converting from camera-to-world transforms to the 
        required world-to-camera matrices and calculating appropriate camera matrices.
        """
        data = np.load(camera_path)
        views = {}
        for i, cam_data in enumerate(data):
            c2w = cam_data[:16].reshape(4, 4)
            focal = cam_data[16]
            
            world_mat = np.linalg.inv(c2w)
            
            camera_mat = np.array([
                [focal, 0, 512, 0],
                [0, focal, 512, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            views[i] = {
                'world_mat': world_mat,
                'camera_mat': camera_mat,
                'scale_mat': np.eye(4)
            }
        return views

    @staticmethod
    def load_tyt_cameras(camera_path: str) -> Dict:
        """
        Load camera data in Tanks and Temples (TYT) format, handling the specialized 
        scaling and intrinsic parameters needed for this dataset.
        """
        data = np.load(camera_path)
        views = {}
        
        positions = data[:, [3, 7, 11]]  
        center = np.mean(positions, axis=0)
        scale = 1.0 / np.max(np.abs(positions - center))
        
        H = 543
        W = 979
        
        fx = 501.0  
        fy = 277.0
        cx = W // 2
        cy = H // 2
        
        for i, pose_data in enumerate(data):
            c2w = np.eye(4)
            c2w[:3, :4] = pose_data[:12].reshape(3, 4)
            
            w2c = np.linalg.inv(c2w)
            
            camera_mat = np.array([
                [fx, 0, cx, 0],
                [0, fy, cy, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            near, far = pose_data[12:] * scale
            
            views[i] = {
                'world_mat': w2c,          
                'camera_mat': camera_mat, 
                'scale_mat': np.eye(4),     
                'bounds': np.array([near, far]),
                'img_size': np.array([W, H])  
            }
        
        return views

    @classmethod
    def load_cameras(cls, camera_path: str) -> Tuple[Dict, str]:
        """
        Unified entry point to load camera data from any supported format.
        Detects format automatically and calls the appropriate loader method.
        Returns: (views_dict, format_type)
        """
        format_type = cls.detect_format(camera_path)
        if format_type == 'dtu':
            views = cls.load_dtu_cameras(camera_path)
        elif format_type == 'nerf':
            views = cls.load_nerf_cameras(camera_path)
        elif format_type == 'tyt':
            views = cls.load_tyt_cameras(camera_path)
        else:
            raise Exception("Sorry, the dataset is not configured")
        return views, format_type
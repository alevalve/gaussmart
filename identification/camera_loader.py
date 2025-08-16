import numpy as np
from typing import Dict, Tuple, Optional, Any
import os
import warnings


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
            # Check for DTU keys in the archive
            try:
                npz = np.load(camera_path)
                keys = set(npz.files)
                if any(k.startswith('world_mat_') for k in keys) and any(k.startswith('camera_mat_') for k in keys):
                    return 'dtu'
            except Exception:
                pass
        elif ext == '.npy':
            # Could be NeRF or Tanks & Temples
            try:
                data = np.load(camera_path)
                if data.ndim == 2:
                    cols = data.shape[1]
                    if cols in (17, 19):
                        return 'nerf'
                    elif cols in (14, 16):
                        return 'tyt'
            except Exception:
                pass
        raise ValueError(f"Unrecognized camera data format for file: {camera_path}")

    @staticmethod
    def load_dtu_cameras(camera_path: str) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Load camera data in DTU format from an NPZ file, organizing views by their indices.
        """
        npz_data = np.load(camera_path)
        views: Dict[int, Dict[str, Any]] = {}
        for key in npz_data.files:
            if '_' in key:
                mat_type, view_str = key.rsplit('_', 1)
                if not view_str.isdigit():
                    continue
                view = int(view_str)
                views.setdefault(view, {})[mat_type] = npz_data[key]
        # Validate each view
        for vid, cam in views.items():
            assert 'world_mat' in cam and 'camera_mat' in cam and 'scale_mat' in cam, \
                f"DTU view {vid} missing required matrices"
        return views

    @staticmethod
    def load_nerf_cameras(
        camera_path: str,
        img_wh: Tuple[int, int] = (1024, 1024),
        assume_bounds: bool = True
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Load camera data in NeRF format, converting from camera-to-world transforms
        to world-to-camera matrices and calculating intrinsics.
        Args:
            camera_path: Path to .npy file containing Nx(17 or 19) floats per camera.
            img_wh: Tuple (width, height) of training images (default 1024x1024).
            assume_bounds: If True and data has 19 floats, reads last two as [near, far].
        """
        data = np.load(camera_path)
        H, W = img_wh[1], img_wh[0]
        views: Dict[int, Dict[str, Any]] = {}
        for i, cam_data in enumerate(data):
            # First 16 values are flatten 4x4 c2w matrix
            c2w = cam_data[:16].reshape(4, 4)
            world_mat = np.linalg.inv(c2w)

            # Focal length is next value
            focal = float(cam_data[16])
            # Principal point assumed at image center
            cx = W / 2.0
            cy = H / 2.0
            camera_mat = np.array([
                [focal,    0.0, cx, 0.0],
                [   0.0, focal, cy, 0.0],
                [   0.0,    0.0, 1.0, 0.0],
                [   0.0,    0.0, 0.0, 1.0]
            ], dtype=float)

            entry: Dict[str, Any] = {
                'world_mat': world_mat,
                'camera_mat': camera_mat,
                'scale_mat': np.eye(4, dtype=float)
            }
            if assume_bounds and cam_data.size >= 18:
                bounds = cam_data[17:19].astype(float)
                entry['bounds'] = bounds
            views[i] = entry
        return views

    @staticmethod
    def load_tyt_cameras(
        camera_path: str,
        img_wh: Optional[Tuple[int, int]] = None,
        intrinsics: Optional[Dict[str, float]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Load camera data in Tanks & Temples (TYT) format, handling scaling and intrinsics.
        Args:
            camera_path: Path to .npy file containing Nx14 or Nx16 floats per camera.
            img_wh: Optional (width, height). If None, defaults must be provided in intrinsics.
            intrinsics: Optional dict with keys 'fx','fy','cx','cy'. If None, defaults are used.
        """
        data = np.load(camera_path)

        half_point = data.shape[0] // 2

        data = data[:half_point]
        # Default image size / intrinsics if not provided
        if img_wh is None:
            img_wh = (979, 543)
        if intrinsics is None:
            intrinsics = {'fx': 501.0, 'fy': 277.0, 'cx': img_wh[0]/2.0, 'cy': img_wh[1]/2.0}

        H, W = img_wh[1], img_wh[0]
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        # Compute scene scale from camera positions
        positions = data[:, [3, 7, 11]]  # translation components
        center = np.mean(positions, axis=0)
        scale = 1.0 / np.max(np.abs(positions - center))

        views: Dict[int, Dict[str, Any]] = {}
        for i, pose in enumerate(data):
            # Build c2w and invert
            c2w = np.eye(4, dtype=float)
            c2w[:3, :4] = pose[:12].reshape(3, 4)
            world_mat = np.linalg.inv(c2w)

            camera_mat = np.array([
                [fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=float)

            entry: Dict[str, Any] = {
                'world_mat': world_mat,
                'camera_mat': camera_mat,
                'scale_mat': np.eye(4, dtype=float),
                'img_size': np.array([W, H], dtype=int)
            }
            # Parse near/far if present
            if pose.size >= 14:
                near, far = pose[12:14].astype(float) * scale
                entry['bounds'] = np.array([near, far], dtype=float)
            views[i] = entry
        return views

    @classmethod
    def load_cameras(
        cls,
        camera_path: str,
        **kwargs
    ) -> Tuple[Dict[int, Dict[str, Any]], str]:
        """
        Unified entry point to load camera data from any supported format.
        Detects format automatically and calls the appropriate loader.
        Extra keyword args are passed to specific loaders.
        Returns: (views_dict, format_type)
        """
        fmt = cls.detect_format(camera_path)
        if fmt == 'dtu':
            views = cls.load_dtu_cameras(camera_path)
        elif fmt == 'nerf':
            views = cls.load_nerf_cameras(camera_path, **kwargs)
        elif fmt == 'tyt':
            views = cls.load_tyt_cameras(camera_path, **kwargs)
        else:
            raise ValueError(f"Unsupported camera format: {fmt}")

        for vid, cam in views.items():
            if 'world_mat' not in cam or 'camera_mat' not in cam:
                raise AssertionError(f"View {vid} missing required matrices in format {fmt}")
        return views, fmt

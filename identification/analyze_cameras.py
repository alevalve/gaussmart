import numpy as np
from typing import List, Tuple, Dict
from camera_loader import CameraLoader

class AnalyzeCameras:
    def __init__(self, camera_path: str, images_dir: str, device='cpu'):
        self.device = device
        self.camera_path = camera_path
        self.images_dir = images_dir
        self.cameras = None
        self.camera_data = {}
        self.format_type = None
        self.open_cameras()
    
    def open_cameras(self):
        """Load and organize camera data"""
        try:
            self.views, self.format_type = CameraLoader.load_cameras(self.camera_path)
            print(f"Loaded {len(self.views)} views in {self.format_type} format")
            
        except Exception as e:
            print(f"Error loading cameras: {e}")

    def analyze_cameras(self) -> Dict:
        """Analyze camera positions and orientations"""
        if not self.views:
            return {}
        
        positions = []
        rotations = []
        
        for view_num, matrices in self.views.items():
            if 'world_mat' in matrices:
                world_mat = matrices['world_mat']
                pos = world_mat[:3, 3]
                rot = world_mat[:3, :3]
                
                positions.append(pos)
                rotations.append(rot)
        
        positions = np.array(positions)
        rotations = np.array(rotations)
        
        stats = {
            'format_type': self.format_type,
            'num_cameras': len(positions),
            'position_range': {
                'x': (float(positions[:,0].min()), float(positions[:,0].max())),
                'y': (float(positions[:,1].min()), float(positions[:,1].max())),
                'z': (float(positions[:,2].min()), float(positions[:,2].max()))
            },
            'position_mean': positions.mean(axis=0).tolist(),
            'position_std': positions.std(axis=0).tolist()
        }
        
        if len(rotations) > 0:
            angles = self._compute_angles(rotations)
            stats['angle_distribution'] = {
                'mean': angles.mean(axis=0).tolist(),
                'std': angles.std(axis=0).tolist()
            }
        
        return stats

    def _compute_angles(self, rotations: np.ndarray) -> np.ndarray:
        """Convert rotation matrices to Euler angles"""
        angles = []
        for R in rotations:
            roll = np.arctan2(R[2,1], R[2,2])
            pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
            yaw = np.arctan2(R[1,0], R[0,0])
            angles.append([roll, pitch, yaw])
        return np.degrees(np.array(angles))
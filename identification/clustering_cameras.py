import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Any
from identification.analyze_cameras import AnalyzeCameras

class CameraClustering:
    """
    Cluster camera poses and select representative views, ensuring consistent
    handling for DTU, NeRF, and Tanks & Temples formats via explicit c2w matrices.
    """
    def __init__(self, analyzer: AnalyzeCameras):
        self.camera_analyzer = analyzer
        self.positions: np.ndarray = np.empty((0, 3), dtype=float)
        self.view_directions: np.ndarray = np.empty((0, 3), dtype=float)
        self._extract_camera_data()

    @staticmethod
    def _normalize_positions(positions: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        # Center on mean and scale by per-axis std (robust)
        center = positions.mean(axis=0)
        centered = positions - center
        scale = np.std(centered, axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        normalized = centered / scale
        return normalized, center, scale

    def _extract_camera_data(self) -> None:
        # Extract camera centers and forward axes (z) from c2w
        pos_list: List[np.ndarray] = []
        dir_list: List[np.ndarray] = []
        for vid, mats in self.camera_analyzer.views.items():
            if 'c2w' in mats:
                c2w = mats['c2w']
            elif 'world_mat' in mats:
                c2w = np.linalg.inv(mats['world_mat'])
            else:
                continue
            pos_list.append(c2w[:3, 3])          # camera position
            dir_list.append(c2w[:3, 2])          # forward direction
        if pos_list:
            self.positions = np.vstack(pos_list)
            self.view_directions = np.vstack(dir_list)

    @staticmethod
    def _angular_distance_matrix(dirs: np.ndarray, in_degrees: bool = False) -> np.ndarray:
        # Compute angular distances between all direction pairs
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs_norm = dirs / np.maximum(norms, 1e-8)
        cos_mat = np.clip(dirs_norm @ dirs_norm.T, -1.0, 1.0)
        angles = np.arccos(cos_mat)
        return np.degrees(angles) if in_degrees else angles

    def analyze_optimal_k(self, min_k: int = 3, max_k: int = None) -> int:
        n = len(self.positions)
        max_k = max_k or min(15, max(min_k + 1, n // 2))
        best_score = -np.inf
        best_k = min_k
        X_norm, center, scale = self._normalize_positions(self.positions)
        for k in range(min_k, max_k + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X_norm)
            # Coverage: average spatial spread + angular diversity per cluster
            cov = 0.0
            for c in range(k):
                idxs = np.where(labels == c)[0]
                pts = self.positions[idxs]
                dirs = self.view_directions[idxs]
                if len(idxs) < 1:
                    continue
                spread = float(np.mean(np.std(pts, axis=0))) if len(idxs) > 1 else 0.0
                if len(idxs) > 1:
                    angs = self._angular_distance_matrix(dirs, in_degrees=True)
                    tri_idxs = np.triu_indices(len(idxs), k=1)
                    ang_div = float(np.mean(angs[tri_idxs]))
                else:
                    ang_div = 90.0
                cov += spread + ang_div / 180.0
            cov /= k
            # Compactness: negative inertia normalized by total variance
            compact = -km.inertia_ / (np.linalg.norm(X_norm) + 1e-8)
            score = 0.4 * cov + 0.6 * compact
            if score > best_score:
                best_score = score
                best_k = k
        return best_k

    def select_representative_cameras(
        self,
        min_cameras: int = 3,
        max_cameras: int = None
    ) -> Dict[str, Any]:
        # Determine optimal cluster count
        k = self.analyze_optimal_k(min_k=min_cameras, max_k=max_cameras)
        X_norm, center, scale = self._normalize_positions(self.positions)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_norm)

        selected: List[int] = []
        cluster_info: Dict[int, Any] = {}
        for c in range(k):
            idxs = np.where(labels == c)[0]
            pts = self.positions[idxs]
            dirs = self.view_directions[idxs]
            # Compute cluster world center
            center_norm = km.cluster_centers_[c]
            center_world = center_norm * scale + center
            # Score each camera by proximity + angular uniqueness
            scores: List[float] = []
            for i in idxs:
                dist = np.linalg.norm(self.positions[i] - center_world)
                dist_score = 1.0 / (1.0 + dist)
                
                # Fix: Create a mask to exclude current direction from others
                current_dir_idx = np.where(idxs == i)[0][0]
                other_indices = np.concatenate([
                    np.arange(current_dir_idx), 
                    np.arange(current_dir_idx + 1, len(dirs))
                ])
                
                if len(other_indices) > 0:
                    other_dirs = dirs[other_indices]
                    # Combine current direction with other directions for distance calculation
                    combined_dirs = np.vstack([self.view_directions[i][None, :], other_dirs])
                    angs = self._angular_distance_matrix(combined_dirs, in_degrees=True)
                    # Take distances from first row (current direction) to all others
                    uniq_score = float(np.mean(angs[0, 1:])) / 180.0
                else:
                    uniq_score = 1.0
                    
                scores.append(0.5 * dist_score + 0.5 * uniq_score)
            
            # Pick best-scoring camera
            best_idx = idxs[int(np.argmax(scores))]
            selected.append(best_idx)
            cluster_info[c] = {
                'members': idxs.tolist(),
                'selected': int(best_idx),
                'score': float(np.max(scores))
            }
        return {'selected_indices': selected, 'cluster_info': cluster_info}
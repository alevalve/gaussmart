import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Dict, List
from analyze_cameras import AnalyzeCameras

class CameraClustering:
    """
    Class for clustering camera positions and selecting representative views
    for multi-view processing tasks.
    """
    def __init__(self, analyzer: AnalyzeCameras):
        self.camera_analyzer = analyzer
        self.positions = None
        self.view_directions = None
        self.extract_camera_data()

    def _normalize_positions(self, positions):
        """
        Normalize camera positions by centering and scaling to improve clustering.
        Returns normalized positions and transformation parameters.
        """
        center = np.mean(positions, axis=0)
        centered = positions - center
        scale = np.max(np.abs(centered))
        return centered / scale, center, scale

    def extract_camera_data(self):
        """
        Extract camera positions and view directions from the camera analyzer.
        """
        positions = []
        view_directions = []
        
        for view_num, matrices in self.camera_analyzer.views.items():
            if 'world_mat' in matrices:
                world_mat = matrices['world_mat']
                pos = world_mat[:3, 3]
                direction = world_mat[:3, 2]
                positions.append(pos)
                view_directions.append(direction)
        
        self.positions = np.array(positions)
        self.view_directions = np.array(view_directions)

    def _compute_angle_similarity(self, directions1, directions2):
        """
        Compute angular similarity between camera viewing directions.
        Returns angles in degrees between direction vectors.
        """
        dirs1_norm = directions1 / np.linalg.norm(directions1, axis=1)[:, np.newaxis]
        dirs2_norm = directions2 / np.linalg.norm(directions2, axis=1)[:, np.newaxis]
        
        dot_product = np.clip(np.sum(dirs1_norm * dirs2_norm, axis=1), -1.0, 1.0)
        
        angles = np.arccos(dot_product) * 180 / np.pi
        
        return angles

    def analyze_optimal_k(self, min_k=3, max_k=None):
        """
        Determine the optimal number of clusters using multiple evaluation metrics:
        elbow method, silhouette score, and camera direction variance.
        Returns the optimal k value based on the combined metrics.
        """
        if max_k is None:
            max_k = min(20, len(self.positions) // 2)
            
        max_k = max(min_k + 1, max_k) 
        K = range(min_k, max_k + 1)
        
        scaler = StandardScaler()
        positions_normalized = scaler.fit_transform(self.positions)
        
        distortions = []
        silhouette_scores = []
        direction_variances = []
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(positions_normalized)
            
            distortions.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(positions_normalized, labels))
            
            direction_var = 0
            for i in range(k):
                cluster_dirs = self.view_directions[labels == i]
                if len(cluster_dirs) > 1:
                    angles = self._compute_angle_similarity(cluster_dirs, cluster_dirs)
                    direction_var += np.std(angles)
            direction_variances.append(direction_var / k)
        
        norm_distortions = (distortions - np.min(distortions)) / (np.max(distortions) - np.min(distortions))
        norm_silhouette = (silhouette_scores - np.min(silhouette_scores)) / (np.max(silhouette_scores) - np.min(silhouette_scores))
        norm_dir_var = (direction_variances - np.min(direction_variances)) / (np.max(direction_variances) - np.min(direction_variances))
        
        combined_score = norm_distortions + (1 - norm_silhouette) + norm_dir_var
        
        optimal_k = K[np.argmin(combined_score)]
        print(f"Selected optimal k={optimal_k} based on combined metric analysis")
        
        return optimal_k

    def select_representative_cameras(self, min_cameras=3, max_cameras=None) -> Dict:
        """
        Select representative cameras from each cluster to maximize coverage
        while minimizing redundancy. Returns selected camera indices and
        cluster information for analysis.
        """
        
        optimal_k = self.analyze_optimal_k(min_k=min_cameras, max_k=max_cameras)
        
        scaler = StandardScaler()
        positions_normalized = scaler.fit_transform(self.positions)
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(positions_normalized)
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        selected_cameras = []
        cluster_info = {
            'sizes': {},
            'mean_distances': {},
            'cameras_in_cluster': {},
            'view_direction_stats': {}
        }
        
        for cluster_idx in range(optimal_k):
            cluster_mask = cluster_labels == cluster_idx
            cluster_positions = self.positions[cluster_mask]
            cluster_directions = self.view_directions[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            cluster_info['sizes'][cluster_idx] = len(cluster_indices)
            cluster_info['cameras_in_cluster'][cluster_idx] = cluster_indices.tolist()
            
            center = cluster_centers[cluster_idx]
            distances = np.linalg.norm(cluster_positions - center, axis=1)
            cluster_info['mean_distances'][cluster_idx] = float(np.mean(distances))
            
            if len(cluster_directions) > 1:
                angles = self._compute_angle_similarity(cluster_directions, cluster_directions)
                cluster_info['view_direction_stats'][cluster_idx] = {
                    'mean_angle': float(np.mean(angles)),
                    'std_angle': float(np.std(angles)),
                    'max_angle': float(np.max(angles))
                }
            
            scores = []
            for i, (pos, dir) in enumerate(zip(cluster_positions, cluster_directions)):
                pos_score = np.linalg.norm(pos - center)
                
                dir_scores = self._compute_angle_similarity(
                    dir.reshape(1, -1),
                    cluster_directions
                )
                dir_score = np.mean(dir_scores)
                
                scores.append(pos_score + dir_score)
            
            best_local_idx = np.argmin(scores)
            selected_camera_idx = cluster_indices[best_local_idx]
            selected_cameras.append(selected_camera_idx)
        
        return {
            'selected_indices': selected_cameras,
            'n_selected': len(selected_cameras),
            'cluster_info': cluster_info
        }
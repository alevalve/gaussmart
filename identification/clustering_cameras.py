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

    def normalize_positions(self, positions):
        """
        Normalization using scene center and scale.
        """
        center = positions.mean(axis=0)
        centered = positions - center
        # Use robust scaling based on scene extent
        scale = np.std(centered, axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale) 
        normalized = centered / scale
        return normalized, center, scale
        
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
                direction = -world_mat[:3, 2]
                positions.append(pos)
                view_directions.append(direction)
        
        self.positions = np.array(positions)
        self.view_directions = np.array(view_directions)
    
    def cosine_similarity_matrix(self, dirs):

        """Obtain cosine similarity matrix (N,N) from direction vectors (N,3)"""

        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs_normalized = dirs / np.maximum(norms, 1e-8)
        return np.clip(dirs_normalized @ dirs_normalized.T, -1.0, 1.0)

    def angular_distance_matrix(self, dirs, use_degrees=False):
        """
        (N,N) matrix of angular distances
        """
        cosM = self.cosine_similarity_matrix(dirs)
        angles = np.arccos(np.clip(cosM, -1.0, 1.0))
        if use_degrees:
            angles = np.degrees(angles)
        return angles

    def compute_angle_similarity(self, directions1, directions2):
        """
        Compute angular similarity between camera viewing directions.
        Returns angles in degrees between direction vectors.
        """

        dirs1_norm = directions1 / (np.linalg.norm(directions1, axis=1)[:, np.newaxis] + 1e-8)
        dirs2_norm = directions1 / (np.linalg.norm(directions2, axis=1)[:, np.newaxis] + 1e-8)

        dot_product = np.clip(np.sum(dirs1_norm * dirs2_norm, axis=1), -1.0, 1.0)
        angles = np.arccos(np.abs(dot_product)) * 180 / np.pi

        return angles
    

    def analyze_optimal_k(self, min_k=3, max_k=None):
        """
        Improved method to determine optimal number of clusters.
        Fixed scoring to prioritize scene coverage over clustering metrics.
        """
        if max_k is None:
            max_k = min(15, len(self.positions) // 2)  # More reasonable upper bound
            
        max_k = max(min_k + 1, max_k) 
        K = range(min_k, max_k + 1)
        
        # Use improved normalization
        positions_normalized, _, _ = self.normalize_positions(self.positions)
        
        distortions = []
        silhouette_scores = []
        coverage_scores = []  # New: measure actual scene coverage
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(positions_normalized)
            
            distortions.append(kmeans.inertia_)
            
            if k > 1:
                silhouette_scores.append(silhouette_score(positions_normalized, labels))
            else:
                silhouette_scores.append(0)
            
            # Coverage score: combination of spatial spread and angular diversity
            coverage_score = 0
            for i in range(k):
                cluster_mask = labels == i
                if np.sum(cluster_mask) > 0:
                    cluster_positions = self.positions[cluster_mask]
                    cluster_directions = self.view_directions[cluster_mask]
                    
                    # Spatial spread within cluster
                    if len(cluster_positions) > 1:
                        spatial_spread = np.mean(np.std(cluster_positions, axis=0))
                    else:
                        spatial_spread = 0
                    
                    # Angular diversity within cluster  
                    if len(cluster_directions) > 1:
                        angles = self.angular_distance_matrix(cluster_directions, use_degrees=True)
                        angular_diversity = np.mean(angles[np.triu_indices_from(angles, k=1)])
                    else:
                        angular_diversity = 90  # Single camera gets average score
                    
                    # Combined coverage for this cluster
                    coverage_score += spatial_spread + angular_diversity / 180.0
            
            coverage_scores.append(coverage_score / k)  # Average per cluster
        
        # Normalize all metrics to [0,1]
        norm_distortions = np.array(distortions)
        norm_distortions = 1 - (norm_distortions - norm_distortions.min()) / (norm_distortions.max() - norm_distortions.min() + 1e-8)
        
        norm_silhouette = np.array(silhouette_scores)
        if norm_silhouette.max() > norm_silhouette.min():
            norm_silhouette = (norm_silhouette - norm_silhouette.min()) / (norm_silhouette.max() - norm_silhouette.min())
        
        norm_coverage = np.array(coverage_scores)
        if norm_coverage.max() > norm_coverage.min():
            norm_coverage = (norm_coverage - norm_coverage.min()) / (norm_coverage.max() - norm_coverage.min())
        
        # emphasize coverage and efficiency for 3D scenes
        combined_score = (
            0.3 * norm_distortions +    # Cluster compactness
            0.3 * norm_silhouette +     # Cluster separation  
            0.4 * norm_coverage 
        )
        
        optimal_k = K[np.argmax(combined_score)]
        print(f"Selected optimal k={optimal_k} based on improved scene coverage analysis")
        
        return optimal_k

    def select_representative_cameras(self, min_cameras=3, max_cameras=None) -> Dict:
        """
        Camera selection focusing on scene coverage.
        """
        optimal_k = self.analyze_optimal_k(min_k=min_cameras, max_k=max_cameras)
        
        # Use improved normalization
        positions_normalized, center, scale = self.normalize_positions(self.positions)
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(positions_normalized)
        
        selected_cameras = []
        cluster_info = {
            'sizes': {},
            'mean_distances': {},
            'cameras_in_cluster': {},
            'view_direction_stats': {},
            'coverage_scores': {}  # track coverage per cluster
        }
        
        for cluster_idx in range(optimal_k):
            cluster_mask = cluster_labels == cluster_idx
            cluster_positions = self.positions[cluster_mask]
            cluster_directions = self.view_directions[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            cluster_info['sizes'][cluster_idx] = len(cluster_indices)
            cluster_info['cameras_in_cluster'][cluster_idx] = cluster_indices.tolist()
            
            # Cluster statistics
            cluster_center_norm = kmeans.cluster_centers_[cluster_idx]
            cluster_center_world = cluster_center_norm * scale + center
            distances = np.linalg.norm(cluster_positions - cluster_center_world, axis=1)
            cluster_info['mean_distances'][cluster_idx] = float(np.mean(distances))
            
            # Improved view direction statistics
            if len(cluster_directions) > 1:
                angles = self.compute_angle_similarity(cluster_directions, cluster_directions)
                cluster_info['view_direction_stats'][cluster_idx] = {
                    'mean_angle': float(np.mean(angles)),
                    'std_angle': float(np.std(angles)),
                    'max_angle': float(np.max(angles))
                }
            else:
                cluster_info['view_direction_stats'][cluster_idx] = {
                    'mean_angle': 0.0, 'std_angle': 0.0, 'max_angle': 0.0
                }
            
            # Improved selection within cluster
            if len(cluster_indices) == 1:
                selected_cameras.append(cluster_indices[0])
                cluster_info['coverage_scores'][cluster_idx] = 1.0
            else:
                scores = []
                for i, (pos, direction) in enumerate(zip(cluster_positions, cluster_directions)):
                    # Distance to cluster center (normalized)
                    pos_score = 1.0 / (1.0 + np.linalg.norm(pos - cluster_center_world))
                    
                    # Angular uniqueness within cluster
                    other_dirs = np.delete(cluster_directions, i, axis=0)
                    if len(other_dirs) > 0:
                        angles_to_others = self.compute_angle_similarity(
                            direction.reshape(1, -1), other_dirs
                        )
                        # Higher angle difference is better 
                        angle_score = np.mean(angles_to_others) / 180.0
                    else:
                        angle_score = 1.0
                    
                    # Balanced scoring: position + uniqueness
                    combined_score = 0.5 * pos_score + 0.5 * angle_score
                    scores.append(combined_score)
                
                best_local_idx = np.argmax(scores)
                selected_camera_idx = cluster_indices[best_local_idx]
                selected_cameras.append(selected_camera_idx)
                cluster_info['coverage_scores'][cluster_idx] = float(scores[best_local_idx])
        
        return {
            'selected_indices': selected_cameras,
            'n_selected': len(selected_cameras),
            'cluster_info': cluster_info
        }
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

class HullRemoval:
    def __init__(self, point_cloud, theta=1.96):
        self.point_cloud = point_cloud
        self.theta = theta
    
    def compute_hull_distances(self, points, hull):
        hull_equations = hull.equations
        dot_products = np.dot(points, hull_equations[:, :3].T) + hull_equations[:,3]
        abs_distances = np.abs(dot_products)
        norms = np.linalg.norm(hull_equations[:, :3], axis=1)
        normalized_distances = abs_distances / norms
        return np.min(normalized_distances, axis=1)
    
    def filtering(self, points):
        
        hull = ConvexHull(points)

        distances = self.compute_hull_distances(points, hull)

        mean_distances = np.mean(distances)
        std_distances = np.std(distances)

        z = (distances - mean_distances) / std_distances
        
        filtered_indices = z >= -self.theta
        filtered_points = points[filtered_indices]

        return filtered_points, hull
    
    def forward(self):
        pcd = self.point_cloud
        points = np.asarray(pcd.points)

        # Check for colors and normals to maintain them

        has_colors = len(np.asarray(pcd.colors)) > 0
        has_normals = len(np.asarray(pcd.normals)) > 0

        filtered_points, hull = self.filtering(points)

        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)

        if has_colors or has_normals:

            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            neareast_indices = []

            for point in filtered_points:
                _, idx, _ = pcd_tree.search_knn_vector_3d(point, 1)
                neareast_indices.append(idx[0])
            
            if has_colors:
                colors = np.asarray(pcd.colors)
                filtered_colors = colors[neareast_indices]
                pcd_filtered.colors = o3d.utility.Vector3dVector(filtered_colors)
            
            if has_normals:
                normals = np.asarray(pcd.normals)
                filtered_normals = normals[neareast_indices]
                pcd_filtered.normals = o3d.utility.Vector3dVector(filtered_normals)
            
        
        return filtered_points, hull, pcd_filtered



 




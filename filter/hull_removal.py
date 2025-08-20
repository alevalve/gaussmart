import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

class HullRemoval:
    def __init__(self, point_cloud, theta=1.96):
        self.point_cloud = point_cloud
        self.theta = theta
        
    def compute_hull_distances(self, points, hull):
        hull_equations = hull.equations
        dot_products = np.dot(points, hull_equations[:, :3].T) + hull_equations[:, 3]
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
        return filtered_indices, hull
    
    def forward(self):
        # Convert points to numpy array regardless of point cloud type
        points = np.asarray(self.point_cloud.points)
        
        filtered_indices, hull = self.filtering(points)
        filtered_points = points[filtered_indices]
        
        # Create new point cloud with filtered points
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
        
        # Transfer properties if they exist
        if self.point_cloud.has_colors():
            colors = np.asarray(self.point_cloud.colors)
            pcd_filtered.colors = o3d.utility.Vector3dVector(colors[filtered_indices])
            
        if self.point_cloud.has_normals():
            normals = np.asarray(self.point_cloud.normals)
            pcd_filtered.normals = o3d.utility.Vector3dVector(normals[filtered_indices])
            
        return filtered_points, hull, pcd_filtered


 







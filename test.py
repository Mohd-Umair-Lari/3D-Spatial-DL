import open3d as o3d

# dataset  = o3d.data.PLYPointCloud()
# pcd = o3d.io.read_point_cloud(dataset.path)
pcd = o3d.io.read_point_cloud(".\dataset\synthetic_dataset\scan_023_missing_well.ply")

o3d.visualization.draw_geometries([pcd])

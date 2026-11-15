import open3d as o3d

dataset  = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(dataset.path)

o3d.visualization.draw_geometries([pcd])

import open3d
pcd = open3d.io.read_point_cloud(r"E:\ABB\AI\Depth-Anything-V2\Point_cloud_files\demo26\colored_point_cloud1026.ply")

# pcd = open3d.io.read_point_cloud("E:\ABB\AI\Depth-Anything-V2\metric_depth\output26\color_image_20241026-194401.ply")

# pcd = open3d.io.read_point_cloud("E:\ABB\segment-anything\realsense_yolov9_simulation_detection\SAM26\5ML_sorting_tube_rack_5_point_cloud.npy")

open3d.visualization.draw_geometries([pcd])



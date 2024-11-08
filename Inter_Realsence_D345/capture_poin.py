import pyrealsense2 as rs
import open3d
import cv2
import numpy as np


def generate_point_cloud(color_image, depth_image, intrinsic_matrix):
    height, width = depth_image.shape
    point_cloud = open3d.geometry.PointCloud()

    # 获取相机内参
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    # 将 BGR 转换为 RGB，因为 Open3D 使用 RGB
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    points = []
    colors = []

    # 遍历每个像素
    for v in range(height):
        for u in range(width):
            depth = depth_image[v, u] * 0.001  # 将深度值从毫米转换为米

            if depth > 0:
                # 计算 3D 坐标
                z = depth
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])

                # 获取对应颜色
                color = color_image[v, u] / 255.0  # 归一化到 [0, 1]
                colors.append(color)

    # 将 3D 点和颜色转换为 Open3D 格式
    point_cloud.points = open3d.utility.Vector3dVector(np.array(points))
    point_cloud.colors = open3d.utility.Vector3dVector(np.array(colors))

    return point_cloud


# 初始化 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

# 对齐深度图像到彩色图像
align = rs.align(rs.stream.color)

try:
    while True:
        # 获取帧数据并对齐
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        if not aligned_color_frame or not aligned_depth_frame:
            continue  # 如果没有成功获取到帧，则继续

        # 将帧转换为 NumPy 数组
        color_image = np.asanyarray(aligned_color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        # 显示彩色图像和深度图像（调试用）
        cv2.imshow("Realsense Color Image", color_image)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("Aligned Depth Image", depth_colormap)

        # 按 Enter 键生成点云
        key = cv2.waitKey(1)
        if key == 13:  # Enter 键
            # 打印调试信息，检查深度图像是否有有效值
            print(f"Depth image min: {np.min(depth_image)}, max: {np.max(depth_image)}")

            # 获取相机的内参
            depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            intrinsic_matrix = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                                         [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                                         [0, 0, 1]])

            # 打印相机内参（调试用）
            print(f"Intrinsics: fx={depth_intrinsics.fx}, fy={depth_intrinsics.fy}, ppx={depth_intrinsics.ppx}, ppy={depth_intrinsics.ppy}")

            # 生成带颜色的 3D 点云
            point_cloud = generate_point_cloud(color_image, depth_image, intrinsic_matrix)

            # 保存点云到 PLY 文件
            open3d.io.write_point_cloud("../Point_cloud_files/output_point_cloud.ply", point_cloud)
            print("点云已保存为 output_point_cloud.ply")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()



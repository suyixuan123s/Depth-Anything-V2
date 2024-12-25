"""
Author: Yixuan Su
Date: 2024/11/18 13:56
File: visualization_point_cloud.py

"""

import open3d as o3d


def view_point_cloud(file_path):
    # 加载点云文件
    point_cloud = o3d.io.read_point_cloud(file_path)

    # 检查是否成功加载点云
    if point_cloud.is_empty():
        print("点云文件为空或加载失败。请检查文件路径和格式。")
        return

    # 可视化点云
    o3d.visualization.draw_geometries([point_cloud])


# 用法示例
if __name__ == "__main__":
    # 指定点云文件路径
    file_path = r'E:\ABB\AI\Depth-Anything-V2\suyixuan\Realsense_Point_cloud_Datasets\output_point_cloud01.ply'

    # 查看点云
    view_point_cloud(file_path)

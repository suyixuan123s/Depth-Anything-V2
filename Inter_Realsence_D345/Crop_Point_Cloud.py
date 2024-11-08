import open3d as o3d
import numpy as np


def crop_point_cloud(pcd_path, z_min=0.0, z_max=0.6):
    """
    裁剪点云，只保留桌面和离桌面高60厘米区域的点云，并显示裁剪后的点云。
    :param pcd_path: 点云文件路径
    :param z_min: 桌面高度的最小值（米）
    :param z_max: 离桌面高度的最大值（米）
    """
    # 读取点云文件
    pcd = o3d.io.read_point_cloud(pcd_path)

    # 将点云转换为 NumPy 数组
    pcd_np = np.asarray(pcd.points)

    # 进行高度裁剪，只保留在 z_min 和 z_max 之间的点
    mask = (pcd_np[:, 2] > z_min) & (pcd_np[:, 2] < z_max)
    pcd_cropped_np = pcd_np[mask]

    # 如果点云包含颜色数据，也需要进行裁剪
    if pcd.has_colors():
        pcd_colors = np.asarray(pcd.colors)
        pcd_colors_cropped = pcd_colors[mask]
    else:
        pcd_colors_cropped = None

    # 创建裁剪后的点云对象
    pcd_cropped = o3d.geometry.PointCloud()
    pcd_cropped.points = o3d.utility.Vector3dVector(pcd_cropped_np)

    if pcd_colors_cropped is not None:
        pcd_cropped.colors = o3d.utility.Vector3dVector(pcd_colors_cropped)

    # 显示裁剪后的点云
    o3d.visualization.draw_geometries([pcd_cropped], window_name="Cropped Point Cloud")


# 示例使用
if __name__ == '__main__':
    pcd_path = r'E:\ABB\AI\Depth-Anything-V2\Point_cloud_files\demo24\output_point_cloud1111.ply'
    crop_point_cloud(pcd_path, z_min=0.0, z_max=0.6)

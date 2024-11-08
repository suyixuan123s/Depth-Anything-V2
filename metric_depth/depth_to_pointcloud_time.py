import argparse
import cv2
import glob
import numpy as np
import open3d as o3d
import os
from PIL import Image
import torch
import time  # 引入 time 模块以测量处理时间

from depth_anything_v2.dpt import DepthAnythingV2


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='从图像生成深度图和点云。')
    parser.add_argument('--encoder', default='vitl', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='使用的模型编码器。')
    parser.add_argument('--load-from',
                        default=r'E:\ABB\AI\Depth-Anything-V2\checkpoints\depth_anything_v2_metric_hypersim_vitl.pth',
                        type=str,
                        help='预训练模型权重的路径。')
    parser.add_argument('--max-depth', default=2, type=float,
                        help='深度图的最大深度值。')
    parser.add_argument('--img-path', type=str,
                        default=r'E:\ABB\AI\Depth-Anything-V2\Inter_Realsence_D345_Datasets\color_image_20241026-194402.jpg',
                        help='输入图像或包含图像的目录路径。')
    parser.add_argument('--outdir', type=str, default=r'E:\ABB\AI\Depth-Anything-V2\metric_depth\output01',
                        help='保存输出点云的目录。')
    parser.add_argument('--focal-length-x', default=909.939697265625, type=float,
                        help='x轴的焦距。')
    parser.add_argument('--focal-length-y', default=909.9850463867188, type=float,
                        help='y轴的焦距。')

    args = parser.parse_args()

    # 确定使用的设备（CUDA, MPS 或 CPU）
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # 根据选择的编码器配置模型
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # 使用指定配置初始化 DepthAnythingV2 模型
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # 获取要处理的图像文件列表
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    # 如果输出目录不存在，则创建它
    os.makedirs(args.outdir, exist_ok=True)

    # 处理每个图像文件
    for k, filename in enumerate(filenames):
        print(f'正在处理 {k + 1}/{len(filenames)}: {filename}')

        # 开始计时
        start_time = time.time()

        # 加载图像
        color_image = Image.open(filename).convert('RGB')
        width, height = color_image.size

        # 使用 OpenCV 读取图像
        image = cv2.imread(filename)
        pred = depth_anything.infer_image(image, height)

        # 调整深度预测的大小以匹配原始图像的尺寸
        resized_pred = Image.fromarray(pred).resize((width, height), Image.NEAREST)

        # 生成网格并计算点云坐标
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / args.focal_length_x
        y = (y - height / 2) / args.focal_length_y
        z = np.array(resized_pred)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = np.array(color_image).reshape(-1, 3) / 255.0

        # 创建点云并将其保存到输出目录
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + ".ply"),
                                 pcd)

        # 结束计时并打印每张图片的处理时间
        end_time = time.time()
        print(f'{filename} 处理时间: {end_time - start_time:.2f} 秒')


if __name__ == '__main__':
    main()


'''
E:\ABB\AI\Depth-Anything-V2\venv\Scripts\python.exe E:\ABB\AI\Depth-Anything-V2\metric_depth\depth_to_pointcloud_time.py 
xFormers not available
xFormers not available
正在处理 1/1: E:\ABB\AI\Depth-Anything-V2\Inter_Realsence_D345_Datasets\color_image_20241026-194402.jpg
E:\ABB\AI\Depth-Anything-V2\Inter_Realsence_D345_Datasets\color_image_20241026-194402.jpg 处理时间: 6.86 秒

Process finished with exit code 0


'''
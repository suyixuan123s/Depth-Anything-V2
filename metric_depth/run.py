'''
DPT（Depth Prediction Transformer）
模型本身不直接使用深度相机的数据进行训练或估计。
它是一个通过计算机视觉技术从RGB图像中估计深度的模型，
主要依赖于视觉信息，
而不是深度相机提供的物理深度数据。

1. DPT的工作原理
DPT通过深度学习模型从2D RGB图像中推断出场景的深度图。
它不依赖于深度相机的原始深度数据，
而是通过Transformer架构从图像中的像素和上下文信息中提取全局和局部特征，
并预测每个像素的深度值。具体而言，
DPT模型利用自注意力机制来捕捉远距离像素之间的依赖关系，
能够更好地处理复杂的场景和几何结构。
'''

import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import time
from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')

    parser.add_argument('--img-path', type=str, default=r'E:\ABB\AI\Depth-Anything-V2\Inter_Realsence_D345_Datasets\color_image_20241026-194402.jpg')
    parser.add_argument('--input-size', type=int, default=320)
    parser.add_argument('--outdir', type=str, default=r'E:\ABB\AI\Depth-Anything-V2\metric_depth\output1')

    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default=r'E:\ABB\AI\Depth-Anything-V2\checkpoints\depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=2)

    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))

    depth_anything = depth_anything.to(DEVICE).eval()

    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral')

    for k, filename in enumerate(filenames):
        print(f'Progress {k + 1}/{len(filenames)}: {filename}')

        # 开始计时
        start_time = time.time()


        raw_image = cv2.imread(filename)

        depth = depth_anything.infer_image(raw_image, args.input_size)

        if args.save_numpy:
            output_path = os.path.join(args.outdir,
                                       os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
            np.save(output_path, depth)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
        if args.pred_only:
            cv2.imwrite(output_path, depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])

            cv2.imwrite(output_path, combined_result)

            # 结束计时并打印每张图片的处理时间
            end_time = time.time()
            print(f'{filename} 处理时间: {end_time - start_time:.2f} 秒')


#  python run.py --encoder vitl --img-path E:\ABB\AI\yolov9\data\data_realsense\color_image_20241016-215422.jpg --outdir E:\ABB\AI\Depth-Anything-V2\metric_depth\output --input-size 480

'''
E:\ABB\AI\Depth-Anything-V2\venv\Scripts\python.exe E:\ABB\AI\Depth-Anything-V2\metric_depth\run.py 
xFormers not available
xFormers not available
Progress 1/1: E:\ABB\AI\Depth-Anything-V2\Inter_Realsence_D345_Datasets\color_image_20241026-194402.jpg
E:\ABB\AI\Depth-Anything-V2\Inter_Realsence_D345_Datasets\color_image_20241026-194402.jpg 处理时间: 4.54 秒

Process finished with exit code 0
'''

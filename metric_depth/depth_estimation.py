import os
import cv2
import torch
from depth_anything_v2.dpt import DepthAnythingV2

# 模型配置字典
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

# 选择编码器和数据集
encoder = 'vitl'  # 或 'vits', 'vitb'
dataset = 'hypersim'  # 'hypersim' 用于室内模型，'vkitti' 用于室外模型
max_depth = 20  # 室内模型的最大深度为20米，室外模型可为80米

# 模型权重文件路径
weights_path = r'E:\ABB\AI\Depth-Anything-V2\checkpoints\depth_anything_v2_metric_hypersim_vitl.pth'

# 检查路径是否存在
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"模型权重文件未找到: {weights_path}")
else:
    print(f"模型权重文件已找到: {weights_path}")

# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 初始化并加载模型
try:
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)  # 将模型移动到指定设备（GPU或CPU）
    model.eval()
    print("模型加载成功！")
except Exception as e:
    print(f"加载模型时出错: {e}")

# 图像文件路径
image_path = r'E:\ABB\AI\Depth-Anything-V2\Inter_Realsence_D345_Datasets\color_image_20241018-105337.jpg'

# 检查图像路径是否存在
if not os.path.exists(image_path):
    raise FileNotFoundError(f"图像文件未找到: {image_path}")
else:
    print(f"图像文件已找到: {image_path}")

# 读取图像并进行深度推断
try:
    # 使用OpenCV读取图像
    raw_img = cv2.imread(image_path)

    # 检查图像是否加载成功
    if raw_img is None:
        raise ValueError(f"无法加载图像: {image_path}")

    print(f"成功加载图像，图像尺寸为: {raw_img.shape}")

    # 转换为RGB格式（OpenCV读取的是BGR格式，需要转换为RGB）
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    print("图像成功转换为RGB格式")

    # 转换图像为torch张量，并将其移动到与模型相同的设备
    raw_img_tensor = torch.from_numpy(raw_img).float().permute(2, 0, 1).unsqueeze(0).to(device)
    print(f"图像张量的形状为: {raw_img_tensor.shape}")

    # 执行深度推断
    depth = model.infer_image(raw_img_tensor)
    print("深度估计成功！")
    print(f"深度图的形状: {depth.shape}")
except Exception as e:
    print(f"推断深度时出错: {e}")



# python run.py \  --encoder vitl \  --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \  --max-depth 20 \  --img-path E:\ABB\AI\yolov9\data\data_realsense\color_image_20241016-215422.jpg --outdir E:\ABB\AI\Depth-Anything-V2\metric_depth\output [--input-size (480, 640, 3)] [--save-numpy]
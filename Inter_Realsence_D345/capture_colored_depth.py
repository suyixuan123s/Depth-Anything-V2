import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

# 定义保存图像的目录
save_directory = r'E:\ABB\AI\Depth-Anything-V2\Inter_Realsence_D345_Datasets'
os.makedirs(save_directory, exist_ok=True)

# 初始化RealSense管道
pipeline = rs.pipeline()
config = rs.config()

# 配置深度相机流
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# 启动管道
pipeline.start(config)

try:
    while True:
        # 获取一帧深度数据
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # 将深度帧转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())

        # 过滤掉不在范围内的深度值（例如 1.28 米至 2 米之间）
        depth_image[(depth_image < 1280) | (depth_image > 2000)] = 0

        # 将裁剪后的深度数据归一化到 0-255
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 使用多颜色映射模式进行伪彩色显示
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # 显示带多颜色信息的深度图像
        cv2.imshow('Colored Depth Image', depth_colored)

        # 按 Enter 键保存彩色深度图像
        key = cv2.waitKey(1)
        if key == 13:  # Enter key
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            depth_image_path = os.path.join(save_directory, f'depth_image_colored_{timestamp}.png')
            cv2.imwrite(depth_image_path, depth_colored)
            print(f'Saved colored depth image as {depth_image_path}')

        # 按 'q' 键退出
        if key & 0xFF == ord('q'):
            break
finally:
    # 停止管道
    pipeline.stop()
    cv2.destroyAllWindows()

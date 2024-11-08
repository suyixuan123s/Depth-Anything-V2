import os
import cv2
import numpy as np
import pyrealsense2 as rs

def capture_chessboard_images(output_dir, num_images=1):
    """
    使用 RealSense 相机拍摄多张棋盘格图像并保存。
    :param output_dir: 图像保存的目录
    :param num_images: 拍摄的图像数量
    """
    # 配置 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # 启动管道
    pipeline.start(config)

    # 创建对齐对象，将深度帧对齐到彩色帧
    align_to = rs.stream.color
    align = rs.align(align_to)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        captured_images = 0
        while captured_images < num_images:
            # 获取帧
            frames = pipeline.wait_for_frames()

            # 对齐深度帧到彩色帧
            aligned_frames = align.process(frames)

            # 获取对齐后的彩色帧
            color_frame = aligned_frames.get_color_frame()

            if not color_frame:
                continue

            # 将彩色帧转换为 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())

            # 显示彩色图像
            cv2.imshow('Captured Image', color_image)
            print("按下 Enter 键拍摄照片，按下 ESC 退出")
            key = cv2.waitKey(1)

            if key == 27:  # 按下 ESC 键退出
                break
            elif key == 13:  # 按下 Enter 键拍摄图像
                # 保存彩色图像
                image_path = os.path.join(output_dir, f"chessboard_image_{captured_images}.jpg")
                cv2.imwrite(image_path, color_image)
                print(f"保存图像: {image_path}")
                captured_images += 1

    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()

# 示例使用
if __name__ == '__main__':
    output_directory = r'E:\ABB-Project\cc-wrs\ABB_Intel_Realsense\Dataset2'
    capture_chessboard_images(output_directory, num_images=10)
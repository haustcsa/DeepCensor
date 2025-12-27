import torch
import cv2
import numpy as np
from config import training_config as cfg
import matplotlib.pyplot as plt
from pytorch_wavelets import DTCWTForward, DTCWTInverse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def images_U_dtcwt_with_low(images_U):
    xfm = DTCWTForward(J=2, biort='near_sym_b', qshift='qshift_b').to(cfg.device)
    low_pass, high_pass = xfm(images_U)
    return low_pass, high_pass

def dtcwt_images_U(low_pass, high_pass):
    ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').to(cfg.device)
    return ifm((low_pass, high_pass))

def images_U_dtcwt_without_low(images_U):
    xfm = DTCWTForward(J=2, biort='near_sym_b', qshift='qshift_b').to(cfg.device)
    _, high_pass = xfm(images_U)
    return high_pass


if __name__ == '__main__':
    indices_encoder = torch.tensor([0, 2]).to(cfg.device)

    # 1. 读取图片
    image_path = 'E:/scpcode/processed_dataset/train_128/00003.jpg'  # 替换为你的图片路径
    img_bgr = cv2.imread(image_path)  # OpenCV 读取为 BGR 格式
    if img_bgr is None:
        raise FileNotFoundError(f"无法加载图片：{image_path}")

    # 2. 调整图片尺寸（确保为 2 的倍数）
    img_bgr = cv2.resize(img_bgr, (256, 256))  # 固定为 256x256

    # 3. 将 BGR 转换为 RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 4. 将 RGB 转换为 YUV
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)

    # 5. 提取 U 通道并转换为张量
    U = img_yuv[:, :, 1]  # 蓝色色度通道，形状 (256, 256)
    U = (U / 127.5) - 1.0  # 归一化到 [-1, 1]
    U_tensor = torch.tensor(U, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # 形状 [1, 1, 256, 256]

    # 6. 应用 DTCWT 分解
    l, h = images_U_dtcwt_with_low(U_tensor)
    selected_areas_embed = torch.index_select(h[1], 2, indices_encoder)[:, :, :, :, :, 0].squeeze(1)

    # 7. 打印形状
    print(f"Low-pass subband (l) shape: {l.shape}")
    print(f"High-pass subband (h[0], Level 1) shape: {h[0].shape}")
    print(f"High-pass subband (h[1], Level 2) shape: {h[1].shape}")
    print(f"Selected areas for embedding shape: {selected_areas_embed.shape}")

    # 8. 可视化低频分量 (l)
    plt.figure(figsize=(5, 5))
    l_img = l[0, 0, :, :].cpu().numpy()
    l_img = (l_img - l_img.min()) / (l_img.max() - l_img.min() + 1e-10)  # 归一化
    plt.imshow(l_img, cmap='gray')
    plt.title(f'Low-pass (l)\nShape: {l.shape}')
    plt.axis('off')
    plt.show()

    # 9. 可视化第一层高频分量 (h[0])
    plt.figure(figsize=(15, 5))
    for i in range(6):
        subband = h[0][0, 0, i, :, :, 0].cpu().numpy()  # 取实部
        subband = np.abs(subband)  # 取绝对值以便可视化
        subband = (subband - subband.min()) / (subband.max() - subband.min() + 1e-10)  # 归一化
        plt.subplot(1, 6, i + 1)
        plt.title(f'h[0] Subband {i}\n(±{(15 + 30 * (i // 2))}°)')
        plt.imshow(subband, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 10. 可视化第二层高频分量 (h[1])
    plt.figure(figsize=(15, 5))
    for i in range(6):
        subband = h[0][0, 0, i, :, :, 0].cpu().numpy()  # 取实部
        subband = np.abs(subband)
        subband = (subband - subband.min()) / (subband.max() - subband.min() + 1e-10)
        plt.subplot(1, 6, i + 1)
        plt.title(f'h[1] Subband {i}\n(±{(15 + 30 * (i // 2))}°)')
        plt.imshow(subband, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
# if __name__ == '__main__':
#     indices_encoder = torch.tensor([0, 2]).to(cfg.device)
#     # 1. 读取图片
#     image_path = 'C:/dataset/NeuralTextures/000_003_1.jpg'  # 替换为你的图片路径
#     img_bgr = cv2.imread(image_path)  # OpenCV 读取为 BGR 格式
#     if img_bgr is None:
#         raise FileNotFoundError(f"无法加载图片：{image_path}")
#
#     # 2. 调整图片尺寸（确保为 2 的倍数）
#     img_bgr = cv2.resize(img_bgr, (256, 256))  # 固定为 256x256
#
#     # 3. 将 BGR 转换为 RGB
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#
#     # 4. 将 RGB 转换为 YUV
#     img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
#
#     # 5. 提取 U 通道并转换为张量
#     U = img_yuv[:, :, 1]  # 蓝色色度通道，形状 (256, 256)
#     U = (U / 127.5) - 1.0  # 归一化到 [-1, 1]
#     U_tensor = torch.tensor(U, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # 形状 [1, 1, 256, 256]
#
#     # 6. 应用 DTCWT 分解
#     l, h = images_U_dtcwt_with_low(U_tensor)
#     selected_areas_embed = torch.index_select(h[1], 2, indices_encoder)[:, :, :, :, :, 0].squeeze(1)
#
#     # 7. 打印形状
#     print(f"Low-pass subband shape: {l.shape}")
#     print(f"High-pass subband (Level 1) shape: {h[1].shape}")
#     print(f"High-pass subband (Level 1) shape1111: {selected_areas_embed.shape}")
#
#     # 8. 可视化第一层高通子带的 6 个方向子带
#     plt.figure(figsize=(15, 5))
#     for i in range(6):
#         subband = h[0][0, 0, :, :, i, 0].cpu().numpy()  # 取实部
#         subband = np.abs(subband)  # 取绝对值以便可视化
#         subband = (subband - subband.min()) / (subband.max() - subband.min() + 1e-10) * 255  # 归一化到 [0, 255]
#         plt.subplot(1, 6, i + 1)
#         plt.title(f'Subband {i} (±{(15 + 30 * (i // 2))}°)')
#         plt.imshow(subband, cmap='gray')
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()

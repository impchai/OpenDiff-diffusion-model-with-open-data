from PIL import Image
import numpy as np
import os
# 定义图像数据集的路径
dataset_path = "./datasets/e_shanghai/"

# 初始化累积和和样本数
mean_accumulator = np.zeros(3)
std_accumulator = np.zeros(3)
sample_count = 0
from tqdm import tqdm
# 遍历数据集
image_files = [os.path.join(root, filename) for root, _, files in os.walk(dataset_path) for filename in files
               if filename.endswith(('.jpg', '.jpeg', '.png'))]

# 使用tqdm创建进度条
for image_file in tqdm(image_files, desc="Processing Images", unit="image"):
    # 打开图像文件并将其转换为NumPy数组
    image = np.array(Image.open(image_file))
    image_normalized = image / 255.0
    # 计算每个通道的累积和
    mean_accumulator += image_normalized.mean(axis=(0, 1))
    std_accumulator += image_normalized.std(axis=(0, 1))

    # 增加样本数
    sample_count += 1

# 计算平均值和标准差
mean = mean_accumulator / sample_count
std = std_accumulator / sample_count

print("Mean:", mean)
print("Std:", std)
import os
from PIL import Image
import numpy as np

# 提取文件的路径
output_directory = os.path.dirname('../testA_1.npy')
# 提取文件名
output_name = os.path.splitext(os.path.basename("testA1.npy"))[0]
# 提取 npy 文件中的数组
arr = np.load('../testA_1.npy')

# 确保数组数据类型为 uint8
arr = (arr / arr.max() * 255).astype(np.uint8)

# 将 NumPy 数组转换为 Pillow Image 对象
img = Image.fromarray(arr)

# 指定新的尺寸，如果需要调整的话
new_size = (512, 512)  # 这里和原始尺寸相同，所以实际上不需要调整

# 使用 Pillow 调整图像尺寸（如果需要的话）
# 注意：这里的 resize 方法是可选的，因为新尺寸和原始尺寸相同
# img_resized = img.resize(new_size, Image.ANTIALIAS)

# 保存图片
img.save(os.path.join(output_directory, "{}_disp.png".format(output_name)))
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 设置图像路径
image_path = '../testA1_disp.png'  # 替换为你的图像文件路径

# 加载图像
img = Image.open(image_path)

# 确保图像是3层的RGB图像
if img.mode != 'RGB':
    img = img.convert('RGB')

# 获取图像尺寸
width, height = img.size

# 计算每个正方形的尺寸
square_size = width // 8  # 512 // 8 = 64

# 初始化一个列表来存储所有的正方形
squares = []

# 切分图像为32个正方形
for i in range(4):  # 4行
    for j in range(8):  # 8列
        x = j * square_size
        y = i * square_size
        square = img.crop((x, y, x + square_size, y + square_size))
        squares.append(square)

# 显示图像
fig, axes = plt.subplots(4, 8, figsize=(64, 32))  # 4行8列

# 遍历正方形列表并显示每个正方形
for i, ax in enumerate(axes.flat):
    if i < len(squares):
        square = squares[i]
        # 将PIL图像转换为numpy数组，以便matplotlib可以显示
        img_array = np.array(square)
        ax.imshow(img_array)
        ax.axis('off')  # 关闭坐标轴
    else:
        ax.axis('off')  # 如果没有足够的正方形，关闭剩余的画布

plt.tight_layout()
plt.show()
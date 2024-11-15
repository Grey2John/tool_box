import numpy as np
import matplotlib.pyplot as plt
import math

camera_params = np.array([907.3834838867188, 907.2291870117188, 644.119384765625, 378.90472412109375])
K = np.array([
    [camera_params[0], 0, camera_params[2]],
    [0, camera_params[1], camera_params[3]],
    [0, 0, 1]
])
K1 = np.linalg.inv(K)
H = 2.5
theta = 23
theta = math.radians(theta)


def fill_depth(u, v):
    matrix = np.zeros((u, v))  # 720, 1280

    # 使用公式填充矩阵
    for i in range(u):
        for j in range(v):
            Lc = np.dot(K1, np.array([j, i, 1]))
            t = H / (math.sin(theta) + Lc[1] * math.cos(theta))
            d = np.linalg.norm(Lc*t)
            matrix[i, j] = d
    cropped_image = matrix[:, 160:1120]
    return cropped_image


m = fill_depth(720, 1280)
plt.matshow(m, cmap='viridis')
plt.colorbar()  # 添加颜色条
plt.title("Matrix Visualization")
plt.show()
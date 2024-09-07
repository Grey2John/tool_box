import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from itertools import combinations
import math


class GridUnit:
    def __init__(self, row, col, size=8, downsample_pixel=1):
        """the grid in image, for indexing the points"""
        self.row = row
        self.col = col
        # self.max_depth = 0
        # self.min_depth = 0
        self.point_state_flag = [0, 0, 0, 0, 0, 0, 0]
        # self.class_count = [0, 0, 0, 0, 0, 0, 0]  # 观测类计数

        self.point_index = []  # the index of the points in this grid
        self.point_xyzdepth_c = []  # xyz_cam relate to point
        self.point_obs_state = []  # observing state
        self.point_3state = []  # direct state

        self.upsampling_map = PixelMapInGrid(size, downsample_pixel)

    def adjust_grid(self, point_index, point_obs_state, label_state, h_w, xyzdepth_c):
        """adjust each point projection"""
        self.point_index.append(point_index)
        self.point_xyzdepth_c.append(xyzdepth_c)  # xyz based on camera
        self.point_obs_state.append(point_obs_state)
        self.point_3state.append(label_state)
        self.point_state_flag[point_obs_state] = 1
        # self.class_count[point_obs_state] += 1

        self.upsampling_map.adjust_map(h_w, xyzdepth_c)


class PixelMapInGrid:
    def __init__(self, size=8, downsample_pixel=1):
        """8*8 size is the sparse map"""
        self.size = size
        self.downsample_pixel = downsample_pixel
        self.local_pixel_list = []  # pixel info in this grid [[hw, xyz, depth_min, depth_max]
        # self.max_depth_matrix = np.zeros([size, size])
        # self.min_depth_matrix = np.zeros([size, size])
        self.exist_map = np.zeros([size, size], dtype=int)
        self.index_map = np.zeros([size, size], dtype=int)
        self.num = 0  # 存在的像素数量

    def adjust_map(self, h_w, depth_c):
        h = h_w[0] % self.size  # local h
        w = h_w[1] % self.size

        if not self.exist_map[h, w]:
            self.local_pixel_list.append([h, w]+depth_c+[depth_c[-1]])  # [hw, xyz, depth_min, depth_max] 要无重复
            self.exist_map[h, w] = 1
            self.index_map[h, w] = self.num + 1
            self.num += 1
        else:
            if depth_c[-1] > self.local_pixel_list[self.index_map[h, w]-1][-1]:
                self.local_pixel_list[self.index_map[h, w] - 1][-1] = depth_c[-1]
            if depth_c[-1] < self.local_pixel_list[self.index_map[h, w] - 1][-2]:
                self.local_pixel_list[self.index_map[h, w] - 1][-2] = depth_c[-1]
        # if depth_c[-1] > self.max_depth_matrix[h, w]:
        #     self.max_depth_matrix[h, w] = depth_c[-1]
        #
        # if depth_c[-1] < self.min_depth_matrix[h, w] or self.min_depth_matrix[h, w] == 0:
        #     self.min_depth_matrix[h, w] = depth_c[-1]

    # def one_pixel_grad_check(self, thread_hold):
    #     """单个像素内，多个点的梯度超过阈值"""
    #     delta_depth = np.abs(self.max_depth_matrix - self.min_depth_matrix)/self.downsample_pixel
    #     if np.any(delta_depth > thread_hold):
    #         return True
    #     else:
    #         return False


def delaunay_crack_detect_old(depth_matrix, count, thread_hold=1, save_path=None):
    """go through all the edge, find the grad > thread_hold
    输入对象：[x,y,D] 的list进行两两做梯度
    """
    non_zero_indices = list(zip(*np.nonzero(depth_matrix)))
    non_zero_indices_np = np.array(non_zero_indices)  # 取整点坐标
    n = len(non_zero_indices_np)
    if 1 < n <= 10:
        for i in range(n-1):
            for j in range(i + 1, n):
                D = distance(non_zero_indices_np[i], non_zero_indices_np[j])
                grad = abs(depth_matrix[non_zero_indices[i]] - depth_matrix[non_zero_indices[j]]) / D
                if grad >= thread_hold:
                    return True
                else:
                    return False  # 有问题
    elif n < 2:
        return False

    if is_column_equal(non_zero_indices_np):
        return False

    triangulation = Delaunay(non_zero_indices_np)
    sorted_simplices = np.sort(triangulation.simplices, axis=1)
    all_combinations = []
    for matrix in sorted_simplices:
        matrix_combinations = list(combinations(matrix, 2))
        for m in matrix_combinations:
            if m not in all_combinations:
                all_combinations.append(m)
                D = distance(non_zero_indices_np[m[0]], non_zero_indices_np[m[1]])
                grad = abs(depth_matrix[non_zero_indices[m[0]]] - depth_matrix[non_zero_indices[m[1]]]) / D
                if grad >= thread_hold:
                    # print(element)
                    delaunay_visual(triangulation, non_zero_indices_np, count+1, save_path)
                    return True
    return False


def delaunay_crack_angle_detect(local_map_np, count, thread_hold=0.98, save_path=None):
    """
    go through all the edge, find the grad > thread_hold
    local_map_np输入对象：[hw, xyz, depth] 的numpy进行两两做梯度
    """
    n = local_map_np.shape[0]  # pixel number
    if 1 < n <= 6:
        for i in range(n-1):
            for j in range(i + 1, n):
                if local_map_np[i, -1] > local_map_np[j, -1]:
                    angle = angle_between_points(local_map_np[i, 2:5], local_map_np[j, 2:5])
                else:
                    angle = angle_between_points(local_map_np[j, 2:5], local_map_np[i, 2:5])
                if angle >= thread_hold:
                    return True
        return False  # 没有找到
    elif n < 2:
        return False

    if is_column_equal(local_map_np):
        return False  # 连成线
    # Delaunay  整合三角形连线并算法梯度
    triangulation = Delaunay(local_map_np[:, :2])
    sorted_simplices = np.sort(triangulation.simplices, axis=1)  # 三个点组成的三角形 [[点序号]，[]]
    all_line_list = []  # 防止重复
    for tri_index in sorted_simplices:
        two_p_list = list(combinations(tri_index, 2))  # [0, 3, 6] 变 [(0, 3), (0, 6), (3, 6)]
        for two_p in two_p_list:  # (0, 3)
            if two_p not in all_line_list:
                all_line_list.append(two_p)
                if local_map_np[two_p[0], -1] > local_map_np[two_p[1], -1]:
                    angle = angle_between_points(local_map_np[two_p[0], 2:5],
                                                 local_map_np[two_p[1], 2:5])
                else:
                    angle = angle_between_points(local_map_np[two_p[1], 2:5],
                                                 local_map_np[two_p[0], 2:5])
                if angle >= thread_hold:
                    delaunay_visual(triangulation, local_map_np, count+1, save_path)
                    return True
    return False


def delaunay_crack_detect(local_map_np, count, location, scale=8, coe=1, save_path=None):
    """
    三种类型：单像素超过，少量点连线，多点delaunay连线
    local_map_np输入对象：[hw(local), xyz, depth_min, depth_max] 的numpy进行两两做梯度
    """
    n = local_map_np.shape[0]  # pixel number
    line_list = []
    if_break = False
    if is_column_equal(local_map_np):
        return False  # 连成线

    """单像素检索"""
    base_pixel = [i*scale for i in location]
    for p in local_map_np:
        d = m_standard[int(p[0]+base_pixel[0]+1)][int(p[1]+base_pixel[1])] - \
            m_standard[int(p[0]+base_pixel[0])][int(p[1]+base_pixel[1])]
        if p[-1] - p[-2] > coe*abs(d):
            return True
    """多像素检索"""
    if n < 2:
        return False
    elif 1 < n <= 6:
        for i in range(n-1):
            for j in range(i + 1, n):
                line_list.append([i, j])
    else:
        """Delaunay  整合三角形连线并算法梯度"""
        triangulation = Delaunay(local_map_np[:, :2])
        sorted_simplices = np.sort(triangulation.simplices, axis=1)  # 三个点组成的三角形 [[点序号]，[]]
        for tri_index in sorted_simplices:
            line_list += list(combinations(tri_index, 2))  # [0, 3, 6] 变 [(0, 3), (0, 6), (3, 6)]
    """ 计算差值 """
    all_line_list = []  # 防止重复
    for two_p in line_list:  # (0, 3)
        if two_p not in all_line_list:
            all_line_list.append(two_p)
            p1 = local_map_np[two_p[0]]
            p2 = local_map_np[two_p[1]]
            d = m_standard[int(p1[0]+base_pixel[0])][int(p1[1]+base_pixel[1])] - \
                m_standard[int(p2[0]+base_pixel[0])][int(p2[1]+base_pixel[1])]
            if abs(p1[-1]-p2[-1]) > coe*abs(d):
                if_break = True
    # if if_break:
    #     delaunay_visual(triangulation, local_map_np, count+1, save_path)
    return if_break


def angle_between_points(OA, OB):
    # 向量 A是远点，输入为numpy [1,2,3]
    dot_product = np.dot(OA, OB)
    # 计算模长
    norm_OA = np.linalg.norm(OA)
    norm_OB = np.linalg.norm(OB)
    # 计算夹角的余弦值
    cos_angle = dot_product / (norm_OA * norm_OB)
    # 防止数值误差导致的超出范围
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    # 计算夹角（弧度）
    # angle = np.arccos(cos_angle)
    # angle = np.degrees(np.arccos(cos_angle))  # 转换为角度
    return cos_angle


def delaunay_visual(triangulation, points, count, save_path=None):
    # max_size =
    plt.plot(points[:, 0], points[:, 1], 'o', markersize=12, label='Point in Pixel')

    # Plot the Delaunay triangulation
    plt.triplot(points[:, 0], points[:, 1], triangulation.simplices, linewidth=1, color='red')

    # Label the triangles
    for j, p in enumerate(points):
        plt.text(p[0], p[1], f'{j}', ha='left', va='top', fontsize=18)

    plt.grid(True, color='gray', linestyle='--')
    plt.xlabel('X-axis', fontsize=18)
    plt.ylabel('Y-axis', fontsize=18)
    plt.xticks(range(0, 25, 8), fontsize=18)
    plt.yticks(range(0, 25, 8), fontsize=18)
    plt.xlim(-1, 25)
    plt.ylim(-1, 25)
    plt.gca().invert_yaxis()
    plt.title('Delaunay Triangulation', fontsize=20)

    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, "{}.svg".format(count)), format='svg')
    plt.show()
    plt.clf()


def grad_cal(point_list, two_p_index):
    D2F = distance(point_list[two_p_index[0]][0:2], point_list[two_p_index[1]][0:2])
    grad = abs(point_list[two_p_index[0]][2] - point_list[two_p_index[1]][2]) / D2F
    return grad


def distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def is_column_equal(matrix):
    rows = len(matrix)

    col1_same = True
    col2_same = True
    for i in range(1, rows):
        if matrix[i, 0] != matrix[0, 0]:
            col1_same = False
            break
    for i in range(1, rows):
        if matrix[i, 1] != matrix[0, 1]:
            col2_same = False
            break
    return (col1_same or col2_same)


def fill_depth(u, v):
    matrix = np.zeros((u, v))  # 720, 1280
    H = 2.5
    theta = 23
    camera_params = np.array([907.3834838867188, 907.2291870117188, 644.119384765625, 378.90472412109375])
    K = np.array([
        [camera_params[0], 0, camera_params[2]],
        [0, camera_params[1], camera_params[3]],
        [0, 0, 1]
    ])
    # 使用公式填充矩阵
    for i in range(u):
        for j in range(v):
            Lc = np.dot(K, np.array([j, i, 1]))
            t = H / (math.sin(theta) + Lc[1] * math.cos(theta))
            d = np.linalg.norm(Lc*t)
            matrix[i, j] = d
    cropped_image = matrix[:, 160:1120]
    return cropped_image


m_standard = fill_depth(720, 1280)
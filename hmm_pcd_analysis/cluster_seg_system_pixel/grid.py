import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from itertools import combinations


class GridUnit:
    def __init__(self, row, col, size=4):
        """the grid in image, for indexing the points"""
        self.row = row
        self.col = col
        # self.max_depth = 0
        # self.min_depth = 0
        self.point_state_flag = [0, 0, 0, 0, 0, 0, 0]
        self.class_count = [0, 0, 0, 0, 0, 0, 0]

        self.point_index = []  # the index of the points in this grid
        self.point_xydepth = []  # xyz_cam relate to point
        self.point_obs_state = []  # observing state
        self.point_3state = []  # direct state

        self.upsampling_map = PixelMapInGrid(size)

    def adjust_grid(self, point_index, point_obs_state, label_state, h_w, xydepth_c):
        """adjust each point projection"""
        self.point_index.append(point_index)
        self.point_xydepth.append(xydepth_c)  # xyz based on camera
        self.point_obs_state.append(point_obs_state)
        self.point_3state.append(label_state)
        self.point_state_flag[point_obs_state] = 1
        self.class_count[point_obs_state] += 1

        self.upsampling_map.adjust_map(h_w, xydepth_c[2])


class PixelMapInGrid:
    def __init__(self, size=4):
        """3*size is the sparse map"""
        self.size = size
        self.max_depth_matrix = np.zeros([size, size])
        self.min_depth_matrix = np.zeros([size, size])
        self.exist_map = np.zeros([size, size], dtype=int)

    def adjust_map(self, h_w, depth_c):
        h = h_w[0] % self.size
        w = h_w[1] % self.size
        self.exist_map[h, w] = 1
        if depth_c > self.max_depth_matrix[h, w]:
            self.max_depth_matrix[h, w] = depth_c
        if depth_c < self.min_depth_matrix[h, w] or self.min_depth_matrix[h, w] == 0:
            self.min_depth_matrix[h, w] = depth_c

    def one_pixel_grad_check(self, thread_hold):
        delta_depth = np.abs(self.max_depth_matrix - self.min_depth_matrix)/1
        if np.any(delta_depth > thread_hold):
            return True
        else:
            return False


def delaunay_crack_detect(depth_matrix, count, thread_hold=10, save_path=None):
    """go through all the edge, find the grad > thread_hold
    if number of point <= 5, using combinations
    else using delaunay
    """
    non_zero_indices = list(zip(*np.nonzero(depth_matrix)))
    non_zero_indices_np = np.array(non_zero_indices)  # 取整点坐标
    n = len(non_zero_indices_np)
    if 1 < n <= 8:
        for i in range(n-1):
            for j in range(i + 1, n):
                D = distance(non_zero_indices_np[i], non_zero_indices_np[j])
                grad = abs(depth_matrix[non_zero_indices[i]] - depth_matrix[non_zero_indices[j]]) / D
                if grad >= thread_hold:
                    return True
                else:
                    return False
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
                    # delaunay_visual(triangulation, non_zero_indices_np, count+1, save_path)
                    return True
    return False


def delaunay_visual(triangulation, points, count, save_path=None):
    # max_size =
    plt.plot(points[:, 0], points[:, 1], 'o', markersize=12, label='Point in Pixel')

    # Plot the Delaunay triangulation
    plt.triplot(points[:, 0], points[:, 1], triangulation.simplices, linewidth=1, color='red')

    # Label the triangles
    for j, p in enumerate(points):
        plt.text(p[0], p[1], f'{j}', ha='left', va='top', fontsize=18)

    plt.xlabel('X-axis', fontsize=18)
    plt.ylabel('Y-axis', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(-1, 25)
    plt.ylim(-1, 25)
    plt.gca().invert_yaxis()
    plt.title('Delaunay Triangulation', fontsize=20)

    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, "{}.svg".format(count)), format='svg')
    plt.show()


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

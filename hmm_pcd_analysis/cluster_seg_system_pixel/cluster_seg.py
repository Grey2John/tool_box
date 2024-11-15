import os.path
import copy
import math
import numpy as np
from grid import GridUnit, delaunay_crack_detect, m_standard
from dbscan_cus import DBSCAN_CUS

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from visualizer import resize_image
import matplotlib.pyplot as plt
import time
import threading


mask_size = [720, 960]  # 矩阵是HxW
state_standard = {
    0: [0, 1, 2],
    1: [[0, 1], [0, 2], [1, 2]],
    2: [[0, 1, 2]]
}
grid_state = {
    3: [0, 1, 3],
    4: [0, 2, 4],
    5: [1, 2, 5]
}
two_edge_state = [[0, 1], [0, 2], [1, 2]]
neighbor_8 = [[i, j] for i in range(-1, 2) for j in range(-1, 2)]


class ClusterSegmentSystem:
    def __init__(self, mask=None, grid_size=8, downsample_pixel=1):
        self.grid_size = grid_size
        self.downsample_pixel = downsample_pixel

        self.mask = mask  # 整体计算的时候要删除，非常耗时
        self.grid_map_size = [int(s / grid_size / downsample_pixel) for s in mask_size]
        self.index_map = np.zeros(self.grid_map_size, dtype=np.int16)  # down sampling
        self.exist_map = np.zeros(self.grid_map_size, dtype=np.int8)  # down sampling
        self.edge_grid_matrix = np.zeros(([3] + self.grid_map_size), dtype=np.uint8)  # 3 channels, 3,4,5 label exist
        self.key_grid_map = np.zeros(self.grid_map_size, dtype=np.int8)  # for visualization
        self.index_numer = 0

        # COO matrix, sparse representation
        self.row_col = []  # [[20, 30], []]
        self.class_list = []  # [[0,0,0,0,0,1,0], ]
        self.grid_list = []  # [GridUnit, ]

        self.edge_state_pixel_dic = {3: [], 4: [], 5: []}  # {3:[index, ], 4:[], 5:[]}
        self.extend_grid_group = []  # 直接根据边缘网格扩展的grid group
        self.state_change_dic_p_index = {0: [], 1: [], 2: [], "none": []}

        self.crack_key_grid = np.zeros(self.grid_map_size, dtype=np.int8)
        self.grid_point_num = np.zeros(self.grid_map_size, dtype=np.int16)
        self.clustered_grid_group = {3: [], 4: [], 5: []}  # for one frame of image

        self.d_time = 0  # delaunay 计时

    def _init_data(self):
        """消除重复元素"""
        for key in self.edge_state_pixel_dic:
            self.edge_state_pixel_dic[key] = list(set(self.edge_state_pixel_dic[key]))

    def grid_map_update(self, point_index, one_point_obs, label_state, uv, uv_int, xyzdepth_c):
        """
            for every point projection, build the grid map
            basic task for cluster segmentation
        """
        # down_uv = np.floor(uv / self.grid_size).astype(int)  # 从round改过来
        # w = int(down_uv[0]/self.downsample_pixel)  # x related to w
        # h = int(down_uv[1]/self.downsample_pixel)
        w = uv_int[0] // self.grid_size
        h = uv_int[1] // self.grid_size
        h_w_origin = np.array([uv_int[1], uv_int[0]])
        self.grid_point_num[h, w] += 1
        if self.exist_map[h, w] == 0:  # new grid
            self.exist_map[h, w] = 1
            self.index_numer += 1
            if one_point_obs in [3, 4, 5]:
                self.key_grid_map[h, w] = 1
                self.edge_grid_matrix[one_point_obs-3, h, w] = 1
                self.edge_state_pixel_dic[one_point_obs].append(self.index_numer-1)
            self.index_map[h, w] = self.index_numer  # 从1开始索引, 元素-1就是稀疏表征的序号

            self.row_col.append([h, w])
            zero_list = [0, 0, 0, 0, 0, 0, 0]
            zero_list[one_point_obs] = 1
            self.class_list.append(zero_list)
            new_grid = GridUnit(h, w, size=self.grid_size, downsample_pixel=self.downsample_pixel)
            new_grid.adjust_grid(point_index, one_point_obs, label_state, h_w_origin, xyzdepth_c)
            self.grid_list.append(new_grid)
        else:
            if one_point_obs in [3, 4, 5]:
                self.key_grid_map[h, w] = 1
                self.edge_grid_matrix[one_point_obs - 3, h, w] = 1
                self.edge_state_pixel_dic[one_point_obs].append(self.index_map[h, w] - 1)
            self.class_list[self.index_map[h, w] - 1][one_point_obs] = 1
            self.grid_list[self.index_map[h, w] - 1].adjust_grid(point_index, one_point_obs,
                                                                 label_state, h_w_origin,
                                                                 xyzdepth_c)

    def cluster_detect(self, min_pts=4, save_path=None):
        """
        for one frame of image
        1. 扩展 Scan-Line Filling 或者 Seed Filling
        2. 聚类，检测判断
        3. 获得key grid-
        4. 新线程做三角形检测，角度梯度变化检测，判定是否优化
        5. 重新给点上标签
        """
        save_path_visual = save_path
        detect_count = 0
        cluster_group_count = 0
        map_time = 0
        cluster_time = 0
        detect_time = 0

        for c, grid_index_list in self.edge_state_pixel_dic.items():  # key class 3, 4, 5，一共执行三次
            # plt.matshow(self.edge_grid_matrix[c-3, :, :])  # 这里设置颜色为红色，也可以选择其他颜色
            # plt.show()
            self.grid_index_list_c = copy.deepcopy(grid_index_list)
            while len(self.grid_index_list_c) > 0:
                """1. extend map"""
                start_map_time = time.time()
                self.fill_record = np.zeros(self.grid_map_size, dtype=np.int8)  # one frame on matrix
                self.temp_key_grid_index = []  # 4 领域
                self.temp_grid_index = []  # 8 领域
                h = self.row_col[self.grid_index_list_c[0]][0]
                w = self.row_col[self.grid_index_list_c[0]][1]
                self._seed_filling(h, w, c)  # Recursive function
                temp_grid_index = list(set(self.temp_grid_index+self.temp_key_grid_index))
                # self.need_cluster_grid_group[c].append(temp_grid_index)
                map_time += (time.time() - start_map_time)
                """2. clustering and judge"""
                start_cluster_time = time.time()
                local_point_index = []
                local_point_depth = []
                local_point_grid_index = []
                local_point_3state = []
                for g in temp_grid_index:
                    local_point_index += self.grid_list[g].point_index
                    local_point_depth += [d[-1] for d in self.grid_list[g].point_xyzdepth_c]
                    local_point_grid_index += [g] * len(self.grid_list[g].point_index)
                    local_point_3state += self.grid_list[g].point_3state
                # cluster
                dbscan_eps = 0.2 if h > 45 else round(0.2 + 1 * (45 - h)/45, 2)
                cluster = DBSCAN_CUS(dbscan_eps, min_pts, local_point_depth)
                clusters_result = cluster.fit()
                if 1 not in clusters_result:
                    continue  # all the point belong to one class
                """3. 历遍gird，查询8邻域，找到多簇的key grid"""
                # for visualization
                cluster_group_count += 1
                class_num_in_grid = {}  # 统计
                key_grid_check = []
                dir_grid2key_grid = {}
                for gg in temp_grid_index:
                    # 观察key grid
                    _h = self.row_col[gg][0]
                    _w = self.row_col[gg][1]
                    _, k_g_list, _ = self.near_grid(_h, _w, c, build_local_map=False)
                    if gg not in dir_grid2key_grid.keys():
                        dir_grid2key_grid[gg] = k_g_list
                    else:
                        dir_grid2key_grid[gg] = list(set(dir_grid2key_grid[gg] + k_g_list))
                for i, g in enumerate(local_point_grid_index):
                    if clusters_result[i] == -1:
                        continue
                    for j in dir_grid2key_grid[g]:
                        if j not in class_num_in_grid.keys():
                            class_num_in_grid[j] = [clusters_result[i]]
                        elif clusters_result[i] not in class_num_in_grid[j]:
                            class_num_in_grid[j].append(clusters_result[i])

                # class_num_in_grid = list(dict.fromkeys(class_num_in_grid))  # list of grid index
                key_grid_check = [key for key, values in class_num_in_grid.items() if len(values) > 1]
                self.clustered_grid_group[c] += key_grid_check  # 可视化用
                cluster_time += (time.time() - start_cluster_time)
                """ 4.denauny"""
                if len(key_grid_check) == 0:
                    continue
                start_detect_time = time.time()
                for task in key_grid_check:
                    detect_count += 1
                    if_break = self.delaunay_detect(c, task, clusters_result,
                                                     local_point_index,
                                                     local_point_3state,
                                                     detect_count,
                                                     save_path_visual)
                    if if_break:
                        break
                detect_time += (time.time() - start_detect_time)
        # self.images_result_show(save_path=save_path_visual)  # for visualization
        print("cluster_group is {}, the number of grid need be detect is {}".format(cluster_group_count, detect_count))
        print("map time is {}, cluster time is {}, detection time is {}".format(map_time, cluster_time, detect_time))
        """according to self.state_change_dic_p_index, fix point label"""
        return self.state_change_dic_p_index

    def delaunay_detect(self, c, g_index, clusters_result, point_index,
                        point_3state, count, save_path=None):
        """二次判定，存入self.state_change_dic_p_index, 使用角度梯度来判定"""
        h = self.row_col[g_index][0]
        w = self.row_col[g_index][1]
        """build pixel grad map 其实可以不建立小矩阵，浪费了资源"""
        # local_map_list is [ [u, v, D], ... ] 9个8*8的小矩阵中的像素数据
        temp_grid_index, near_key_grid_index_list, local_map_np = self.near_grid(h, w, c, build_local_map=True)
        if_break = delaunay_crack_detect(local_map_np,
                                         count,
                                         [h, w],
                                         scale=self.grid_size,
                                         coe=1.8,
                                         save_path=save_path)
        if if_break:
            type_count = np.zeros([np.max(clusters_result) + 1, 2], dtype=np.int16)
            temp_cluster_point_index = dict([(k, []) for k in range(0, np.max(clusters_result) + 1)])
            for i, cluster_label in enumerate(clusters_result):
                if cluster_label == -1:
                    self.state_change_dic_p_index["none"].append(point_index[i])
                    continue  # 2 class
                if point_3state[i] in two_edge_state[c - 3]:
                    type_count[cluster_label, two_edge_state[c - 3].index(point_3state[i])] += 1
                    temp_cluster_point_index[cluster_label].append(point_index[i])
            peak_i = np.argmax(type_count, axis=1)

            for row, p_i in enumerate(peak_i):
                self.state_change_dic_p_index[two_edge_state[c - 3][p_i]] += temp_cluster_point_index[row]
        return if_break

    def near_grid(self, h, w, key_state, build_local_map=False):  # h,w center pixel
        temp_grid_index = []
        near_key_grid_index_list = []  # near key grid
        # detect_map_index = []  # [h=1, w=2, grid index] 24*24
        local_map_list = []
        local_map_np = None

        max_h, max_w = self.grid_map_size
        for row, col in neighbor_8:
            h_ = h + row
            w_ = w + col
            if 0 <= h_ < max_h and 0 <= w_ < max_w and self.exist_map[h_, w_]:  # index start from 1
                index = self.index_map[h_, w_] - 1
                temp_grid_index.append(index)  # temporary grid index list
                if build_local_map:
                    # self.grid_list[index].upsampling_map.local_pixel_list  [hw, xyz, depth_min, depth_max]
                    local_map_list += [np.array(one_pixel) + np.array([row+1, col+1, 0, 0, 0, 0, 0])*self.grid_size
                                       for one_pixel in self.grid_list[index].upsampling_map.local_pixel_list]
                    local_map_np = np.array(local_map_list)

                if self.class_list[index][key_state] == 1 and (row+col) % 2:  # key grid传播使用4邻域
                    near_key_grid_index_list.append(index)  # include the center grid
                    # map_key_index.append([row + 1, col + 1, index])  # 24*24
        return temp_grid_index, near_key_grid_index_list, local_map_np

    def _seed_filling(self, h, w, c):
        """Seed Filling: 递归检索，
        统计通过的key grid index，通过的 normal grid index
        使用扩展的矩阵来搜索padding，避免超出索引"""
        if self.exist_map[h, w]:
            index = self.index_map[h, w] - 1
            self.temp_grid_index.append(index)
        if self.edge_grid_matrix[c-3, h, w] != 0 and self.fill_record[h, w] != 1:
            try:
                self.grid_index_list_c.remove(index)
            except ValueError:
                pass
            for i in [-1, 1]:
                for j in [-1, 1]:
                    _h, _w = h + i, w + j
                    if 0 <= _h < self.exist_map.shape[0] and 0 <= _w < self.exist_map.shape[1]:
                        if self.exist_map[_h, _w]:
                            self.temp_grid_index.append(self.index_map[_h, _w] - 1)
            self.temp_key_grid_index.append(index)
            self.fill_record[h, w] = 1
            if h + 1 < self.exist_map.shape[0]:
                self._seed_filling(h + 1, w, c)
            if h - 1 >= 0:
                self._seed_filling(h - 1, w, c)
            if w + 1 < self.exist_map.shape[1]:
                self._seed_filling(h, w + 1, c)
            if w - 1 >= 0:
                self._seed_filling(h, w - 1, c)

    def images_result_show(self, save_path=None):
        """展示聚类的结果，展示delaunay检索的点，展示delaunay超过阈值的点"""
        need_cluster_grid_map = np.zeros(self.grid_map_size, dtype=np.int8)
        for k, group in self.clustered_grid_group.items():
            for g in group:
                coord = self.row_col[g]
                need_cluster_grid_map[coord[0], coord[1]] = k

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 20))
        axes[0].imshow(self.exist_map + self.key_grid_map)  # cmap='gray'表示使用灰度颜色映射
        axes[0].set_title('Grids with observed points in grid map')

        axes[1].imshow(self.exist_map + need_cluster_grid_map + self.crack_key_grid * 4)
        axes[1].set_title('Extension groups to be clustered in grid map')

        axes[2].imshow(self.mask + resize_image(need_cluster_grid_map, int(self.grid_size*self.downsample_pixel)))
        axes[2].set_title('Grid groups with gradient mutations in mask')
        # axes[1, 0].imshow(self.grid_point_num)
        # axes[1, 0].set_title('number of point in grid')

        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=20)
        # 设置x轴和y轴标签的字体大小
        for ax in axes:
            ax.set_xlabel('x-axis', fontsize=20)
            ax.set_ylabel('y-axis', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
        # axes[1, 1].imshow(self.exist_map + self.crack_key_grid)
        # axes[1, 1].set_title('creak key grid')

        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "detect_results.svg"), format='svg')
        plt.show()

    def cluster_show(self, group_point_depth, clusters_result):
        """draw the clustered result in one dimension"""
        x = clusters_result.tolist()
        plt.scatter(clusters_result.tolist(), group_point_depth, marker='x', label='Crosses')
        plt.text(0, 0, 'size is {}'.format(len(group_point_depth)), fontsize=12, color='red')
        plt.xlabel('cluster class-axis')
        plt.ylabel('depth-axis')
        plt.title('Scatter Plot of Points')
        plt.legend()
        plt.xticks(range(int(min(x)), int(max(x)) + 1, 1))
        plt.grid(True)
        plt.show()

    def grid_locate_visualizer(self, h, w, if_extension, depth_matrix):
        """show the position of key grid in the grid map"""
        grid = np.zeros(self.grid_map_size, dtype=np.int8)
        grid[h, w] += 1
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))

        axes[0].imshow(self.exist_map + grid)  # cmap='gray'表示使用灰度颜色映射
        axes[0].set_title('grid located in grid map, grid crack {}'.format(if_extension))

        axes[1].imshow(self.mask + resize_image(grid, int(self.grid_size)))
        axes[1].set_title('grid located in mask')
        plt.tight_layout()

        x = np.arange(0, depth_matrix.shape[0], 1)
        y = np.arange(0, depth_matrix.shape[1], 1)
        x, y = np.meshgrid(x, y)
        ax3d = fig.add_subplot(3, 1, 3, projection='3d')
        ax3d.bar3d(x.flatten(), y.flatten(), np.zeros_like(depth_matrix.flatten()),
                   1, 1, depth_matrix.flatten(), shade=True)
        for i, (x_loc, y_loc, z_loc) in enumerate(zip(x.flatten(), y.flatten(), depth_matrix.flatten())):
            ax3d.text(x_loc + 0.5, y_loc + 0.5, z_loc, str(round(z_loc, 2)), ha='center', va='center')

        plt.show()

    def show_3d_process_result(self, point_list):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Extract x, y, and z coordinates from the point list
        x = [point[0] for point in point_list]
        y = [point[1] for point in point_list]
        z = [point[2] for point in point_list]

        # Plot the points
        ax.scatter(x, y, z, c='r', marker='o')

        # Set axis labels
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()


class EdgeDetectors:
    def __init__(self):
        """edge detector for depth image"""
        pass


# class ClusterCOOMatrix:
#     def __init__(self):
#         """the image index map for grid"""
#         self.row_list = []
#         self.col_list = []
#         self.grid_list = []  # GridUnit
#
#     def add_grid(self, new_grid):
#         self.row_list.append(new_grid.row)
#         self.col_list.append(new_grid.col)
#         self.grid_list.append(new_grid)
#
#     def adjust_grid(self, index):
#         # self.grid_list[index]
#         return None

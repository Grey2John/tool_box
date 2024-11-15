import os.path
import copy

import numpy as np
from grid import GridUnit, delaunay_crack_detect
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
        self.edge_grid_matrix = np.zeros((self.grid_map_size + [3]), dtype=np.uint8)  # 3 channels, 3,4,5 label exist
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
        self.need_cluster_grid_group = {3: [], 4: [], 5: []}  # for one frame of image

        self.d_time = 0  # delaunay 计时

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
            if one_point_obs in [3, 4, 5]:
                self.key_grid_map[h, w] = 1
                self.edge_grid_matrix[h, w, one_point_obs-3] = 1
                self.edge_state_pixel_dic[one_point_obs].append(self.index_numer)
            self.index_map[h, w] = self.index_numer + 1  # 从1开始索引, 元素-1就是稀疏表征的序号
            self.index_numer += 1

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
                self.edge_grid_matrix[h, w, one_point_obs - 3] = 1
                self.edge_state_pixel_dic[one_point_obs].append(self.index_map[h, w] - 1)
            self.class_list[self.index_map[h, w] - 1][one_point_obs] = 1
            self.grid_list[self.index_map[h, w] - 1].adjust_grid(point_index, one_point_obs,
                                                                 label_state, h_w_origin,
                                                                 xyzdepth_c)

    def cluster_detect(self, dbscan_eps=0.1, min_pts=5, thread_hold=0.98, save_path=None):
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
        map_time = 0
        cluster_time = 0
        detect_time = 0

        for c, grid_index_list in self.edge_state_pixel_dic.items():  # key class 3, 4, 5，一共执行三次
            # c is the edge class
            grid_index_list_c = copy.deepcopy(grid_index_list)
            key_grid_processed = []  # already add into the cluster group
            for grid_index in grid_index_list:  # grid_index_list 应当先排好顺序
                """1. extend map"""
                start_map_time = time.time()
                if grid_index in key_grid_processed:
                    continue
                key_grid_processed.append(grid_index)  # add into processed group

                h = self.row_col[grid_index][0]
                w = self.row_col[grid_index][1]
                extend_key_grid_processed = [grid_index]  # one grid group
                self.crack_key_grid[h, w] = c  # for visualization
                temp_grid_index, near_key_grid_index_list, _ = self.near_grid(h, w, c)  # 选send点并初始化
                continue_find = True
                while continue_find:  # link to all the key grid
                    rest_near_key_grid_index_list = []  # the key grids don't process
                    for rest in near_key_grid_index_list:
                        if rest not in extend_key_grid_processed:
                            rest_near_key_grid_index_list.append(rest)

                    for g_index in rest_near_key_grid_index_list:
                        key_grid_processed.append(grid_index)  # 被重复
                        _temp_grid_index, _near_key_grid_index_list, _ = self.near_grid(self.row_col[g_index][0],
                                                                                        self.row_col[g_index][1], c)
                        extend_key_grid_processed.append(g_index)
                        temp_grid_index += _temp_grid_index
                        near_key_grid_index_list += _near_key_grid_index_list
                    temp_grid_index = list(set(temp_grid_index))
                    near_key_grid_index_list = list(set(near_key_grid_index_list))

                    if set(near_key_grid_index_list).issubset(set(extend_key_grid_processed)):
                        continue_find = False
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
                cluster = DBSCAN_CUS(dbscan_eps, min_pts, local_point_depth)
                clusters_result = cluster.fit()
                if 1 not in clusters_result:
                    continue  # all the point belong to one class
                """3. 找到需要重新检测的key grid"""
                class_num_in_grid = {}  # 统计
                key_grid_check = []
                for i, g in enumerate(local_point_grid_index):
                    if clusters_result[i] == -1:
                        continue
                    if g not in class_num_in_grid.keys():
                        class_num_in_grid[g] = [clusters_result[i]]
                    elif clusters_result[i] not in class_num_in_grid[g]:
                        class_num_in_grid[g].append(clusters_result[i])
                        key_grid_check.append(g)
                key_grid_check = list(dict.fromkeys(key_grid_check))  # list of grid index
                cluster_time += (time.time() - start_cluster_time)
                """4. new thread and denauny, 只计算前三项，找不到就拉倒"""
                # for task in key_grid_check:
                #     # 创建一个线程并将其目标设置为 worker 函数
                #     thread = threading.Thread(target=self.delaunay_detect, args=(c, task, clusters_result,
                #                                                                  local_point_index,
                #                                                                  local_point_3state,
                #                                                                  thread_hold,
                #                                                                  save_path_visual,))
                #     thread.start()  # 启动线程
                #     threads.append(thread)  # 将线程添加到线程列表中
                # # 等待所有线程完成
                # for thread in threads:
                #     thread.join()
                start_detect_time = time.time()
                for task in key_grid_check:
                    detect_count += 1
                    if_break = self.delaunay_detect(c, task, clusters_result,
                                                     local_point_index,
                                                     local_point_3state,
                                                     thread_hold,
                                                     save_path_visual)
                    if if_break:
                        break
                detect_time += (time.time() - start_detect_time)
        # self.images_result_show(save_path=save_path_visual)  # for visualization
        print("the number of grid need be detect is {}".format(detect_count))
        print("map time is {}, cluster time is {}, detection time is {}".format(map_time, cluster_time, detect_time))
        """according to self.state_change_dic_p_index, fix point label"""
        return self.state_change_dic_p_index

    def delaunay_detect(self, c, g_index, clusters_result, point_index,
                        point_3state, thread_hold=0.98, save_path=None):
        """二次判定，存入self.state_change_dic_p_index, 使用角度梯度来判定"""
        h = self.row_col[g_index][0]
        w = self.row_col[g_index][1]
        """build pixel grad map 其实可以不建立小矩阵，浪费了资源"""
        # local_map_list is [ [u, v, D], ... ] 9个8*8的小矩阵中的像素数据
        temp_grid_index, near_key_grid_index_list, local_map_np = self.near_grid(h, w, c, build_local_map=True)
        if_break = delaunay_crack_detect(local_map_np,
                                         thread_hold=thread_hold,
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
                    # self.grid_list[index].upsampling_map.local_pixel_list  [hw, xyz, depth]
                    local_map_list += [np.array(one_pixel) + np.array([row+1, col+1, 0, 0, 0, 0])*self.grid_size
                                       for one_pixel in self.grid_list[index].upsampling_map.local_pixel_list]
                    local_map_np = np.array(local_map_list)

                if self.class_list[index][key_state] == 1 and (row+col) % 2:  # key grid传播使用4邻域
                    near_key_grid_index_list.append(index)  # include the center grid
                    # map_key_index.append([row + 1, col + 1, index])  # 24*24
        return temp_grid_index, near_key_grid_index_list, local_map_np

    def _seed_filling(self, x, y, fill_label, boundary_label):
        c = getpixel(x, y)
        if c != boundary_label and c != fill_label:
            putpixel(x, y, fill_label)
            self._seed_filling(x + 1, y, fill_label, boundary_label)
            self._seed_filling(x - 1, y, fill_label, boundary_label)
            self._seed_filling(x, y + 1, fill_label, boundary_label)
            self._seed_filling(x, y - 1, fill_label, boundary_label)

    def images_result_show(self, save_path=None):
        need_cluster_grid_map = np.zeros(self.grid_map_size, dtype=np.int8)
        for k, group in self.need_cluster_grid_group.items():
            for g in group:
                for i in g:
                    coord = self.row_col[i]
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


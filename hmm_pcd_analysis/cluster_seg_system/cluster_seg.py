import os.path

import numpy as np
from grid import GridUnit

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from visualizer import edge_mask, resize_image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kdtree import KDTree
from delaunay import delaunay_crack_detect


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


class ClusterSegmentSystem:
    def __init__(self, mask=None, scale_factor=0.2):
        self.scale_factor = scale_factor
        self.scale_d = int(1/scale_factor)
        self.scaled_mask_size_mid_line = [x * scale_factor / 2 for x in mask_size]  # 矩阵是HxW
        self.mask = mask   # 整体计算的时候要删除，非常耗时
        self.down_sampling_size = [int(s*scale_factor) for s in mask_size]
        self.index_map = np.zeros(self.down_sampling_size, dtype=np.int16)  # down sampling
        self.exist_map = np.zeros(self.down_sampling_size, dtype=np.int8)  # down sampling
        self.index_numer = 1

        # COO matrix
        self.row_col = []  # [[20, 30], []]
        self.class_list = []  # [[0,0,0,0,0,1,0], ]
        self.grid_list = []  # main data

        self.crack_key_grid = np.zeros(self.down_sampling_size, dtype=np.int8)
        self.grid_point_num = np.zeros(self.down_sampling_size, dtype=np.int16)
        self.need_cluster_grid_group = {3:[], 4:[], 5:[]}  # for one frame of image

    def grid_map_update(self, point_index, one_point_obs, label_state, xy, xydepth_c):
        """
            for every point projection
            basic task for cluster segmentation
        """
        down_uv = np.round(xy * self.scale_factor).astype(int)
        w = int(down_uv[0])  # x related to w
        h = int(down_uv[1])
        self.grid_point_num[h, w] += 1
        if self.exist_map[h, w] == 0:  # new grid
            self.exist_map[h, w] = 1
            self.index_map[h, w] = self.index_numer  # 从1开始索引
            self.index_numer += 1

            self.row_col.append([h, w])
            zero_list = [0, 0, 0, 0, 0, 0, 0]
            zero_list[one_point_obs] = 1
            self.class_list.append(zero_list)
            new_grid = GridUnit(h, w)
            new_grid.adjust_grid(point_index, one_point_obs, label_state, xy, xydepth_c)
            self.grid_list.append(new_grid)
        else:
            self.class_list[self.index_map[h, w]-1][one_point_obs] = 1
            self.grid_list[self.index_map[h, w] - 1].adjust_grid(point_index, one_point_obs, label_state, xy, xydepth_c)

    def crack_detect(self, thread_hold=5, save_path=None):
        """
        for one frame of image
        Go through all key grid to find crack
        """
        """find the crack place, described by a seed of key grid"""
        crack_key_grid_index = {3: [], 4: [], 5: []}
        crack_count = 0
        check_count = 0
        for c in range(3, 6):  # key class 3, 4, 5
            # temp_class_mix = np.zeros(7, dtype=np.int8)  # class statics
            for i, l in enumerate(self.class_list):  # i, l - grid index, one class_list, load all key grid
                if l[c] != 1:
                    continue
                kdtree_points_one_region = []
                delaunay_points_one_region = []
                square_list = []
                check_count += 1

                h = self.row_col[i][0]
                w = self.row_col[i][1]
                temp_grid_index, near_key_grid_index_list = self.near_grid(h, w, c)

                """using kdtree"""
                # for g_index in temp_grid_index:
                #     for j, one_p in enumerate(self.grid_list[g_index].point_xydepth):
                #         if self.grid_list[g_index].point_obs_state[j] not in grid_state[c]:
                #             continue
                #         kdtree_point = one_p.copy()  # [xyz] in camera
                #         kdtree_point.append(self.grid_list[g_index].point_index[j])  # index
                #         kdtree_points_one_region.append(kdtree_point)
                #
                # if abs(h-self.scaled_mask_size_mid_line[0]) > abs(w-self.scaled_mask_size_mid_line[1]):
                #     x_first = 0
                # else:
                #     x_first = 1  # horizontal cut
                # one_tree = KDTree(kdtree_points_one_region, x_first)
                # # square_list.append(patches.Rectangle((x, y), self.scale_d, self.scale_d, edgecolor='none', facecolor='red', alpha=0.5))
                # if one_tree.crack_detect(thread_hold):
                #     crack_count += 1
                #     # one_tree.kdtree_visualize(crack_count, square_list, save_path)
                #     crack_key_grid_index[c].append(self.index_map[h, w] - 1)
                #     """crack_key_grid is the seed for generating the clustering region """
                #     self.crack_key_grid[h, w] = c  # for visualization
                """using Delaunay"""
                for g_index in temp_grid_index:
                    for j, one_p in enumerate(self.grid_list[g_index].point_xydepth):
                        if self.grid_list[g_index].point_obs_state[j] not in grid_state[c]:
                            continue
                        delaunay_point = one_p.copy()  # [xyz] in camera
                        delaunay_point.append(self.grid_list[g_index].point_index[j])  # index
                        delaunay_points_one_region.append(delaunay_point)
                if delaunay_crack_detect(delaunay_points_one_region, thread_hold):
                    crack_count += 1
                    crack_key_grid_index[c].append(self.index_map[h, w] - 1)
                    self.crack_key_grid[h, w] = c  # for visualization
        print("the number of grid need be check is {}".format(check_count))
        """further more find the seg region, key grid directly linked to build cluster region"""
        key_grid_processed = []  # already add into the cluster group
        for c, g_i_list in crack_key_grid_index.items():  # seed grid
            for g in g_i_list:  # g is the index of seed
                continue_find = True
                one_key_grid_seed = 0
                h = self.row_col[g][0]
                w = self.row_col[g][1]
                temp_grid_index, near_key_grid_index_list = self.near_grid(h, w, c)
                key_grid_processed.append(g)

                while continue_find:  # link to all the key grid
                    rest_near_key_grid_index_list = []
                    for rest in near_key_grid_index_list:
                        if rest not in key_grid_processed:
                            rest_near_key_grid_index_list.append(rest)

                    for g_index in rest_near_key_grid_index_list:
                        h = self.row_col[g_index][0]
                        w = self.row_col[g_index][1]
                        _temp_grid_index, _near_key_grid_index_list = self.near_grid(h, w, c)

                        key_grid_processed.append(g_index)  # add into processed group
                        temp_grid_index += _temp_grid_index
                        near_key_grid_index_list += _near_key_grid_index_list

                        temp_grid_index = list(set(temp_grid_index))
                        near_key_grid_index_list = list(set(near_key_grid_index_list))

                    # if set(near_key_grid_index_list).issubset(set(key_grid_processed)) or \
                    #         len(near_key_grid_index_list) > 20:
                    if set(near_key_grid_index_list).issubset(set(key_grid_processed)):
                        continue_find = False
                        break

                self.need_cluster_grid_group[c].append(temp_grid_index)
        # self.images_result_show(save_path)  # for visualization

    def near_grid(self, h, w, key_state):  # h,w center pixel
        temp_grid_index = []
        near_key_grid_index_list = []  # near key grid
        # max_depth = []
        # min_depth = []
        # depth_xyz_c = [] # [max, min, [] ]
        for row in range(-1, 2):
            for col in range(-1, 2):
                h_ = h + row
                w_ = w + col
                if (0 <= h_ < self.down_sampling_size[0]) \
                        and (0 <= w_ < self.down_sampling_size[1]) \
                        and self.index_map[h_, w_] != 0:  # index start from 1
                    index = self.index_map[h_, w_] - 1
                    temp_grid_index.append(index)  # temporary grid index list
                    # max_depth.append(self.grid_list[index].max_depth)
                    # depth_xyz_c.append(self.grid_list[index].max_pixel_hw)
                    # min_depth.append(self.grid_list[index].min_depth)
                    # depth_xyz_c.append(self.grid_list[index].min_pixel_hw)

                    if self.class_list[index][key_state] == 1:
                        near_key_grid_index_list.append(index)   # include the center grid
        return temp_grid_index, near_key_grid_index_list

    def cluster_process(self, dbscan_eps=10, min_samples=4):
        """for one frame of image
        find the points, cluster the points, seg the points
        return: fix point dictionary"""
        state_change_dic_p_index = {0:[], 1:[], 2:[], "none": []}

        for c, group in self.need_cluster_grid_group.items():
            for g in group:
                # g is one cluster seg
                group_point_index = []
                group_point_depth = []
                group_point_state = []
                for i_grid in g:
                    group_point_index += self.grid_list[i_grid].point_index
                    group_point_depth += [d[-1] for d in self.grid_list[i_grid].point_xydepth]
                    group_point_state += self.grid_list[i_grid].point_obs_state

                cluster_data = np.array(group_point_depth)
                cluster_data2D = cluster_data.reshape(-1, 1)
                cluster_data2D_matrix = squareform(pdist(cluster_data2D, metric='euclidean'))
                dbscan = DBSCAN(eps=dbscan_eps, min_samples=min_samples, metric='precomputed')
                clusters_result = dbscan.fit_predict(cluster_data2D_matrix)  # point index, -1 is noise
                # [0,  0,  0,  0 , 0 , 1,  1,  1,  1, -1,  2,  2,  2,  2,  2]
                if 2 not in list(set(clusters_result)):  # 2 class
                    type_count = np.zeros([2, 2], dtype=np.int8)  # row is cluster class
                    for i in clusters_result:
                        if i >= 0 and (group_point_state[i] in two_edge_state[c-3]):
                            type_count[i, two_edge_state[c-3].index(group_point_state[i])] += 1
                    peak_i = np.argmax(type_count, axis=1)
                    for j, l in enumerate(clusters_result):
                        if l == -1:
                            state_change_dic_p_index["none"].append(group_point_index[j])
                            continue
                        state_change_dic_p_index[ two_edge_state[c-3][peak_i[l]] ].append(group_point_index[j])
                else:  # 多个类直接观测作废
                    for k, l in enumerate(group_point_index):
                        state_change_dic_p_index["none"].append(l)
        return state_change_dic_p_index

    def images_result_show(self, save_path=None):
        need_cluster_grid_map = np.zeros(self.down_sampling_size, dtype=np.int8)
        for k, group in self.need_cluster_grid_group.items():
            for g in group:
                for i in g:
                    coord = self.row_col[i]
                    need_cluster_grid_map[coord[0], coord[1]] = k

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
        axes[0, 0].imshow(self.exist_map)  # cmap='gray'表示使用灰度颜色映射
        axes[0, 0].set_title('exist map')

        axes[0, 1].imshow(self.mask + resize_image(need_cluster_grid_map, int(1/self.scale_factor)))
        axes[0, 1].set_title('crack grid')

        # axes[1, 0].imshow(self.grid_point_num)
        # axes[1, 0].set_title('number of point in grid')
        axes[1, 0].imshow(self.exist_map + need_cluster_grid_map + self.crack_key_grid*4)
        axes[1, 0].set_title('mix map')

        axes[1, 1].imshow(self.exist_map + self.crack_key_grid)
        axes[1, 1].set_title('creak key grid')

        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "detect_results.svg"), format='svg')
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


class ClusterCOOMatrix:
    def __init__(self):
        """the image index map for grid"""
        self.row_list = []
        self.col_list = []
        self.grid_list = []  # GridUnit

    def add_grid(self, new_grid):
        self.row_list.append(new_grid.row)
        self.col_list.append(new_grid.col)
        self.grid_list.append(new_grid)

    def adjust_grid(self, index):
        # self.grid_list[index]
        return None
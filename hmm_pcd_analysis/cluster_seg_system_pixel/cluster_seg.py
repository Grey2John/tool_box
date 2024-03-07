import os.path

import numpy as np
from grid import GridUnit, delaunay_crack_detect

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from visualizer import edge_mask, resize_image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    def __init__(self, mask=None, scale_factor=8, upsample_pixel=2):
        self.scale_factor = scale_factor
        # self.upsample_pixel = upsample_pixel
        self.scale_d = scale_factor
        self.scaled_mask_size_mid_line = [x / scale_factor / 2 for x in mask_size]  # 矩阵是HxW
        self.mask = mask   # 整体计算的时候要删除，非常耗时
        self.down_sampling_size = [int(s/scale_factor) for s in mask_size]
        self.index_map = np.zeros(self.down_sampling_size, dtype=np.int16)  # down sampling
        self.exist_map = np.zeros(self.down_sampling_size, dtype=np.int8)  # down sampling
        self.key_grid_map = np.zeros(self.down_sampling_size, dtype=np.int8)  # for visualization
        self.index_numer = 0

        # COO matrix
        self.row_col = []  # [[20, 30], []]
        self.class_list = []  # [[0,0,0,0,0,1,0], ]
        self.grid_list = []  # [GridUnit, ]
        self.edge_state_pixel_dic = {3:[], 4:[], 5:[]}  # {3:[index, ], 4:[], 5:[]}

        self.crack_key_grid = np.zeros(self.down_sampling_size, dtype=np.int8)
        self.grid_point_num = np.zeros(self.down_sampling_size, dtype=np.int16)
        self.need_cluster_grid_group = {3:[], 4:[], 5:[]}  # for one frame of image

    def grid_map_update(self, point_index, one_point_obs, label_state, xy, xy_int, xydepth_c):
        """
            for every point projection
            basic task for cluster segmentation
        """
        down_uv = np.floor(xy / self.scale_factor).astype(int)  # 从round改过来
        w = int(down_uv[0])  # x related to w
        h = int(down_uv[1])
        h_w_origin = np.array([xy_int[1], xy_int[0]])
        self.grid_point_num[h, w] += 1
        if self.exist_map[h, w] == 0:  # new grid
            self.exist_map[h, w] = 1
            if one_point_obs in [3, 4, 5]:
                self.key_grid_map[h, w] = 1
                self.edge_state_pixel_dic[one_point_obs].append(self.index_numer)
            self.index_map[h, w] = self.index_numer+1  # 从1开始索引
            self.index_numer += 1

            self.row_col.append([h, w])
            zero_list = [0, 0, 0, 0, 0, 0, 0]
            zero_list[one_point_obs] = 1
            self.class_list.append(zero_list)
            new_grid = GridUnit(h, w, size=self.scale_d)
            new_grid.adjust_grid(point_index, one_point_obs, label_state, h_w_origin, xydepth_c)
            self.grid_list.append(new_grid)
        else:
            if one_point_obs in [3, 4, 5]:
                self.key_grid_map[h, w] = 1
                self.edge_state_pixel_dic[one_point_obs].append(self.index_map[h, w]-1)
            self.class_list[self.index_map[h, w]-1][one_point_obs] = 1
            self.grid_list[self.index_map[h, w] - 1].adjust_grid(point_index, one_point_obs, label_state, h_w_origin, xydepth_c)

    # def crack_detect(self, thread_hold=3, save_path=None):
    #     """
    #     for one frame of image
    #     Go through all key grid to find crack
    #     """
    #     """find the crack place, described by a seed of key grid"""
    #     check_count = 0
    #     crack_count = 0
    #
    #     for c in range(3, 6):  # key class 3, 4, 5
    #         # temp_class_mix = np.zeros(7, dtype=np.int8)  # class statics
    #         key_grid_processed = []  # already add into the cluster group
    #         for grid_index, l in enumerate(self.class_list):  # i, l - grid index, one class_list, load all key grid
    #             if l[c] != 1 or (grid_index in key_grid_processed):
    #                 continue
    #             check_count += 1
    #             key_grid_processed.append(grid_index)
    #
    #             if_extension = False
    #             if_multi_detect = False
    #             if self.grid_list[grid_index].upsampling_map.one_pixel_grad_check(thread_hold):
    #                 """the gradient break in one pixel"""
    #                 if_extension = True
    #                 h = self.row_col[grid_index][0]
    #                 w = self.row_col[grid_index][1]
    #                 temp_grid_index, near_key_grid_index_list, combination_key_map = self.near_grid(h, w, c)
    #             else:
    #                 h = self.row_col[grid_index][0]
    #                 w = self.row_col[grid_index][1]
    #                 temp_grid_index, near_key_grid_index_list, combination_key_map = self.near_grid(h, w, c)
    #                 """build pixel grad map"""
    #                 zero = np.zeros([self.scale_d, self.scale_d])
    #                 combination_exit_list = [[zero, zero, zero],
    #                                          [zero, zero, zero],
    #                                          [zero, zero, zero]]
    #                 combination_depth_list = combination_exit_list.copy()
    #                 for cm in combination_key_map:
    #                     combination_exit_list[cm[0]][cm[1]] = self.grid_list[cm[2]].upsampling_map.exist_map
    #                     combination_depth_list[cm[0]][cm[1]] = self.grid_list[cm[2]].upsampling_map.max_depth_matrix
    #                 combination_exit_matrix = np.block(combination_exit_list)
    #                 combination_depth_matrix = np.block(combination_depth_list)
    #                 if_multi_detect = True
    #
    #                 if_extension = delaunay_crack_detect(combination_exit_matrix,
    #                                                      combination_depth_matrix,
    #                                                      crack_count,
    #                                                      thread_hold=thread_hold,
    #                                                      save_path=save_path)
    #                 # self.grid_locate_visualizer(h, w, if_extension, combination_depth_matrix)  # key grid visualization
    #             """if crack and extension"""
    #             """further more find the seg region, key grid directly linked to build cluster region"""
    #             if if_extension:
    #                 crack_count += 1
    #                 self.crack_key_grid[h, w] = c  # for visualization
    #
    #                 continue_find = True
    #                 while continue_find:  # link to all the key grid
    #                     rest_near_key_grid_index_list = []  # the key grids don't process
    #                     for rest in near_key_grid_index_list:
    #                         if rest not in key_grid_processed:
    #                             rest_near_key_grid_index_list.append(rest)
    #
    #                     for g_index in rest_near_key_grid_index_list:
    #                         h_ = self.row_col[g_index][0]
    #                         w_ = self.row_col[g_index][1]
    #                         _temp_grid_index, _near_key_grid_index_list, _ = self.near_grid(h_, w_, c)
    #
    #                         key_grid_processed.append(g_index)  # add into processed group
    #                         temp_grid_index += _temp_grid_index
    #                         near_key_grid_index_list += _near_key_grid_index_list
    #                         temp_grid_index = list(set(temp_grid_index))
    #                         near_key_grid_index_list = list(set(near_key_grid_index_list))
    #
    #                     if set(near_key_grid_index_list).issubset(set(key_grid_processed)):
    #                         continue_find = False
    #                         break
    #                 self.need_cluster_grid_group[c].append(temp_grid_index)
    #             elif if_multi_detect:
    #                 key_grid_processed += near_key_grid_index_list  # reduce cost time
    #     print("the number of grid need be check is {}".format(check_count))
        # self.images_result_show(save_path)  # for visualization

    def crack_detect(self, thread_hold=3, save_path=None):
        """
        for one frame of image
        Go through all key grid to find crack
        """
        """find the crack place, described by a seed of key grid"""
        check_count = 0
        crack_count = 0

        for c, grid_index_list in self.edge_state_pixel_dic.items():  # key class 3, 4, 5
            # temp_class_mix = np.zeros(7, dtype=np.int8)  # class statics
            key_grid_processed = []  # already add into the cluster group
            for grid_index in grid_index_list:
                if grid_index in key_grid_processed:
                    continue
                check_count += 1
                key_grid_processed.append(grid_index)

                if_extension = False
                if_multi_detect = False
                if self.grid_list[grid_index].upsampling_map.one_pixel_grad_check(thread_hold):
                    """the gradient break in one pixel"""
                    if_extension = True
                    h = self.row_col[grid_index][0]
                    w = self.row_col[grid_index][1]
                    temp_grid_index, near_key_grid_index_list, combination_key_map = self.near_grid(h, w, c)
                else:
                    h = self.row_col[grid_index][0]
                    w = self.row_col[grid_index][1]
                    temp_grid_index, near_key_grid_index_list, combination_key_map = self.near_grid(h, w, c)
                    """build pixel grad map"""
                    zero = np.zeros([self.scale_d, self.scale_d])
                    combination_exit_list = [[zero, zero, zero],
                                             [zero, zero, zero],
                                             [zero, zero, zero]]
                    combination_depth_list = combination_exit_list.copy()
                    for cm in combination_key_map:
                        # combination_exit_list[cm[0]][cm[1]] = self.grid_list[cm[2]].upsampling_map.exist_map
                        combination_depth_list[cm[0]][cm[1]] = self.grid_list[cm[2]].upsampling_map.max_depth_matrix
                    # combination_exit_matrix = np.block(combination_exit_list)
                    combination_depth_matrix = np.block(combination_depth_list)
                    if_multi_detect = True

                    if_extension = delaunay_crack_detect(combination_depth_matrix,
                                                         crack_count,
                                                         thread_hold=thread_hold,
                                                         save_path=save_path)
                    # self.grid_locate_visualizer(h, w, if_extension, combination_depth_matrix)  # key grid visualization
                """if crack and extension"""
                """further more find the seg region, key grid directly linked to build cluster region"""
                if if_extension:
                    crack_count += 1
                    self.crack_key_grid[h, w] = c  # for visualization

                    continue_find = True
                    while continue_find:  # link to all the key grid
                        rest_near_key_grid_index_list = []  # the key grids don't process
                        for rest in near_key_grid_index_list:
                            if rest not in key_grid_processed:
                                rest_near_key_grid_index_list.append(rest)

                        for g_index in rest_near_key_grid_index_list:
                            h_ = self.row_col[g_index][0]
                            w_ = self.row_col[g_index][1]
                            _temp_grid_index, _near_key_grid_index_list, _ = self.near_grid(h_, w_, c)

                            key_grid_processed.append(g_index)  # add into processed group
                            temp_grid_index += _temp_grid_index
                            near_key_grid_index_list += _near_key_grid_index_list
                            temp_grid_index = list(set(temp_grid_index))
                            near_key_grid_index_list = list(set(near_key_grid_index_list))

                        if set(near_key_grid_index_list).issubset(set(key_grid_processed)):
                            continue_find = False
                            break
                    self.need_cluster_grid_group[c].append(temp_grid_index)
                elif if_multi_detect:
                    key_grid_processed += near_key_grid_index_list  # reduce cost time
        print("the number of grid need be check is {}, crack is {}".format(check_count, crack_count))

    def near_grid(self, h, w, key_state):  # h,w center pixel
        temp_grid_index = []
        near_key_grid_index_list = []  # near key grid
        map_key_index = []  # [h=1, w=2, grid index]

        for row in range(-1, 2):
            for col in range(-1, 2):
                h_ = h + row
                w_ = w + col
                if (0 <= h_ < self.down_sampling_size[0]) \
                        and (0 <= w_ < self.down_sampling_size[1]) \
                        and self.exist_map[h_, w_] != 0:  # index start from 1
                    index = self.index_map[h_, w_] - 1
                    temp_grid_index.append(index)  # temporary grid index list

                    if self.class_list[index][key_state] == 1:
                        near_key_grid_index_list.append(index)   # include the center grid
                        map_key_index.append([row + 1, col + 1, index])
        return temp_grid_index, near_key_grid_index_list, map_key_index

    def cluster_process(self, dbscan_eps=2, min_pts=3):
        """for one frame of image
        find the points, cluster the points, seg the points
        return: fix point dictionary
        it can be down by multi-thread method"""
        state_change_dic_p_index = {0:[], 1:[], 2:[], "none": []}

        for c, group in self.need_cluster_grid_group.items():
            for g_index in group:
                # g_index is one cluster seg
                group_point_index = []
                group_point_depth = []
                group_point_3state = []
                for i_grid in g_index:
                    group_point_index += self.grid_list[i_grid].point_index
                    group_point_depth += [d[-1] for d in self.grid_list[i_grid].point_xydepth]
                    group_point_3state += self.grid_list[i_grid].point_3state  # direct segmentation state

                cluster_data = np.array(group_point_depth)
                cluster_data2D = cluster_data.reshape(-1, 1)
                cluster_data2D_matrix = squareform(pdist(cluster_data2D, metric='euclidean'))
                dbscan = DBSCAN(eps=dbscan_eps, min_samples=min_pts, metric='precomputed')
                clusters_result = dbscan.fit_predict(cluster_data2D_matrix)  # point index, -1 is noise
                # self.cluster_show(group_point_depth, clusters_result)    # for visualization
                # [0,  0,  0,  0 , 0 , 1,  1,  1,  1, -1,  2,  2,  2,  2,  2]
                if 1 not in clusters_result:
                    continue  # all the point belong to one class
                type_count = np.zeros([np.max(clusters_result)+1, 2], dtype=np.int16)
                temp_cluster_point_index = dict([(k, []) for k in range(0, np.max(clusters_result)+1)])
                for i, cluster_label in enumerate(clusters_result):
                    if cluster_label == -1:
                        state_change_dic_p_index["none"].append(group_point_index[i])
                        continue  # 2 class
                    # row is cluster class, col is the origin label, element is count number
                    if group_point_3state[i] in two_edge_state[c - 3]:
                        type_count[cluster_label, two_edge_state[c-3].index(group_point_3state[i])] += 1
                        if cluster_label not in temp_cluster_point_index.keys():
                            temp_cluster_point_index[cluster_label] = [group_point_index[i]]
                        else:
                            temp_cluster_point_index[cluster_label].append(group_point_index[i])
                peak_i = np.argmax(type_count, axis=1)

                for row, p_i in enumerate(peak_i):
                    state_change_dic_p_index[ two_edge_state[c-3][p_i] ] += temp_cluster_point_index[row]
        return state_change_dic_p_index

    def images_result_show(self, save_path=None):
        need_cluster_grid_map = np.zeros(self.down_sampling_size, dtype=np.int8)
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

        axes[2].imshow(self.mask + resize_image(need_cluster_grid_map, int(1/self.scale_factor)))
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
        plt.xlabel('cluster class-axis')
        plt.ylabel('depth-axis')
        plt.title('Scatter Plot of Points')
        plt.legend()
        plt.xticks(range(int(min(x)), int(max(x)) + 1, 1))
        plt.grid(True)
        plt.show()

    def grid_locate_visualizer(self, h, w, if_extension, depth_matrix):
        """show the position of key grid in the grid map"""
        grid = np.zeros(self.down_sampling_size, dtype=np.int8)
        grid[h, w] += 1
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))

        axes[0].imshow(self.exist_map + grid)  # cmap='gray'表示使用灰度颜色映射
        axes[0].set_title('grid located in grid map, grid crack {}'.format(if_extension))

        axes[1].imshow(self.mask + resize_image(grid, int(1/self.scale_factor)))
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
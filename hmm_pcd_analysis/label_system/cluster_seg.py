import numpy as np
from grid import GridUnit

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


down_sampling_size = [180, 240]  # 矩阵是HxW
state_standard = {
    0: [0, 1, 2],
    1: [[0, 1], [0, 2], [1, 2]],
    2: [[0, 1, 2]]
}
two_edge_state = [[0, 1], [0, 2], [1, 2]]


class ClusterSegmentSystem:
    def __init__(self):
        self.index_map = np.zeros(down_sampling_size, dtype=np.int16)  # down sampling
        self.exist_map = np.zeros(down_sampling_size, dtype=np.int8)  # down sampling
        self.index_numer = 1

        # COO matrix
        self.row_col = []  # [[20, 30], []]
        self.class_list = []  # [[0,0,0,0,0,1,0], ]
        self.grid_list = []  # main data

        self.need_cluster_grid = {3:[], 4:[], 5:[]}  # for one frame of image

    def grid_map_update(self, point_index, down_uv, one_point_obs, depth_c):
        """
            for every point projection
            basic task for cluster segmentation
        """
        w = int(down_uv[0])  # x related to w
        h = int(down_uv[1])
        if self.exist_map[h, w] == 0:  # new grid
            self.exist_map[h, w] = 1
            self.index_map[h, w] = self.index_numer
            self.index_numer += 1

            self.row_col.append([h, w])
            zero_list = [0, 0, 0, 0, 0, 0, 0]
            zero_list[one_point_obs] = 1
            self.class_list.append(zero_list)
            new_grid = GridUnit(h, w)
            new_grid.adjust_grid(point_index, one_point_obs, depth_c)
            self.grid_list.append(new_grid)
        else:
            self.class_list[self.index_map[h, w]-1][one_point_obs] = 1
            self.grid_list[self.index_map[h, w]-1].adjust_grid(point_index, one_point_obs, depth_c)

    def crack_detect(self, thread_hold=0.8):
        """
        for one frame of image
        Go through all key grid to find crack
        """
        for c in range(3, 6):  # key class 3, 4, 5
            for i, l in enumerate(self.class_list):  # i, l - grid index, one class_list
                if l[c] != 1:
                    continue
                h = self.row_col[i][0]
                w = self.row_col[i][1]
                temp_grid_index, near_key_grid_index_list, \
                    temp_class_mix, max_depth, min_depth = self.near_grid(h, w, c)

                max_d = max(max_depth)
                min_d = min(min_depth)
                if (max_d - min_d) > thread_hold:
                    # further more find the seg region
                    if temp_class_mix[state_standard[1][c-3][0]] > 0 \
                            and temp_class_mix[state_standard[1][c-3][1]] > 0:
                        self.need_cluster_grid[c].append(temp_grid_index)
                    else:  # extend
                        for g_index in near_key_grid_index_list:
                            h = self.row_col[g_index][0]
                            w = self.row_col[g_index][1]
                            _temp_grid_index, _near_key_grid_index_list, \
                                _temp_class_mix, _max_depth, _min_depth = self.near_grid(h, w, c)
                            temp_grid_index += _temp_grid_index
                            temp_class_mix += _temp_class_mix
                        temp_grid_index = list(set(temp_grid_index))
                        self.need_cluster_grid[c].append(temp_grid_index)

    def near_grid(self, h, w, key_state):
        temp_grid_index = []
        near_key_grid_index_list = []  # 临近的key grid
        temp_class_mix = np.zeros(7, dtype=np.int8)
        max_depth = []
        min_depth = []
        for row in range(-1, 2):
            for col in range(-1, 2):
                h_ = h + row
                w_ = w + col
                index = self.index_map[h_, w_] - 1
                if (0 <= h_ < down_sampling_size[0]) \
                        and (0 <= w_ < down_sampling_size[1]) \
                        and index != 0:
                    temp_grid_index.append(index)  # temporary grid index list
                    temp_class_mix += np.array(self.class_list[index])
                    max_depth.append(self.grid_list[index].max_depth)
                    min_depth.append(self.grid_list[index].min_depth)
                    if self.class_list[index][key_state] == 1:
                        near_key_grid_index_list.append(index)
        return temp_grid_index, near_key_grid_index_list, temp_class_mix, max_depth, min_depth

    def cluster_process(self, DBSCAN_eps=0.2, min_samples=3):
        """for one frame of image
        find the points, cluster the points, seg the points
        return: fix point dictionary"""
        state_change_dic_p_index = {0:[], 1:[], 2:[], "none": []}

        for c, group in self.need_cluster_grid.items():

            for g in group:
                # g is one cluster seg
                group_point_index = []
                group_point_depth = []
                group_point_state = []
                for i_grid in g:
                    group_point_index += self.grid_list[i_grid].point_index
                    group_point_depth += self.grid_list[i_grid].point_depth2c
                    group_point_state += self.grid_list[i_grid].point_state

                cluster_data = np.array(group_point_depth)
                cluster_data2D = cluster_data.reshape(-1, 1)
                cluster_data2D_matrix = squareform(pdist(cluster_data2D, metric='euclidean'))
                dbscan = DBSCAN(eps=DBSCAN_eps, min_samples=min_samples, metric='precomputed')
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
                    for j, l in enumerate(clusters_result):
                        state_change_dic_p_index["none"].append(group_point_index[j])
        return state_change_dic_p_index


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
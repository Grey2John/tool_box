import os


class GridUnit:
    def __init__(self, row, col, size=4):
        """the grid in image, for indexing the points"""
        self.row = row
        self.col = col
        self.max_depth = 0
        self.min_depth = 0
        self.point_state_flag = [0, 0, 0, 0, 0, 0, 0]
        self.class_count = [0, 0, 0, 0, 0, 0, 0]

        self.point_index = []  # the index of the points in this grid
        self.point_xydepth = []  # xyz_cam relate to point
        self.point_obs_state = []  # observing state
        self.point_3state = []  # direct state

    def adjust_grid(self, point_index, point_obs_state, label_state, xy, xydepth_c):
        """adjust each point projection"""
        self.point_index.append(point_index)
        self.point_xydepth.append(xydepth_c)  # xyz based on camera
        self.point_obs_state.append(point_obs_state)
        self.point_3state.append(label_state)
        self.point_state_flag[point_obs_state] = 1
        self.class_count[point_obs_state] += 1

        if self.max_depth == 0 and self.min_depth == 0:
            self.max_depth = xydepth_c[2]
            self.min_depth = xydepth_c[2]

        else:
            if xydepth_c[2] > self.max_depth:
                self.max_depth = xydepth_c[2]

            elif xydepth_c[2] < self.min_depth:
                self.min_depth = xydepth_c[2]

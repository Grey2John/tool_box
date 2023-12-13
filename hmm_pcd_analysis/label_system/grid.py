import os


class GridUnit:
    def __init__(self, row, col, size=4):
        """the grid in image, for indexing the points"""
        self.row = row
        self.col = col
        self.max_depth = 0
        self.max_pixel_hw = None  # 原尺寸像素
        self.min_depth = 0
        self.min_pixel_hw = None
        self.point_state_flag = [0, 0, 0, 0, 0, 0, 0]
        self.class_count = [0, 0, 0, 0, 0, 0, 0]

        self.point_index = []  # the index of the points in this grid
        self.point_xyz2c = []  # xyz_cam relate to point
        self.point_state = []  # depth relate to point

    def adjust_grid(self, point_index, point_obs_state, xy, xyz_c):
        """adjust each point projection"""
        self.point_index.append(point_index)
        self.point_xyz2c.append(xyz_c)  # xyz based on camera
        self.point_state.append(point_obs_state)
        self.point_state_flag[point_obs_state] = 1
        self.class_count[point_obs_state] += 1
        depth = xyz_c[-1]
        if self.max_depth == 0 and self.min_depth == 0:
            self.max_depth = depth
            self.min_depth = depth
            self.max_pixel_hw = xyz_c[0:2]
            self.min_pixel_hw = xyz_c[0:2]
        else:
            if depth > self.max_depth:
                self.max_depth = depth
                self.max_pixel_hw = xyz_c[0:2]
            elif depth < self.min_depth:
                self.min_depth = depth
                self.min_pixel_hw = xyz_c[0:2]

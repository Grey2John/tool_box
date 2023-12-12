import os


class GridUnit:
    def __init__(self, row, col):
        """the grid in image, for indexing the points"""
        self.row = row
        self.col = col
        self.max_depth = 0
        self.min_depth = 0
        self.point_state_flag = [0, 0, 0, 0, 0, 0, 0]
        self.class_count = [0, 0, 0, 0, 0, 0, 0]

        self.point_index = []  # the index of the points in this grid
        self.point_depth2c = []  # depth relate to point
        self.point_state = []  # depth relate to point

    def adjust_grid(self, point_index, point_obs_state, depth):
        """adjust each point projection"""
        self.point_index.append(point_index)
        self.point_index.append(depth)
        self.point_state.append(point_obs_state)
        self.point_state_flag[point_obs_state] = 1
        self.class_count[point_obs_state] += 1
        if self.max_depth == 0 and self.min_depth == 0:
            self.max_depth = depth
            self.min_depth = depth
        else:
            self.max_depth = max(self.max_depth, depth)
            self.min_depth = min(self.min_depth, depth)

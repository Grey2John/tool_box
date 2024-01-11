import os
import sys
from data_loader import PointDataLoader
import open3d as o3d
import numpy as np
from PIL import Image
from filter import PointList2RGBPCD as P2pcd

label_rgb = [[ 255, 0, 0],[0, 255, 0],[0, 0, 255],   # 0, 1, 2
             [255, 0, 255],[255, 255, 0],[0, 0, 0],   # 3, 4, 5
             [0, 255, 255]]   # 6
# label_rgb = [[0, 0, 255],[0, 0, 255],[0, 0, 255],   # 0, 1, 2
#              [0, 0, 255],[0, 0, 255],[0, 0, 255],   # 3, 4, 5
#              [0, 0, 255]]   # 6
state_standard_one = {
    0: [0],
    1: [1],
    2: [2],
    3: [0, 1],
    4: [0, 2],
    5: [1, 2],
    6: [0, 1, 2]
}


class Visualize:
    """单帧的图像的观测值的pcd"""
    def __init__(self, point_list, point_dic):
        self.point_list = point_list  #
        self.point_dic = point_dic

    def save_pcd_one_obs(self, save_path):
        points_k = sorted(self.point_dic.keys())
        for k in points_k:
            one_pcd_point = []
            for i in self.point_dic[k]:
                one_point = self.point_list[i][0:3]
                obs_index = 2
                for l in range(int(len(self.point_list[i][8:])/2)):
                    if int(self.point_list[i][8+2*l+1]) == k:
                        obs_index = 8+2*l
                        break
                color = label_rgb[self.point_list[i][obs_index]]
                one_point = one_point + color
                one_pcd_point.append(one_point)
            save_class = P2pcd(one_pcd_point)
            save_class.generate(save_path, str(k))


class OutlierFilter:
    def __init__(self, point_list):
        """input a point list [xyz, obs_state] for one time observation"""
        self.point_list = point_list

    def distance(self, point1, point2):
        return


def edge_mask(mask, pixel_set=4):
    rows, cols = mask.shape
    zero_matrix = np.zeros_like(mask)

    for h in range(rows):
        for w in range(cols):
            state = 0
            state_lib = []
            for i in range(-pixel_set, pixel_set + 1):
                for j in range(-pixel_set, pixel_set + 1):
                    if 0 <= (h + i) < rows and 0 <= (w + j) < cols:
                        state_pixel = mask[h + i, w + j]
                        if state_pixel not in state_lib:
                            state_lib.append(state_pixel)
            for j, v in state_standard_one.items():
                if v == sorted(state_lib):
                    state = j
                    break
            zero_matrix[h, w] = state
    return zero_matrix


def resize_image(original_matrix, scale_factor):
    original_height, original_width = original_matrix.shape
    new_height, new_width = original_height * scale_factor, original_width * scale_factor

    upscaled_matrix = np.zeros((new_height, new_width), dtype=original_matrix.dtype)

    for i in range(original_height):
        for j in range(original_width):
            value = original_matrix[i, j]
            upscaled_matrix[i * scale_factor:i * scale_factor + scale_factor,
                            j * scale_factor:j * scale_factor + scale_factor] = value

    return upscaled_matrix


if __name__ == "__main__":
    read_tool = PointDataLoader("/media/zlh/zhang/dataset/outline_seg_slam/test1/pt_obs.txt")
    point, point_index = read_tool.read_txt_dic_points_with_obs_times()
    vpcd = Visualize(point, point_index)
    save_path = '/media/zlh/zhang/dataset/outline_seg_slam/test1/origin_pcd'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    vpcd.save_pcd_one_obs(save_path)

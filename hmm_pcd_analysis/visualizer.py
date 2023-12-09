import os
import sys
from data_loader import PointDataLoader
import open3d as o3d
import numpy as np
from filter import PointList2RGBPCD as P2pcd

label_rgb = [[255, 0, 0],[0, 255, 0],[0, 0, 255],   # 0, 1, 2
             [255, 0, 255],[255, 255, 0],[0, 0, 0],   # 3, 4, 5
             [0, 255, 255]]   # 6
# label_rgb = [[0, 0, 255],[0, 0, 255],[0, 0, 255],   # 0, 1, 2
#              [0, 0, 255],[0, 0, 255],[0, 0, 255],   # 3, 4, 5
#              [0, 0, 255]]   # 6


class Visualize:
    def __init__(self, point_list, point_dic):
        self.point_list = point_list
        self.point_dic = point_dic

    def save_pcd_one_obs(self, sava_path):
        for k, v in self.point_dic.items():
            one_pcd_point = []
            for i in v:
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
            save_class.generate(sava_path, str(k))


class OutlierFilter:
    def __init__(self, point_list):
        """input a point list [xyz, obs_state] for one time observation"""
        self.point_list = point_list

    def distance(self, point1, point2):
        return


if __name__ == "__main__":
    read_tool = PointDataLoader("F:\earth_rosbag\data\\test3\obs_times_txt\\bag11.txt")
    point, point_index = read_tool.read_txt_dic_points_with_obs_times()
    vpcd = Visualize(point, point_index)
    save_path = 'F:\earth_rosbag\data\\test3\obs_times_txt\\bag11'
    vpcd.save_pcd_one_obs(save_path)

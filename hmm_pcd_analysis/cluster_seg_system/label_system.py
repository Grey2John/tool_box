import os
import sys
import time
import cv2
# import rosbag
# from cv_bridge import CvBridge
import numpy as np
import matplotlib.pyplot as plt

from cluster_seg import ClusterSegmentSystem
from filter import PointList2RGBPCD as P2pcd
from filter import LabelPCD
"""
整个观测系统的程序
loading data
data process
"""
# m_cam_K = np.array([[907.3834838867188,  0.0, 644.119384765625],
#                   [0.0, 907.2291870117188, 378.90472412109375],
#                   [0.0, 0.0, 1.0]])
mask_size = [1280, 720]
pixel_set = 4
state_standard = {
    0: [0, 1, 2],
    1: [[0, 1], [0, 2], [1, 2]],
    2: [[0, 1, 2]]
}
state_standard_one = {
    0: [0],
    1: [1],
    2: [2],
    3: [0, 1],
    4: [0, 2],
    5: [1, 2],
    6: [0, 1, 2]
}
two_edge_state = [[0, 1], [0, 2], [1, 2]]
label_rgb = [[255, 0, 0],[0, 255, 0],[0, 0, 255],   # 0, 1, 2
             [255, 0, 255],[255, 255, 0],[0, 0, 0],   # 3, 4, 5
             [0, 255, 255]]   # 6


class Point:
    def __init__(self, xyz, hmm, annotated_label=None):
        self.coordinate = np.array(xyz)
        self.filter_prob = np.array(hmm)
        self.obs_state = None
        self.truth_label = annotated_label
        self.if_obs = True
        self.obs_times = 0

    def update_tims(self, max_time=100):  # reduce observation
        self.obs_times += 1
        if self.obs_times >= max_time:
            self.if_obs = False


class ImagePose:
    """ one mask image
    save mask, points index; for projection
    """
    def __init__(self, mask, observing_num, pose_r, pose_t, cam_k, image_seq):
        self.mask = mask  # 960*720
        self.observing_num = observing_num
        self.pose_r = pose_r
        self.pose_t = pose_t
        self.cam_k = np.array([ [cam_k[0], 0, cam_k[2]],
                                [0, cam_k[1], cam_k[3]],
                                [0, 0, 1]])  # [fx, fy, cx, cy] to matrix
        self.image_seq = image_seq
        self.point_index_list = []

    def observe_state(self, h, w):
        state = 0
        state_lib = []
        for i in range(-pixel_set, pixel_set+1):
            for j in range(-pixel_set, pixel_set + 1):
                if 0 <= (h + i) < mask_size[1] and 0 <= (w + j) < mask_size[0]:
                    state_pixel = self.mask[h + i, w + j]
                    if state_pixel not in state_lib:
                        state_lib.append(state_pixel)
        for j, v in enumerate(state_standard[len(state_lib) - 1]):
            if isinstance(v, (int, float)):
                v = [v]
            if v == sorted(state_lib):
                state = (len(state_lib) - 1)*3 + j
        return state


class ImagePoseDic:
    """multi image from camera
    multi cluster
    """
    def __init__(self, _intrinsic_scale):
        self.intrinsic_scale = _intrinsic_scale
        self.image_dic = {}  # {frame_id: ImagePose}
        self.point_list_lib = []  # [xyz, p0p1p2-hmm, obs_state] Class Point
        self.one_CSS = None  # clustering class for one frame

        self.seq2num = {}   # to find the observing number

    def add_image_frame(self, image_pose):
        self.image_dic[image_pose.observing_num] = image_pose

    def project_3Dpoints_and_grid_build(self, frame, scale_factor=0.25):
        proj_state_count = 0
        for index_p in self.image_dic[frame].point_index_list:  # every point in each frame
            if self.point_list_lib[index_p].if_obs:
                xy, xyz_c = project_3d_point_in_img(self.point_list_lib[index_p].coordinate,
                                                      self.image_dic[frame].cam_k,
                                                      self.image_dic[frame].pose_r,
                                                      self.image_dic[frame].pose_t)
                if xy is False:
                    self.point_list_lib[index_p].obs_state = None
                    continue
                proj_state_count += 1
                self.point_list_lib[index_p].update_tims()
                xy_int = np.round(xy).astype(int)
                obs_state = self.image_dic[frame].observe_state(xy_int[1], xy_int[0])  # observing state [y, x]
                # 先y后x，这是因为在NumPy中，数组的第一个索引对应于行，第二个索引对应于列]
                self.point_list_lib[index_p].obs_state = obs_state  # point state from first obs
                self.one_CSS.grid_map_update(index_p, obs_state, xy, xyz_c)

        print("the number of project points is {}".format(proj_state_count))

    def point_state_optim(self, fix_dic):
        for k, v in fix_dic.items():
            if k == "none":
                state = None
            else:
                state = k
            for i in v:
                self.point_list_lib[i].obs_state = state

    def one_frame_pcd_data_generate(self, f):
        points_obs = []
        for index_p in self.image_dic[f].point_index_list:
            state = self.point_list_lib[index_p].obs_state
            if state is not None:
                one_point = self.point_list_lib[index_p].coordinate.tolist()
                color = label_rgb[state]
                one_p = one_point + color
                points_obs.append(one_p)
        return points_obs

    def process(self, save_path):
        """main function to process every image"""
        frame_list = sorted(self.image_dic.keys())
        for f in frame_list:
            print("\033[32m===== image No. is {} =====\033[0m".format(f))
            start_time = time.time()
            self.one_CSS = ClusterSegmentSystem(self.image_dic[f].mask)  # init  self.image_dic[f].mask
            self.project_3Dpoints_and_grid_build(f)
            self.one_CSS.crack_detect()
            state_change_dic = self.one_CSS.cluster_process()
            self.point_state_optim(state_change_dic)  # fix point state
            print("frame {} cost {} s".format(f, (time.time() - start_time)))
            cpd_start_time = time.time()

            points_obs = self.one_frame_pcd_data_generate(f)
            pcd_generate(points_obs, f, save_path, "cluster_")
            print("pcd generate {} cost {} s".format(f, (time.time() - cpd_start_time)))
            # image_visual = self.image_dic[frame].mask
            image_visual = self.one_CSS.exist_map
            # process_visualizer(image_visual)
            # user_input = input("按下N键进入下一个循环，按下其他键退出：")
            # if user_input.lower() != 'n':
            #     break


def pcd_generate(points_obs, frame, save_path, prefix=None):
    # points_obs [xyz, rgb]
    non_filter_name = "{}{}".format(prefix, frame)
    save_class = P2pcd(points_obs)
    save_class.generate(save_path, non_filter_name)
    print("generate the {}".format(non_filter_name))


def process_visualizer(image_visual):
    plt.imshow(image_visual)
    plt.draw()  # 绘制图像
    plt.pause(0.05)


def project_3d_point_in_img(xyz, K, pose_r, pose_t):
    # xyz_1 = np.append(xyz, 1)
    camera_point = np.dot(pose_r, xyz) + pose_t
    # depth = camera_point[-1]
    # if depth < 0.001:
    #     return False, False
    image_xy = np.dot(K, camera_point)
    u_v = image_xy[:2] / image_xy[2]  # [x, y] [heng, zong] [w, h] [col, row]
    cut_uv = u_v-np.array([160, 0])
    if 2 <= cut_uv[0] < 958 and 2 <= cut_uv[1] < 718:  # available observing region
        return cut_uv, camera_point  # not Rounding for keeping accuracy
    else:
        return False, False


def one_frame_task(image_pose, points, frame, point_origin_state, save_path, scale_factor=0.25):
    """one frame task"""
    IPD = ImagePoseDic(1.0)
    IPD.add_image_frame(image_pose)
    IPD.point_list_lib = points
    IPD.image_dic[frame].point_index_list = list(range(len(points)))
    # start
    IPD.process(save_path)
    pcd = LabelPCD(point_origin_state)  # origin pcd
    pcd.generate(save_path, str(frame))

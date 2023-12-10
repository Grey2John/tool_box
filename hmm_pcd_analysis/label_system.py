import os
import sys
import time
import cv2
# import rosbag
# from cv_bridge import CvBridge
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation
from data_loader import PointDataLoader

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
"""
整个观测系统的程序
loading data
data process
"""
m_cam_K = np.array([[907.3834838867188, 0.0, 644.119384765625],
                  [0.0, 907.2291870117188, 378.90472412109375],
                  [0.0, 0.0, 1.0]])
intrinsic_scale = 1.0
filter_init = [0.4, 0.3, 0.3]
down_sampling_size = [180, 240]  # 矩阵是HxW
mask_size = [1280, 720]
pixel_set = 4
state_standard = {
    0: [0, 1, 2],
    1: [[0, 1], [0, 2], [1, 2]],
    2: [[0, 1, 2]]
}


class Point:
    def __init__(self, xyz, hmm):
        self.coordinate = np.array(xyz)
        self.filter_prob = np.array(hmm)
        self.obs_state = None
        self.if_obs = True
        self.obs_times = 0

    def update_tims(self, max_time=100):  # reduce observation
        self.obs_times += 1
        if self.obs_times >= max_time:
            self.if_obs = False


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

    def crack_detect(self, thread_hold=2):
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
        state_change_dic_p_index = {0:[], 1:[], 2:[]}

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
                # [0  0  0  0  0  1  1  1  1 -1  2  2  2  2  2]
        return state_change_dic_p_index


class ImagePose:
    """ one mask image
    save mask, points index; for projection
    """
    def __init__(self, mask, frame_id, pose_r, pose_t):
        self.mask = mask  # 960*720
        self.frame_num = frame_id
        self.pose_r = pose_r
        self.pose_t = pose_t
        self.point_index_list = []

    def observe_state(self, h, w):
        state = 0
        state_lib = []
        for i in range(-pixel_set, pixel_set+1):
            for j in range(-pixel_set, pixel_set + 1):
                if 0 <= (h + i) < mask_size[1] and 0 <= (w + j) < mask_size[0]:
                    if self.mask[h + i, w + j] not in state_lib:
                        state_lib.append(self.mask[h + i, w + j])
        for k, v in enumerate(state_standard[len(state_lib) - 1]):
            if v == sorted(state_lib):
                state = (len(state_lib) - 1)*3 + k
        return state


class ImagePoseDic:
    """multi image for a camera"""
    def __init__(self, _intrinsic_scale, _m_cam_K):
        self.m_cam_K = _m_cam_K
        self.intrinsic_scale = _intrinsic_scale
        self.image_dic = {}  # {frame_id: ImagePose}
        self.point_list_lib = []  # [xyz, p0p1p2-hmm, obs_state]
        self.one_CSS = None  # clustering

    def add_image_frame(self, image_pose):
        self.image_dic[image_pose.frame_num] = image_pose

    def project_3Dpoints_and_grid_build(self, frame, scale_factor=0.25):
        proj_state_count = 0
        for index_p in self.image_dic[frame].point_index_list:  # every point in each frame
            if self.point_list_lib[index_p].if_obs:
                xy, depth_c = project_3d_point_in_img(self.point_list_lib[index_p].coordinate, self.m_cam_K,
                                                      self.image_dic[frame].pose_r, self.image_dic[frame].pose_t)
                if xy is False:
                    continue
                proj_state_count += 1
                self.point_list_lib[index_p].update_tims()
                xy_int = np.round(xy).astype(int)
                obs_state = self.image_dic[frame].observe_state(xy_int[1], xy_int[0])  # observing state [y, x]
                # 先y后x，这是因为在NumPy中，数组的第一个索引对应于行，第二个索引对应于列
                self.point_list_lib[index_p].obs_state = obs_state  # point state from first obs
                down_xy = np.round(xy * scale_factor).astype(int)
                self.one_CSS.grid_map_update(index_p, down_xy, obs_state, depth_c)

        print("the number of project points is {}".format(proj_state_count))
        return None

    def process(self):
        """main function to process every image"""
        frame_list = sorted(self.image_dic.keys())
        for f in frame_list:
            print("=== image No. is {} ===".format(f))
            start_time = time.time()
            self.one_CSS = ClusterSegmentSystem()  # init
            self.project_3Dpoints_and_grid_build(f)
            self.one_CSS.crack_detect()
            state_change_dic = self.one_CSS.cluster_process()
            end_time = time.time()
            print("frame {} cost {} s".format(f, (end_time - start_time)))
            # visualizer
            # image_visual = self.image_dic[f].mask
            # plt.imshow(image_visual)
            # plt.draw()  # 绘制图像
            # plt.pause(0.001)
            # user_input = input("按下N键进入下一个循环，按下其他键退出：")
            # if user_input.lower() != 'n':
            #     break


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


class OutlineDataLoader:
    def __init__(self, dir_path):
        """ load the outline data from r3live"""
        if not os.listdir(dir_path):
            print("you input the wrong dir path")
            sys.exit()
        self.bag_path = os.path.join(dir_path, "yolo.bag")
        self.obs_txt_path = os.path.join(dir_path, "pt_obs.txt")
        self.image_pose_txt_path = os.path.join(dir_path, "pt_obs_image_pose.txt")
        self.mask_image_dir_path = os.path.join(dir_path, "mask")

        self.image_pose_dic = ImagePoseDic(intrinsic_scale, m_cam_K)

    def bag_read(self):
        """complete image: mask in each ImagePose"""
        start_time = time.time()
        if not os.path.exists(self.bag_path):
            return False
        bridge = CvBridge()
        topic_name = "/yolov5/segment/bgra"
        with rosbag.Bag(self.bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                # if msg._type == 'sensor_msgs/Image':
                frame_id = msg.header.seq
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='rgba8')
                alpha_channel = cv2.split(cv_image)[3]
                if frame_id in self.image_pose_dic.image_dic.keys():
                    self.image_pose_dic.image_dic[frame_id].mask = np.array(alpha_channel)  # 1280*720

        end_time = time.time()
        print("2. finish the bag read")
        print("程序执行时间：{} 秒".format(end_time - start_time))

    def point_read(self):
        """complete the point_index_list in each image_pose"""
        start_time = time.time()
        if not os.path.exists(self.obs_txt_path):
            return False
        points = PointDataLoader(self.obs_txt_path)
        _point_list_source, _points_frame_dic = points.read_txt_dic_points_with_obs_times()
        for p in _point_list_source:
            one_point = Point(p[0:3], filter_init)
            self.image_pose_dic.point_list_lib.append(one_point)

        for frame, dic in _points_frame_dic.items():
            if frame in self.image_pose_dic.image_dic.keys():
                self.image_pose_dic.image_dic[frame].point_index_list = dic

        end_time = time.time()
        print("3. finish the points pose read")
        print("程序执行时间：{} 秒".format(end_time - start_time))

    def image_pose_read(self):
        start_time = time.time()
        if not os.path.exists(self.image_pose_txt_path):
            return False
        with open(self.image_pose_txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                one_list = line.strip().split(', ')
                # [num, xyz, w, x, y, z]
                float_data = [float(item) for item in one_list[2:]]
                R = Rotation.from_quat(float_data[3:])
                t = np.array(float_data[0:3])
                # T = np.eye(4)
                # T[:3, :3] = R.as_matrix()
                # T[:3, 3] = t
                one_frame = ImagePose(None, int(one_list[1]), R.as_matrix(), t)  # the 2.st is the image No.
                self.image_pose_dic.add_image_frame(one_frame)
        end_time = time.time()
        print("1. finish the image pose read")
        print("程序执行时间：{} 秒".format(end_time - start_time))

    def image_file_read(self, file_type=".png"):
        start_time = time.time()
        if not os.path.exists(self.mask_image_dir_path):
            print(f"Error: Folder '{self.mask_image_dir_path}' not found.")
            return False

        file_list = os.listdir(self.mask_image_dir_path)
        for file_name in file_list:
            if file_name.lower().endswith(file_type):
                file_path = os.path.join(self.mask_image_dir_path, file_name)
                frame_id = int(file_name.split(".")[0])
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is not None and frame_id in self.image_pose_dic.image_dic.keys():
                    img = image[:, 159:1120]
                    self.image_pose_dic.image_dic[frame_id].mask = img
        end_time = time.time()
        print("2. finish the image read")
        print("程序执行时间：{} 秒".format(end_time - start_time))

    def data_out_put(self):
        """main"""
        self.image_pose_read()
        # self.bag_read()
        self.image_file_read()
        self.point_read()
        return self.image_pose_dic


def rosbag_rgba_to_image(bag, save_path, rgb_name="rgb", mask_name="mask"):
    """change the bag to the RGB image and mask"""
    rgb_dir = os.path.join(save_path, rgb_name)
    mask_dir = os.path.join(save_path, mask_name)
    if not os.path.exists(rgb_dir):
        os.makedirs(rgb_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    bridge = CvBridge()
    with rosbag.Bag(bag, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            # Assuming the image message is of type sensor_msgs/Image
            if topic == "/yolov5/segment/bgra":
                frame_id = msg.header.seq
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgra8')
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2RGB)
                alpha_channel = cv2.split(cv_image)[3]

                rgb_path = os.path.join(rgb_dir, f"{frame_id}.jpg")
                print(rgb_path)
                cv2.imwrite(rgb_path, rgb_image)
                mask_path = os.path.join(mask_dir, f"{frame_id}.png")
                print(mask_path)
                cv2.imwrite(mask_path, alpha_channel)
                frame_id += 1


def project_3d_point_in_img(xyz, K, pose_r, pose_t):
    # xyz_1 = np.append(xyz, 1)
    camera_point = np.dot(pose_r, xyz) + pose_t
    image_xyz = np.dot(K, camera_point)
    depth = image_xyz[-1]
    u_v = image_xyz[:2] / image_xyz[2]
    cut_uv = u_v-np.array([0, 159])
    if 2 <= cut_uv[0] < 718 and 2 <= cut_uv[1] < 958:  # available observing region
        return cut_uv, depth  # not Rounding for keeping accuracy
    else:
        return False, False


if __name__ == "__main__":
    data = OutlineDataLoader("/media/zlh/zhang/dataset/outline_seg_slam/test1")
    all_image_data = data.data_out_put()
    del data
    print("\033[32m finish all data loading \033[0m")
    # all_image_data processing
    all_image_data.process()

    # rosbag_rgba_to_image('/media/zlh/zhang/dataset/outline_seg_slam/test1/yolo.bag',
    #                      '/media/zlh/zhang/dataset/outline_seg_slam/test1')

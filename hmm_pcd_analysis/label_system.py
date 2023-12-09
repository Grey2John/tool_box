import os
import sys
import time
import cv2
# import rosbag
# from cv_bridge import CvBridge
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation
from data_loader import PointDataLoader
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
down_sampling_size = [320, 180]
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

    def update_tims(self):  # reduce observation
        self.obs_times += 1
        if self.obs_times >= 100:
            self.if_obs = False


class ImagePose:
    """ one mask image, one frame, one_loop"""
    def __init__(self, mask, frame_id, pose_r, pose_t):
        self.mask = mask  # 1280*720
        self.frame_num = frame_id
        self.pose_r = pose_r
        self.pose_t = pose_t
        self.point_index_list = []

        self.index_map = np.zeros(down_sampling_size)  # down sampling
        self.exist_map = np.zeros(down_sampling_size)  # down sampling
        self.key_grid_map = np.zeros(down_sampling_size)  # down sampling
        self.index_numer = 0
        self.grid_list = []

    def grid_update(self, down_uv):
        w = down_uv[0]  # x related to w
        h = down_uv[1]
        if self.exist_map[h, w] == 1:
            self.grid_list[self.index_map[h, w]].adjust_grid()
        else:
            self.exist_map[h, w] = 1
            self.index_map[h, w] = self.index_numer
            self.index_numer += 1
            new_grid = GridUnit(h, w)
            self.grid_list.append(new_grid)

    def observe_state(self, h, w):
        state = 0
        state_lib = []
        for i in range(-pixel_set, pixel_set+1):
            for j in range(-pixel_set, pixel_set + 1):
                if 0 <= (h + i) < mask_size[1] and 0 <= (w + j) < mask_size[0]:
                    if self.mask[h + i, w + j] not in state_lib:
                        state_lib.append(self.mask[h + i, w + j])
        for k, w in enumerate(state_standard[len(state_lib) - 1]):
            if w == sorted(state_lib):
                state = (len(state_lib) - 1)*3 + k

        return state


class ImagePoseDic:
    """multi image for a camera"""
    def __init__(self, _intrinsic_scale, _m_cam_K):
        self.m_cam_K = _m_cam_K
        self.intrinsic_scale = _intrinsic_scale
        self.image_dic = {}  # {frame_id: ImagePose}
        self.point_list_lib = []  # [xyz, p0p1p2-hmm, obs_state]

    def add_image_frame(self, image_pose):
        self.image_dic[image_pose.frame_num] = image_pose

    def project_3Dpoints_and_grid_build(self, frame, scale_factor=0.25):
        proj_count = 0
        for index_p in self.image_dic[frame].point_index_list:
            if self.point_list_lib[index_p].if_obs:
                proj_count += 1
                xy = project_3d_point_in_img(self.point_list_lib[index_p].coordinate, self.m_cam_K,
                                             self.image_dic[frame].pose_r, self.image_dic[frame].pose_t)
                self.point_list_lib[index_p].update_tims()
                xy_int = np.round(xy).astype(int)
                obs_state = self.image_dic[frame].observe_state(xy_int[1], xy_int[0])  # observing state [y, x]
                # 先y后x，这是因为在NumPy中，数组的第一个索引对应于行，第二个索引对应于列
                # grid type
                down_xy = np.round(xy * scale_factor).astype(int)
                self.image_dic[frame].grid_update(down_xy)
        print("the number of project points is {}".format(proj_count))
        return None

    def process(self):
        """main function to process every image"""
        frame_list = sorted(self.image_dic.keys())
        for f in frame_list:
            print("=== image No. is {} ===".format(f))
            start_time = time.time()
            self.project_3Dpoints_and_grid_build(f)
            end_time = time.time()
            print("frame {} cost {} s".format(f, (end_time - start_time)))


class GridUnit:
    def __init__(self, row, col):
        """the grid in image, for indexing the points"""
        self.row = row
        self.col = col
        self.max_depth = 0
        self.min_depth = 0
        self.point_state_list = [0, 0, 0, 0, 0, 0, 0]
        self.class_count = [0, 0, 0, 0, 0, 0, 0]

    def adjust_grid(self, ):

        return None


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
                frame_id = int(msg.header.frame_id)
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
                float_data = [float(item) for item in one_list[1:]]
                R = Rotation.from_quat(float_data[3:])
                t = np.array(float_data[0:3])
                # T = np.eye(4)
                # T[:3, :3] = R.as_matrix()
                # T[:3, 3] = t
                one_frame = ImagePose(None, int(one_list[0]), R.as_matrix(), t)
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
                    # img = np.array(image)
                    self.image_pose_dic.image_dic[frame_id].mask = image
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
    frame_id = 0
    with rosbag.Bag(bag, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            # Assuming the image message is of type sensor_msgs/Image
            if topic == "/yolov5/segment/bgra":
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
    u_v = image_xyz[:2] / image_xyz[2]
    return u_v  # not Rounding for keeping accuracy


if __name__ == "__main__":
    data = OutlineDataLoader("/media/zlh/zhang/dataset/outline_seg_slam/test1")
    all_image_data = data.data_out_put()
    del data
    print("\033[32m finish all data loading \033[0m")
    # all_image_data processing
    all_image_data.process()

    # rosbag_rgba_to_image('/media/zlh/zhang/dataset/outline_seg_slam/test1/yolo.bag',
    #                      '/media/zlh/zhang/dataset/outline_seg_slam/test1')

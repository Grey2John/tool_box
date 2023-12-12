import os
import sys
import time
import numpy as np
import cv2
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from data_loader import PointDataLoader
from label_system import ImagePoseDic, Point, ImagePose

intrinsic_scale = 1.0
filter_init = [0.4, 0.3, 0.3]
m_cam_K = np.array([907.3834838867188,907.2291870117188,644.119384765625,378.90472412109375])


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

        self.image_pose_dic = ImagePoseDic(intrinsic_scale)

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
                one_list = line.strip().split(', ')  # [num, image_seq, xyz, w, x, y, z]
                float_pose_data = [float(item) for item in one_list[2:2+7]]
                R = Rotation.from_quat(float_pose_data[3:])
                t = np.array(float_pose_data[0:3])
                float_cam_k_data = [float(item) for item in one_list[9:]]
                # T = np.eye(4)
                # T[:3, :3] = R.as_matrix()
                # T[:3, 3] = t
                one_frame = ImagePose(None, int(one_list[0]), R.as_matrix(),
                                      t, float_cam_k_data, int(one_list[1]))
                # the 2.st is the image No.
                self.image_pose_dic.add_image_frame(one_frame)
                self.image_pose_dic.seq2num[int(one_list[1])] = int(one_list[0])
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
                seq = int(file_name.split(".")[0])
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if seq in self.image_pose_dic.seq2num.keys():
                    observing_num = self.image_pose_dic.seq2num[seq]
                    if image is not None and observing_num in self.image_pose_dic.image_dic.keys():
                        # img = image[:, 160:1120]
                        img = image
                        self.image_pose_dic.image_dic[observing_num].mask = img
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

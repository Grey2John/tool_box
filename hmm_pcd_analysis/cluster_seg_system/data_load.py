import os
import sys
import time
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
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
        self.point_truth_label_path = os.path.join(dir_path, "labeled_pcd.pcd")

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
        if not os.path.exists(self.point_truth_label_path):
            return False

        points = PointDataLoader(self.obs_txt_path)
        self.point_list_source, points_frame_dic = points.read_txt_dic_points_with_obs_times()
        """
        point_list_source: [xyz, rgb, no, first, obs, frame_id]
        points_frame_dic: [143: [point index]]
        """
        label_truth = read_truth_labeled_rgb_pcd(self.point_truth_label_path)
        annotated_points = point_sample_labeling(label_truth, self.point_list_source, 8)
        """annotated_points: [[xyz, truth, obs, frame], ]"""
        for p in annotated_points:
            one_point = Point(p[0:3], filter_init, p[3])
            self.image_pose_dic.point_list_lib.append(one_point)

        for frame, dic in points_frame_dic.items():
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


def save_frame_data_json(workspace, frame_list, have_truth=False):
    """ pick one frame out
    input: frame list [20, 124, 122]
    save: point list, mask, pose_r, pose_t. the point in point list has annotated label
    save type: json
    """
    save_dir = os.path.join(work_space, "one_frame")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not isinstance(frame_list, list):
        frame_list = [frame_list]
    data = OutlineDataLoader(workspace)
    data.data_out_put()

    for f in frame_list:
        if f not in data.image_pose_dic.image_dic.keys():
            print("{} is not in the frame list".format(f))
            continue
        json_path = os.path.join(save_dir, "{}.json".format(f))
        mask = data.image_pose_dic.image_dic[f].mask.tolist()
        pose_r = data.image_pose_dic.image_dic[f].pose_r.tolist()
        pose_t = data.image_pose_dic.image_dic[f].pose_t.tolist()
        cam_k_ = data.image_pose_dic.image_dic[f].cam_k.tolist()
        cam_k = [cam_k_[0][0], cam_k_[1][1], cam_k_[0][2], cam_k_[1][2]]
        one_frame_json = {"point_list": None,
                          "mask": mask,
                          "pose_r": pose_r,
                          "pose_t": pose_t,
                          "cam_k": cam_k
                          }

        points = []  # [xyz, obs_state, truth label]
        for p_index in data.image_pose_dic.image_dic[f].point_index_list:
            xyz = data.image_pose_dic.point_list_lib[p_index].coordinate.tolist()
            origin_point_list = data.point_list_source[p_index]
            obs_index = None
            for l in range(int(len(origin_point_list[8:]) / 2)):
                if int(origin_point_list[8 + 2 * l + 1]) == f:
                    obs_index = 8 + 2 * l
                    break
            xyz.append(origin_point_list[obs_index])
            xyz.append(data.image_pose_dic.point_list_lib[p_index].truth_label)
            points.append(xyz)
        one_frame_json['point_list'] = points
        with open(json_path, 'w') as ff:
            json.dump(one_frame_json, ff)
            # json.dump(one_frame_json, ff, indent=4)
        print("save the file: {}".format(json_path))


def read_one_frame_data_json(json_path):
    with open(json_path, 'r') as ff:
        one_frame_json = json.load(ff)

    frame_num = int(json_path.split("/")[-1].split(".")[0])
    point_list = []
    point_origin_state = []
    for p in one_frame_json["point_list"]:
        one_point = Point(p[0:3], filter_init, p[4])  # add truth label
        point_list.append(one_point)
        point_origin_state.append(p[0:4])

    mask = np.array(one_frame_json["mask"])
    pose_r = np.array(one_frame_json["pose_r"])
    pose_t = np.array(one_frame_json["pose_t"])
    cam_k = one_frame_json["cam_k"]
    image_pose = ImagePose(mask, frame_num, pose_r, pose_t, cam_k, None)
    return image_pose, point_list, point_origin_state


def read_truth_labeled_rgb_pcd(pcd_path):
    """
    read labled pcd,
     generate the rgb pcd of labeled point cloud (optional)
     VERSION .7
    FIELDS x y z rgb label object
    SIZE 4 4 4 4 4 4
    TYPE F F F I I I
    COUNT 1 1 1 1 1 1
    WIDTH 102024
    HEIGHT 1
    POINTS 102024
    VIEWPOINT 0 0 0 1 0 0 0
    DATA ascii
    14.452689170837402 -4.594688415527344 2.1957247257232666 5069928 1 -1
    ...
    xyz, xxx, label, x
     """
    with open(pcd_path, 'r') as f:
        lines = f.readlines()

    # 获取点云数据的起始行
    for i, line in enumerate(lines):
        if line.startswith('DATA'):
            start_line = i + 1
            break
    # read point cloud
    data = []
    for line in lines[start_line:]:
        x, y, z, rgb, label, obj = line.split()
        data.append([float(x), float(y), float(z), int(label)-1])  # label start from 1
    print(data[5])
    return data


def point_sample_labeling(truth_label_points, sample_points, obs_start_index):
    """
    send a truth label for each point, if not exist, send None
    truth_label_points = [[xyz, truth label], ]
    sample_points = [xyz, ...]
    obs_start_index is the observing number stating index
    """
    points = []  # x, y, z, true label, obs_list
    origin_label_points_dir = {}
    for p in truth_label_points:
        o_name = ""
        for i in range(3):
            o_name = o_name + str(p[i])[:5]
        origin_label_points_dir[o_name] = p[3]  # true label

    none_count = 0
    for s in sample_points:  # [xyz, num, init_label, obs]
        s_name = ""
        for i in range(3):
            s_name = s_name + str(s[i])[:5]
        new_p = []
        new_p += s[0:3]  # xyz
        try:
            new_p.append(origin_label_points_dir[s_name])  # ground truth
            # new_p.append(s[3])  # first label
        except:
            new_p.append(None)
            none_count += 1
        new_p += s[obs_start_index:]
        points.append(new_p)

    print("we get {} matched points, {} points' label is None".format(len(points), none_count))
    return points  # x, y, z, true label, obs_list, frame list


if __name__ == "__main__":
    work_space = "/media/zlh/zhang/dataset/outline_seg_slam/bag2"
    frames = list(range(15, 300, 20))
    save_frame_data_json(work_space, frames)

    # obj, points, point_origin_state = read_one_frame_data_json("/media/zlh/zhang/dataset/outline_seg_slam/test1/one_frame/30.json")
    # image_visual = obj.mask
    # plt.imshow(image_visual)
    # plt.draw()  # 绘制图像
    # plt.waitforbuttonpress()

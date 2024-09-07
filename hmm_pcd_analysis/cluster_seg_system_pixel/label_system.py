import os
import time
# import rosbag
# from cv_bridge import CvBridge
import numpy as np

from cluster_seg import ClusterSegmentSystem
from pcd_gen import PointList2RGBPCD as P2pcd
from commen import add_save_list2txt, save_list2txt
from evaluation import eval_one_times, PointHMMEvaluation

"""
整个观测系统的程序
loading data
data process
"""
# m_cam_K = np.array([[907.3834838867188,  0.0, 644.119384765625],
#                   [0.0, 907.2291870117188, 378.90472412109375],
#                   [0.0, 0.0, 1.0]])
mask_size = [960, 720]
state_standard = {
    0: [0, 1, 2],
    1: [[0, 1], [0, 2], [1, 2]],
    2: [[0, 1, 2]]
}
edge_det = [
    {0: 0, 1: 1, 2: 2},
    {1: 3, 2: 4, 3: 5},
    {3: 6}
]
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
label_rgb = [[255, 0, 0], [0, 255, 0], [0, 0, 255],  # 0, 1, 2
             [255, 0, 255], [255, 255, 0], [100, 100, 100],  # 3, 4, 5
             [0, 255, 255]]  # 6


class Point:
    def __init__(self, xyz, hmm, annotated_label=None):
        self.coordinate = np.array(xyz)
        self.filter_prob = np.array(hmm)
        self.first_state = None
        self.obs_state = None  # point observing have edge detect
        self.label_state = None  # without edge detect, directly obs
        self.truth_label = annotated_label  # read the null will become None
        self.if_obs = True
        self.obs_times = 0

        self.obs_list = []
        self.opti_obs_list = []

    def update_tims(self, obs_state, label_state, max_time=150):  # reduce observation
        if self.obs_times == 0:
            self.first_state = label_state
        self.obs_times += 1
        self.obs_state = obs_state
        self.label_state = label_state
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
        self.cam_k = np.array([[cam_k[0], 0, cam_k[2]],
                               [0, cam_k[1], cam_k[3]],
                               [0, 0, 1]])  # [fx, fy, cx, cy] to matrix
        self.image_seq = image_seq
        self.point_index_list = []

    # def observe_state(self, h, w):
    #     state = 0
    #     state_lib = []
    #     label_state = self.mask[h, w]
    #     for i in range(-pixel_set, pixel_set+1):
    #         for j in range(-pixel_set, pixel_set + 1):
    #             if 0 <= (h + i) < mask_size[1] and 0 <= (w + j) < mask_size[0]:
    #                 state_pixel = self.mask[h + i, w + j]
    #                 if state_pixel not in state_lib:
    #                     state_lib.append(state_pixel)
    #     for j, v in enumerate(state_standard[len(state_lib) - 1]):
    #         if isinstance(v, (int, float)):
    #             v = [v]
    #         if v == sorted(state_lib):
    #             state = (len(state_lib) - 1)*3 + j
    #     return state, label_state

    def observe_state(self, h, w, edge_D=8):
        """input the h, w is int"""
        # state = 0
        label_state = self.mask[h, w]
        r = int(edge_D)  # ???
        sub_mask = self.mask[h-r: h+r, w-r: w+r]
        sub_np = np.unique(sub_mask)
        state = edge_det[sub_np.size-1][np.sum(sub_np)]
        return state, label_state


class ImagePoseDic:
    """multi image from camera
    multi cluster
    """
    def __init__(self, _intrinsic_scale):
        self.intrinsic_scale = _intrinsic_scale
        self.image_dic = {}  # {frame_id: ImagePose}
        self.point_list_lib = []  # [xyz, p0p1p2-hmm, obs_state] Class Point, all points
        self.one_CSS = None  # clustering class for one frame

        self.seq2num = {}  # to find the observing number

    def add_image_frame(self, image_pose):
        self.image_dic[image_pose.observing_num] = image_pose

    def project_3Dpoints_and_grid_build(self, frame, edge_D=8):
        """one frame point projection"""
        proj_state_count = 0
        new_index = []
        for index_p in self.image_dic[frame].point_index_list:  # every point in each frame
            if self.point_list_lib[index_p].if_obs:
                uv, xyzdepth_c = project_3d_point_in_img(self.point_list_lib[index_p].coordinate,
                                                         self.image_dic[frame].cam_k,
                                                         self.image_dic[frame].pose_r,
                                                         self.image_dic[frame].pose_t)
                if uv is False:
                    self.point_list_lib[index_p].obs_state = None
                    continue
                proj_state_count += 1

                uv_int = np.round(uv).astype(int)
                obs_state, label_state = self.image_dic[frame].observe_state(uv_int[1], uv_int[0],
                                                                             edge_D=edge_D)  # observing state
                # 先y后x，这是因为在NumPy中，数组的第一个索引对应于行，第二个索引对应于列
                self.point_list_lib[index_p].update_tims(obs_state, label_state)
                self.one_CSS.grid_map_update(index_p, obs_state, label_state, uv, uv_int, xyzdepth_c)
                new_index.append(index_p)
                """for hmm recording"""
                self.point_list_lib[index_p].obs_list.append(obs_state)
                self.point_list_lib[index_p].opti_obs_list.append(obs_state)  # it will be fixed after opti
        self.image_dic[frame].point_index_list = new_index
        print("the number of project points is {}".format(proj_state_count))
        return proj_state_count

    def point_state_optim(self, fix_dic):
        for k, v in fix_dic.items():
            if k == "none":
                state = None
            else:
                state = k
            for i in v:
                self.point_list_lib[i].obs_state = state  # mid results
                self.point_list_lib[i].label_state = state  # final results
                self.point_list_lib[i].opti_obs_list[-1] = state  # final element fix

    def one_frame_pcd_data_generate(self, f):
        """generate the point data with pcd type"""
        points_obs = []
        points_label = []  # without edge detect
        for index_p in self.image_dic[f].point_index_list:
            state = self.point_list_lib[index_p].obs_state
            state1 = self.point_list_lib[index_p].label_state
            if state is not None:
                one_point = self.point_list_lib[index_p].coordinate.tolist()
                one_point1 = one_point.copy()
                color = label_rgb[state]
                one_p = one_point + color
                points_obs.append(one_p)

                color1 = label_rgb[state1]
                one_p1 = one_point1 + color1
                points_label.append(one_p1)
        return points_obs, points_label

    def output_evaluation_data(self, frame):
        """output the data for evaluation
        [[xyz, label, truth], ...]"""
        evaluation_data = []
        for index_p in self.image_dic[frame].point_index_list:
            if (self.point_list_lib[index_p].truth_label is None) or \
                    (self.point_list_lib[index_p].label_state is None):  # no truth label
                continue
            p_new = []
            p_new += self.point_list_lib[index_p].coordinate.tolist()
            p_new.append(self.point_list_lib[index_p].label_state)
            p_new.append(self.point_list_lib[index_p].truth_label)
            evaluation_data.append(p_new)
        return evaluation_data

    def one_frame_process(self, save_path, scale_factor=8, gen_pcd=True):
        """start"""
        frame_list = sorted(self.image_dic.keys())
        for f in frame_list:
            print("\033[32m===== image No. is {} =====\033[0m".format(f))
            self.one_CSS = ClusterSegmentSystem(self.image_dic[f].mask,
                                                grid_size=scale_factor)  # init self.image_dic[f].mask
            point_num = self.project_3Dpoints_and_grid_build(f, edge_D=10)
            """generate pcd"""
            if gen_pcd:
                points_obs, points_label = self.one_frame_pcd_data_generate(f)
                pcd_generate(points_label, f, save_path, "direct_")
                pcd_generate(points_obs, f, save_path, "edge_")
            """re-segmentation"""
            start_time = time.time()
            self.one_CSS._init_data()
            state_change_dic = self.one_CSS.cluster_detect(save_path=save_path)  # 改为扩展和聚类
            self.point_state_optim(state_change_dic)  # fix point state
            cluster_time = time.time()
            print("frame {}， optimization of frame {},".format(f, (cluster_time - start_time)))
            """generate optimized pcd"""
            if gen_pcd:
                pcd_start_time = time.time()
                points_obs, points_label = self.one_frame_pcd_data_generate(f)  # xyz, rgb
                pcd_generate(points_obs, f, save_path, "cluster_")
                print("cluster_pcd generate {} cost {} s".format(f, (time.time() - pcd_start_time)))
                pcd_start_time1 = time.time()
                pcd_generate(points_label, f, save_path, "fixed_")
                print("fixed_pcd generate {} cost {} s".format(f, (time.time() - pcd_start_time1)))
                # image_visual = self.image_dic[frame].mask

    def multi_process(self, save_path, scale_factor=8, edge_D=8, gen_pcd=False, log_info=False):
        """main function to process every image"""
        """log update"""
        file = open(os.path.join(save_path, "evaluation", "re_seg_eval_origin.txt"), "w")
        file.close()
        file = open(os.path.join(save_path, "evaluation", "re_seg_eval_opti.txt"), "w")
        file.close()
        file = open(os.path.join(save_path, "evaluation", "re_seg_num_time.txt"), "w")
        file.close()
        """start"""
        frame_list = sorted(self.image_dic.keys())
        for f in frame_list:
            print("\033[32m===== image No. is {} =====\033[0m".format(f))
            self.one_CSS = ClusterSegmentSystem(self.image_dic[f].mask,
                                                grid_size=scale_factor)  # init self.image_dic[f].mask
            point_num = self.project_3Dpoints_and_grid_build(f, edge_D=edge_D)
            """evaluate the direct mapping"""

            dirct_map_result = self.output_evaluation_data(f)  # for evaluation [[xyz, obs, truth], ...]
            AP_origin, IoU_origin, p_num_origin, TP_origin, FP_origin = eval_one_times(dirct_map_result, f, 3)
            """generate pcd"""
            if gen_pcd:
                points_obs, points_label = self.one_frame_pcd_data_generate(f)
                pcd_generate(points_label, f, save_path, "direct_")
                pcd_generate(points_obs, f, save_path, "edge_")
            """re-segmentation"""
            start_time = time.time()
            self.one_CSS._init_data()
            state_change_dic = self.one_CSS.cluster_detect(save_path=save_path)  # 改为扩展和聚类
            self.point_state_optim(state_change_dic)  # fix point state
            cluster_time = time.time()
            print("optimization of frame {}, optimization cost {}s".format(f,
                                                                           (cluster_time - start_time)))
            """evaluate the optimized mapping"""
            optimized_map_result = self.output_evaluation_data(f)  # for evaluation [xyz, obs, truth]
            AP_opti, IoU_opti, p_num_opti, TP_opti, FP_opti = eval_one_times(optimized_map_result, f, 3)
            """save log"""
            add_save_list2txt([f, p_num_origin, AP_origin] + IoU_origin,
                              os.path.join(save_path, "evaluation", "re_seg_eval_origin.txt"))
            add_save_list2txt([f, p_num_opti, AP_opti] + IoU_opti,
                              os.path.join(save_path, "evaluation", "re_seg_eval_opti.txt"))
            add_save_list2txt([f, point_num, cluster_time - start_time],
                              os.path.join(save_path, "evaluation", "re_seg_num_time.txt"))
            """generate optimized pcd"""
            if gen_pcd:
                pcd_start_time = time.time()
                points_obs, points_label = self.one_frame_pcd_data_generate(f)  # xyz, rgb
                pcd_generate(points_obs, f, save_path, "cluster_")
                print("cluster_pcd generate {} cost {} s".format(f, (time.time() - pcd_start_time)))
                pcd_start_time1 = time.time()
                pcd_generate(points_label, f, save_path, "fixed_")
                print("fixed_pcd generate {} cost {} s".format(f, (time.time() - pcd_start_time1)))
                # image_visual = self.image_dic[frame].mask
            """evaluation show"""
            if log_info:
                print("\033[32m the direct mapping is {} {} num: {}\033[0m".format(AP_origin, IoU_origin, p_num_origin))
                print("[gravel, sand] TP: {}, FP: {}".format(TP_origin, FP_origin))
                print("\033[32m the optimized mapping is {} {} num: {}\033[0m".format(AP_opti, IoU_opti, p_num_opti))
                print("[gravel, sand] TP: {}, FP: {}".format(TP_opti, FP_opti))
        """for HMM filter, output the semantic recording
        [[xyz, truth, first state, re-seg], []]
        [[xyz, truth, first state, origin], []]"""
        opti_hmm_list = []
        origin_hmm_list = []
        for point in self.point_list_lib:
            if len(point.obs_list) < 2 or point.truth_label is None:
                continue  # 少于2次观测，忽略无真值点
            one_p = []
            one_p += point.coordinate.tolist()
            one_p.append(point.truth_label)
            one_p.append(point.first_state)  # first state

            one_p_opti = one_p.copy()
            one_p_opti += point.opti_obs_list
            opti_hmm_list.append(one_p_opti)

            one_p_origin = one_p.copy()
            one_p_origin += point.obs_list
            origin_hmm_list.append(one_p_origin)
        print("we get {} points for hmm calculate".format(len(origin_hmm_list)))
        return opti_hmm_list, origin_hmm_list


def pcd_generate(points_obs, frame, save_path, prefix=None):
    # points_obs [xyz, rgb]
    non_filter_name = "{}{}".format(prefix, frame)
    save_class = P2pcd(points_obs)
    save_class.generate(save_path, non_filter_name)
    print("generate the {}".format(non_filter_name))


def project_3d_point_in_img(xyz, K, pose_r, pose_t):
    # xyz_1 = np.append(xyz, 1)
    camera_point = np.dot(pose_r, xyz) + pose_t

    image_xy = np.dot(K, camera_point)
    u_v = image_xy[:2] / image_xy[2]  # [x, y] [heng, zong] [w, h] [col, row]
    cut_uv = u_v - np.array([160, 0])  # 进过裁剪，从1280,720 到 960,720，宽度坐标减小
    if 5 <= cut_uv[0] <= 955 and 5 <= cut_uv[1] <= 715:  # available observing region
        list_uvwd = camera_point.tolist()
        list_uvwd.append(np.linalg.norm(camera_point))
        return cut_uv, list_uvwd  # not Rounding for keeping accuracy
    else:
        return False, False


def one_frame_task(image_pose, points, frame, point_origin_state, save_path):
    """one frame task, load data to ImagePoseDic class
    用于读取单帧优化的过程，论文中有个图4张"""
    IPD = ImagePoseDic(1.0)
    IPD.add_image_frame(image_pose)  # image info
    IPD.point_list_lib = points  # point info
    IPD.image_dic[frame].point_index_list = list(range(len(points)))  # directly add point index
    # start
    IPD.one_frame_process(save_path, scale_factor=8) # scale_factor是网格尺寸
    """save the original pcd from r3live observation"""
    # pcd = LabelPCD(point_origin_state)  # origin pcd
    # pcd.generate(save_path, str(frame))
    """sava the estimation results"""


def multi_frame_task(data, save_path):
    """input ImagePoseDic class data:
    1. image dic: {frame: ImagePose}
    2. point_list_lib [Point ...]"""
    opti_hmm_list, origin_hmm_list = data.multi_process(save_path)  # [[xyz, truth, first state, re-seg], []]
    """save two obs results"""
    save_path1 = os.path.join(save_path, "evaluation")
    """[[xyz, truth, first state, re-seg], []]
    there is None truth in list"""
    file = open(os.path.join(save_path1, "opti_for_hmm.txt"), "w")
    file.close()
    file = open(os.path.join(save_path1, "origin_for_hmm.txt"), "w")
    file.close()
    save_list2txt(opti_hmm_list, os.path.join(save_path1, "opti_for_hmm.txt"))
    save_list2txt(origin_hmm_list, os.path.join(save_path1, "origin_for_hmm.txt"))
    print("\033[32m already save the obs results for hmm\033[0m")
    """hmm evaluation report"""
    file = open(os.path.join(save_path1, "opti_eval_hmm.txt"), "w")
    file.close()
    file = open(os.path.join(save_path1, "origin_eval_hmm.txt"), "w")
    file.close()
    opti_hmm = PointHMMEvaluation(opti_hmm_list)
    opti_hmm.filter_process()
    opti_hmm.first_label_rate(os.path.join(save_path1, "opti_eval_hmm.txt"))
    opti_hmm.less_time_result_eval(4, os.path.join(save_path1, "opti_eval_hmm.txt"))

    origin_hmm = PointHMMEvaluation(origin_hmm_list)
    origin_hmm.filter_process()
    origin_hmm.first_label_rate(os.path.join(save_path1, "origin_eval_hmm.txt"))
    origin_hmm.less_time_result_eval(4, os.path.join(save_path1, "origin_eval_hmm.txt"))


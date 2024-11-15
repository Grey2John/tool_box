import os
import time
# import rosbag
# from cv_bridge import CvBridge
import numpy as np
import torch

from filter import transmat, emissionprob_resg
from cluster_seg import ClusterSegmentSystem
from pcd_gen import PointList2RGBPCD as P2pcd
from commen import add_save_list2txt, save_list2txt
from evaluation import eval_one_times, PointHMMEvaluation, _evaluation

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
        self.first_state_rseg = None  # 只经过一次单帧优化
        self.reseg_state = None  # point observing have edge detect
        self.hmm_state = None
        self.label_state = None  # without edge detect, directly obs 最终结果
        self.truth_label = annotated_label  # read the null will become None
        self.if_obs = True
        self.obs_times = 0

        self.direct_obs_list = []   # edge状态的观测序列
        self.opti_obs_list = []

    def update_tims(self, obs_state, hmm_state, label_state, max_time=150):  # reduce observation
        if self.obs_times == 0:
            self.first_state = label_state
        self.obs_times += 1
        self.reseg_state = obs_state
        self.hmm_state = hmm_state
        self.label_state = label_state
        if self.obs_times >= max_time:
            self.if_obs = False

    def filter_prob_update(self):
        self.filter_prob = np.multiply( np.matmul(self.filter_prob, transmat), 
                                        emissionprob_resg[:, self.reseg_state].T )

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

    def observe_state(self, h, w, edge_D=8, hmm_D=4):
        """input the h, w is int"""
        # state = 0
        direct_state = self.mask[h, w]
        # r = int(edge_D)  # ???
        sub_mask = self.mask[h-edge_D: h+edge_D, w-edge_D: w+edge_D]
        sub_np = np.unique(sub_mask)
        # r1 = int(hmm_D)  # ???
        sub_mask1 = self.mask[h - hmm_D: h + hmm_D, w - hmm_D: w + hmm_D]
        sub_np1 = np.unique(sub_mask1)

        state = edge_det[sub_np.size-1][np.sum(sub_np)]
        hmm_state = edge_det[sub_np1.size - 1][np.sum(sub_np1)]
        return state, hmm_state, direct_state


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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def add_image_frame(self, image_pose):
        self.image_dic[image_pose.observing_num] = image_pose

    def project_3Dpoints_and_grid_build(self, frame, edge_D=8, hmm_D=4):
        """one frame point projection"""
        proj_state_count = 0
        new_index = []
        """project 3D points to 2D image in CPU"""
        for index_p in self.image_dic[frame].point_index_list:  # every point in each frame
            if self.point_list_lib[index_p].if_obs:  # 避免重复观测
                uv, xyzdepth_c = project_3d_point_in_img(self.point_list_lib[index_p].coordinate,
                                                         self.image_dic[frame].cam_k,
                                                         self.image_dic[frame].pose_r,
                                                         self.image_dic[frame].pose_t)
                if uv is False:
                    self.point_list_lib[index_p].obs_state = None
                    continue
                proj_state_count += 1

                uv_int = np.round(uv).astype(int)  # 此时，【 x-w, y-h 】
                obs_state, hmm_state, label_state = self.image_dic[frame].observe_state(uv_int[1], uv_int[0],
                                                                             edge_D=edge_D, hmm_D=hmm_D)  # observing state
                # 先y后x，这是因为在NumPy中，数组的第一个索引对应于行，第二个索引对应于列
                """prapare for one frame re-segmentation"""
                self.point_list_lib[index_p].update_tims(obs_state, hmm_state, label_state)
                self.one_CSS.grid_map_update(index_p, obs_state, label_state, uv, uv_int, xyzdepth_c)
                new_index.append(index_p)  # 中间没有被投影到图像中的点被删除，所以要更新
                """for hmm recording"""
                self.point_list_lib[index_p].direct_obs_list.append(hmm_state)  # 用来记录edge结果
                self.point_list_lib[index_p].opti_obs_list.append(hmm_state)  # it will be fixed after opti
        self.image_dic[frame].point_index_list = new_index
        print("the number of project points is {}".format(proj_state_count))
        return proj_state_count

    def project_3Dpoints_and_grid_build_gpu(self, frame, edge_D=8, hmm_D=4):
        """使用GPU并行处理点云投影和状态观测"""
        proj_state_count = 0
        new_index = []
        # 获取需要处理的点的索引
        valid_indices = torch.tensor(np.array([i for i in self.image_dic[frame].point_index_list
                                     if self.point_list_lib[i].if_obs]), device=self.device)
        if len(valid_indices) == 0:
            self.image_dic[frame].point_index_list = []
            return 0
        # 批量获取点云数据
        points = torch.tensor(np.array([self.point_list_lib[i].coordinate for i in valid_indices]),
                              device=self.device)
        # 转换相机参数到GPU
        pose_r = torch.tensor(self.image_dic[frame].pose_r, device=self.device)
        pose_t = torch.tensor(self.image_dic[frame].pose_t, device=self.device)
        cam_k = torch.tensor(self.image_dic[frame].cam_k, device=self.device)
        """并行计算投影"""
        camera_points = torch.matmul(pose_r, points.T).T + pose_t  # [N, 3]
        image_xy = torch.matmul(cam_k, camera_points.T).T  # [N, 3]
        uvs = image_xy[:, :2] / image_xy[:, 2:3]  # [N, 2]
        depths = torch.norm(camera_points, dim=1)  # [N] # 计算深度
        cut_uvs = uvs - torch.tensor([160.0, 0.0], device=self.device) # 裁剪后的坐标
        valid_mask = ((cut_uvs[:, 0] >= 5) & (cut_uvs[:, 0] <= 955) & 
                    (cut_uvs[:, 1] >= 5) & (cut_uvs[:, 1] <= 715)) # 检查有效的投影点
        # 获取有效的投影结果
        valid_cut_uvs = cut_uvs[valid_mask]
        valid_indices = valid_indices[valid_mask]
        valid_camera_points = camera_points[valid_mask]
        valid_depths = depths[valid_mask]
        uv_ints = torch.round(valid_cut_uvs).to(torch.int32) # 转换为整数坐标
        """semantic mapping into mask"""
        batch_obs_states = []
        batch_hmm_states = []
        batch_label_states = []
        # 将mask转换为tensor并移到GPU
        mask_tensor = torch.tensor(self.image_dic[frame].mask, device=self.device)
        for uv_int in uv_ints:
            h, w = uv_int[1].item(), uv_int[0].item()
            sub_mask = mask_tensor[h-edge_D:h+edge_D, w-edge_D:w+edge_D] # 获取子区域
            sub_mask1 = mask_tensor[h-hmm_D:h+hmm_D, w-hmm_D:w+hmm_D]
            sub_np = torch.unique(sub_mask) # 计算唯一值
            sub_np1 = torch.unique(sub_mask1)
            # 计算状态
            state = edge_det[sub_np.size(0)-1][torch.sum(sub_np).item()]
            hmm_state = edge_det[sub_np1.size(0)-1][torch.sum(sub_np1).item()]
            label_state = mask_tensor[h, w].item()
            
            batch_obs_states.append(state)
            batch_hmm_states.append(hmm_state)
            batch_label_states.append(label_state)
        """批量更新点的状态"""
        for idx, (index_p, obs_state, hmm_state, label_state, uv, uv_int, camera_point, depth) in enumerate(
                zip(valid_indices.cpu().numpy(), 
                    batch_obs_states,
                    batch_hmm_states, 
                    batch_label_states,
                    valid_cut_uvs.cpu().numpy(),
                    uv_ints.cpu().numpy(),
                    valid_camera_points.cpu().numpy(),
                    valid_depths.cpu().numpy())):
            
            # 更新点的状态
            self.point_list_lib[index_p].update_tims(obs_state, hmm_state, label_state)
            
            # 更新网格地图
            xyzdepth_c = camera_point.tolist() + [depth]
            self.one_CSS.grid_map_update(index_p, obs_state, label_state, uv, uv_int, xyzdepth_c)
            
            # 记录HMM相关信息
            self.point_list_lib[index_p].direct_obs_list.append(hmm_state)
            self.point_list_lib[index_p].opti_obs_list.append(hmm_state)
            
            new_index.append(index_p)
            proj_state_count += 1
        
        # 更新点索引列表
        self.image_dic[frame].point_index_list = new_index
        print("the number of project points is {}".format(proj_state_count))
        # 在处理完一帧数据后清理GPU缓存
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"GPU processing error: {e}")
        return proj_state_count

    def point_state_optim(self, fix_dic):
        for k, v in fix_dic.items():
            if k == "none":
                state = None
            else:
                state = k
            for i in v:
                self.point_list_lib[i].hmm_state = state  # 单帧优化的结果修改hmm的观测状态
                self.point_list_lib[i].label_state = state  # final results
                self.point_list_lib[i].opti_obs_list[-1] = state  # final element fix

    def point_state_optim_gpu(self, fix_dic):
        """并行更新点状态"""
        if torch.cuda.is_available():
            # 批量更新点状态
            point_indices = []
            new_states = []
            for k, v in fix_dic.items():
                state = None if k == "none" else k
                point_indices.extend(v)
                new_states.extend([state] * len(v))
            
            if point_indices:
                point_indices = torch.tensor(point_indices, device=self.device)
                new_states = torch.tensor(new_states, device=self.device)
                
                # 批量更新
                for i, state in zip(point_indices, new_states):
                    self.point_list_lib[i].hmm_state = state
                    self.point_list_lib[i].label_state = state
                    self.point_list_lib[i].opti_obs_list[-1] = state

    def one_frame_pcd_data_generate(self, f):
        """generate the point data with pcd type"""
        points_obs = [] # 包含边界语义
        points_label = []  # without edge detect
        for index_p in self.image_dic[f].point_index_list:
            state_edge = self.point_list_lib[index_p].hmm_state
            state = self.point_list_lib[index_p].label_state
            if state_edge is not None:
                one_point = self.point_list_lib[index_p].coordinate.tolist()
                one_point1 = one_point.copy()
                color = label_rgb[state_edge]
                one_p = one_point + color
                points_obs.append(one_p)

                color1 = label_rgb[state]
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

    def _evaluate_one_frame(self, frame, num_class=3):
        """evaluate"""
        point_count = 0
        count = np.zeros(num_class ** 2)

        for index_p in self.image_dic[frame].point_index_list:
            if (self.point_list_lib[index_p].truth_label is None) or \
                    (self.point_list_lib[index_p].label_state is None):  # no truth label
                continue
            label_state = self.point_list_lib[index_p].label_state
            truth_label = self.point_list_lib[index_p].truth_label
            point_count += 1
            count[num_class * truth_label + label_state] += 1

        confusion_matrix = count.reshape(num_class, num_class)
        acc, iou_list = _evaluation(confusion_matrix)
        TP = np.array([confusion_matrix[1, 1], confusion_matrix[2, 2]])  # 2 classes
        FP = np.array([confusion_matrix[1, 0] + confusion_matrix[1, 2],
                    confusion_matrix[2, 0] + confusion_matrix[2, 1]])  # 2 classes
        return point_count, acc, iou_list, TP, FP
    
    def _evaluate_opti_frame(self, frame, num_class=3):
        """evaluate the optimized mapping: rIoU, 
        TM: 错误分割点被正确修正, FM: 错误分割点未被修正, FC: 正确分割点被错误修正"""
        point_count = 0
        count = np.zeros(num_class ** 2)
        TM = 0  # True Modification
        FM = 0  # Failed Modification  
        FC = 0  # False Correction
        for index_p in self.image_dic[frame].point_index_list:
            if (self.point_list_lib[index_p].truth_label is None) or \
                    (self.point_list_lib[index_p].label_state is None):  # no truth label
                continue
            reseg_state = self.point_list_lib[index_p].label_state # re-segmentation
            truth_label = self.point_list_lib[index_p].truth_label
            origin_label = self.point_list_lib[index_p].direct_obs_list[-1] # direct obs from projection
            
            point_count += 1
            count[num_class * truth_label + reseg_state] += 1 # 重分割后的整体精度
            if truth_label != origin_label:
                if truth_label == reseg_state:
                    TM += 1  # 错误分割点被正确修正
                else:
                    FM += 1  # 错误分割点未被修正
            elif truth_label == origin_label and truth_label != reseg_state:
                FC += 1 

        confusion_matrix = count.reshape(num_class, num_class)
        acc, iou_list = _evaluation(confusion_matrix)
        TP = np.array([confusion_matrix[1, 1], confusion_matrix[2, 2]])  # 2 classes
        FP = np.array([confusion_matrix[1, 0] + confusion_matrix[1, 2],
                    confusion_matrix[2, 0] + confusion_matrix[2, 1]])  # 2 classes
        rIoU = TM / (TM + FM + FC)
        return point_count, acc, iou_list, TP, FP, TM, FM, FC, rIoU
    
    def one_frame_process(self, save_path, scale_factor=8, gen_pcd=True):
        """start"""
        frame_list = sorted(self.image_dic.keys())
        for f in frame_list:
            print("\033[32m===== image No. is {} =====\033[0m".format(f))
            self.one_CSS = ClusterSegmentSystem(self.image_dic[f].mask,
                                                grid_size=scale_factor)  # init self.image_dic[f].mask
            point_num = self.project_3Dpoints_and_grid_build(f, edge_D=4)
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

    def multi_process(self, save_path, scale_factor=8, edge_D=4, hmm_D=4, gen_pcd=False, log_info=False):
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
            point_num = self.project_3Dpoints_and_grid_build(f, edge_D=edge_D, hmm_D=hmm_D)
            """evaluate the direct mapping"""
            dirct_map_result = self.output_evaluation_data(f)  # for evaluation [[xyz, obs, truth], ...]
            AP_origin, IoU_origin, p_num_origin, TP_origin, FP_origin = eval_one_times(dirct_map_result, f, 3)
            """generate pcd: direct, edge"""
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
            if len(point.direct_obs_list) < 2 or point.truth_label is None:
                continue  # 少于2次观测，忽略无真值点
            one_p = []
            one_p += point.coordinate.tolist()
            one_p.append(point.truth_label)
            one_p.append(point.first_state)  # first state

            one_p_opti = one_p.copy()
            one_p_opti += point.opti_obs_list
            opti_hmm_list.append(one_p_opti)

            one_p_origin = one_p.copy()
            one_p_origin += point.direct_obs_list
            origin_hmm_list.append(one_p_origin)
        print("we get {} points for hmm calculate".format(len(origin_hmm_list)))
        return opti_hmm_list, origin_hmm_list

    def process_eval2each_frame(self, save_path, scale_factor=8, edge_D=8):
        """multi-frame process & evaluation for each frame"""
        """log update, save: frame, TP, FP, time, point num, mapping deviation"""
        origin_seg_eval_txt = os.path.join(save_path, "one_frame_evaluation", "origin_seg_eval.txt")
        os.makedirs(os.path.dirname(origin_seg_eval_txt), exist_ok=True)
        file = open(origin_seg_eval_txt, "w")
        file.close()
        re_seg_eval_txt = os.path.join(save_path, "one_frame_evaluation", "re_seg_eval.txt")
        file = open(re_seg_eval_txt, "w")
        file.close()
        re_seg_filter_eval_txt = os.path.join(save_path, "one_frame_evaluation", "re_seg_filter_eval.txt")
        file = open(re_seg_filter_eval_txt, "w")
        file.close()
        """start"""
        frame_list = sorted(self.image_dic.keys())
        for f in frame_list:
            print("\033[32m===== image No. is {} =====\033[0m".format(f))
            """cal: projection"""
            project_time = time.time()
            self.one_CSS = ClusterSegmentSystem(self.image_dic[f].mask, grid_size=scale_factor)  # init self.image_dic[f].mask
            point_num = self.project_3Dpoints_and_grid_build_gpu(f, edge_D=edge_D)
            project_time_cost = time.time() - project_time
            print("projection of frame {}, cost {}s".format(f, project_time_cost))
            """evaluate the direct mapping in one frame point group"""
            p_num_origin, AP_origin, IoU_origin, _, _ = self._evaluate_one_frame(f, 3)
            add_save_list2txt([f, point_num, p_num_origin, project_time_cost, 
                               AP_origin]+IoU_origin, origin_seg_eval_txt)
            """cal: re-segmentation"""
            opti_time = time.time()
            self.one_CSS._init_data()
            state_change_dic = self.one_CSS.cluster_detect(save_path=save_path)  # 改为扩展和聚类 
            self.point_state_optim_gpu(state_change_dic)  # fix point state
            opti_time_cost = time.time() - opti_time
            print("frame {}, optimization cost {}s".format(f, opti_time_cost))
            """evaluate the optimized mapping"""
            p_num_opti, AP_opti, IoU_opti, _, _, TM, FM, FC, rIoU= self._evaluate_opti_frame(f, 3)
            add_save_list2txt([f, point_num, p_num_opti, opti_time_cost, 
                               AP_opti] + IoU_opti + [TM, FM, FC, rIoU], re_seg_eval_txt)
            """cal: one frame points through HMM filter, only time cost"""
            filter_time = time.time()
            for index_p in self.image_dic[f].point_index_list:
                self.point_list_lib[index_p].filter_prob_update()
            filter_time_cost = time.time() - filter_time
            print("filter optimization cost {}s".format(filter_time_cost))
            add_save_list2txt([f, point_num, filter_time_cost], re_seg_filter_eval_txt)


def pcd_generate(points_obs, frame, save_path, prefix=None):
    # points_obs [xyz, rgb]
    non_filter_name = "{}{}".format(prefix, frame)
    save_class = P2pcd(points_obs)
    save_class.generate(save_path, non_filter_name)
    print("generate the {}".format(non_filter_name))


def project_3d_point_in_img(xyz, K, pose_r, pose_t):
    # xyz_1 = np.append(xyz, 1)
    camera_point = np.dot(pose_r, xyz) + pose_t  # [xyz]

    image_xy = np.dot(K, camera_point)
    u_v = image_xy[:2] / image_xy[2]  # [x, y] [heng, zong] [w, h] [col, row]
    cut_uv = u_v - np.array([160, 0])  # 进过裁剪，从1280,720 到 960,720，宽度坐标减小
    if 5 <= cut_uv[0] <= 955 and 5 <= cut_uv[1] <= 715:  # available observing region
        list_uvwd = camera_point.tolist()
        list_uvwd.append(np.linalg.norm(camera_point))  # [xyz, depth]
        return cut_uv, list_uvwd  # not Rounding,
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


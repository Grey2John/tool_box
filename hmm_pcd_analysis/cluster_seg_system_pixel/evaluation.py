import numpy as np
import os
from multiprocessing import Process
from filter import PointHMM
from data_loader import PointDataLoader


def eval_one_times(data, frame, num_class=3, save_path=None):
    """
    input: data [xyz, label, truth]
    evaluate the one frame segment result
    output: origin seg and optimized seg
    AP IoU
    """
    point_count = 0
    count = np.zeros(num_class ** 2)
    # print("process points at {} frame ".format(frame))
    for p in data:  # p [xyz, label, truth]
        # point = filter.PointHMM(num_class) # 不用每次都filter了
        # filtered_label = point.filter(p[0:3] + p[4:time + 5])[-1]  # 预测值
        truth_label = p[4]
        obs_label = p[3]
        point_count += 1  # 处理点的计数
        count[num_class * truth_label + obs_label] += 1

    confusion_matrix = count.reshape(num_class, num_class)
    # print(confusion_matrix)  # 打印混合矩阵
    acc, iou_list = _evaluation(confusion_matrix)
    TP = np.array([confusion_matrix[1, 1], confusion_matrix[2, 2]])  # 2 classes
    FP = np.array([confusion_matrix[1, 0] + confusion_matrix[1, 2],
                   confusion_matrix[2, 0] + confusion_matrix[2, 1]])  # 2 classes
    return acc, iou_list, point_count, TP, FP


def one_frame_evaluation(fixed_pcd_path, direct_pcd_path, truth_pcd_path):
    """
    step 1: load 3 pcd: truth, direct, fixed
    step 2: point matching 点云配对
    step 3: calculate the TP分割错误的点被正确修改, TN分割没错的点没被正确修改
    step 4: calculate metrics: IoU, precision ...
    step 5: generate the report
    """
    from data_load import readpcd_rgb2label as RRGBL
    point_truth = read_truth_labeled_rgb_pcd(truth_pcd_path)
    point_direct = RRGBL(direct_pcd_path)
    point_fixed = RRGBL(fixed_pcd_path)
    print('truth length {}, direct length {}, fixed length {}'.format(len(point_truth),
                                                                      len(point_direct),
                                                                      len(point_fixed)))
    # step 2 match
    mix_matrix = []
    j=0
    for i in range(len(point_fixed)):
        # 找到b中距离a[i]最近的点
        jump = False
        for k in range(5):
            min_dist_b = sum([x - y for x, y in zip(point_truth[i][0:3], point_direct[i][0:3])])
            if min_dist_b > 0.0001:
                continue
            else:
                j += k
                break
            jump = True

        if jump:
            continue
        else:
            # [xyz, fixed, direct, truth]
            mix_matrix.append(point_fixed[i][0:4] + [ point_direct[i+j][3], point_truth[i+j][3] ])
    mix_matrix = np.array(mix_matrix)
    # step 3 [xyz, fixed, direct, truth]
    seg_wrong = []
    TP = []; FP = []; FN = [] # point index list
    for l in range(mix_matrix.shape[0]):
        if np.all(mix_matrix[l, :][3:] == mix_matrix[l, :][3]):
            continue
        else:
            if mix_matrix[l, :][4] != mix_matrix[l, :][5]:
                seg_wrong.append(l) # 真值和直接不一样
                if mix_matrix[l, :][3] == mix_matrix[l, :][5]:
                    TP.append(l)  # 真值和修复后一样
                else:
                    FN.append(l)  # 真值和修复后不一样
            elif mix_matrix[l, :][3] != mix_matrix[l, :][5]:
                FP.append(l)
    # step 4
    print("TP is {}, FP is {}, FN is {}".format(len(TP), len(FP), len(FN)))
    iou = len(TP)/(len(TP)+len(FP)+len(FN))
    precision = len(TP)/(len(TP)+len(FP))
    recall = len(TP)/(len(TP)+len(FN))
    return mix_matrix, iou, precision, recall, len(seg_wrong)


def read_truth_labeled_rgb_pcd(pcd_path):
    """
    read labled pcd
    [xyz, label]
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
    return data


# def multi_frame_evaluation(truth_pcd_path, fixed_pcd_path):
#     """
#     step 1: load 3 pcd: truth, direct, fixed
#     step 2: point matching 点云配对
#     step 3: calculate the TP分割错误的点被正确修改, TN分割没错的点没被正确修改
#     step 4: calculate metrics: IoU, precision ...
#     step 5: generate the report
#     """
#     point_truth = read_truth_labeled_rgb_pcd(truth_pcd_path)
#     point_fix = read_truth_labeled_rgb_pcd(fixed_pcd_path)
#
#     return acc, iou_list, point_count, TP, FP


def _evaluation1(confusion_matrix):
    acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()  # accuracy
    MIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    MIoU = np.where(np.isnan(MIoU), 0, MIoU)  # mean IoU
    return acc, MIoU.tolist()


def _evaluation(confusion_matrix):
    # Get the number of classes
    num_class = confusion_matrix.shape[0]
    # Initialize the accuracy and IoU list
    acc = 0
    iou_list = []
    # Loop over the classes
    for i in range(num_class):
        # Get the true positives, false positives, and false negatives
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        # Calculate the accuracy and IoU for this class
        acc += tp
        if (tp + fp + fn) != 0:
            iou = tp / (tp + fp + fn)
        else:
            iou = 0
        iou_list.append(iou)
    # Calculate the overall accuracy
    acc = acc / confusion_matrix.sum()
    # Return the accuracy and IoU list
    return acc, iou_list


class PointHMMEvaluation:
    def __init__(self, point_list, if_load=False):
        """match_labeled_points: [xyz, truth, first label, obs] is what we need"""
        if if_load:
            data = PointDataLoader(point_list)  # point_list is path
            self.match_labeled_points = data.read_txt2list_points()
        else:
            self.match_labeled_points = point_list
        self.num_class = 3

        self.time_list = []  # 观测次数的列表
        self.point_list = []
        self.point_dir = {}  # 字典 {观测次数: [[xyz, filtered_list], [], []]}

    def filter_process(self):
        """, """
        eval_result = {}
        for i in self.match_labeled_points:  # i: [xyz, truth(None), first label, obs(empty)]
            point = PointHMM(self.num_class)
            point_filtered = point.filter_all_list(i[0:3] + i[4:])  # 输入 [xyz, first label, obs_list]
            if point_filtered is False:
                continue
            if (len(i) - 5) not in self.time_list:
                self.time_list.append(len(i) - 5)
                self.point_dir[len(i) - 5] = [i[0:4] + point_filtered]  # [xyz, truth, first label, filter_label_list]
            else:
                self.point_dir[len(i) - 5].append(i[0:4] + point_filtered)
            """filtered results"""
            self.point_list.append(i[0:4] + point_filtered)  # [xyz, truth(None), first label, filter_label_list]
        self.time_list.sort()  # 存的所有观测次数
        print("done all points filter")
        print("point list size is {}".format(len(self.point_list)))

    def less_time_result_eval(self, core_num, save_path, if_save=None):
        """
        self.point_list [[xyz, truth, first label, filter_label_list], ...]
        少于某观测次数的所有点- all point
        优化滤波的过程，一次性将滤波结果都算完记录下来
        """
        eval_result = {}
        index_list = []
        for l in range(core_num):
            one_index = [i for i in range(l, len(self.time_list), core_num)]
            index_list.append(one_index)
        print(index_list)

        processes = []
        for loop_num in range(len(index_list[0])):
            for core_n in range(core_num):
                if len(index_list[core_n]) <= loop_num:
                    continue
                p = Process(target=_hmm_eval_less_times,
                            args=(self.point_list, self.time_list[index_list[core_n][loop_num]],
                                  self.num_class, save_path,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            print("down {} loop".format(loop_num))
        return eval_result

    def first_label_rate(self, save_path):
        """ the accuracy of the first observing -- direct label
         just write one line
         """
        point_count = 0
        count = np.zeros(self.num_class ** 2)  # class count
        for p in self.point_list:
            non_filter_label = p[4]  # first label without filter
            truth_label = p[3]  # [xyz, truth, filter_label]
            point_count += 1
            count[self.num_class * truth_label + non_filter_label] += 1

        confusion_matrix = count.reshape(self.num_class, self.num_class)  # to 2D
        acc, iou_list = _evaluation(confusion_matrix)
        eval_out = [acc] + iou_list

        one_line = str(1) + ", " + str(point_count)
        for s in ([acc] + iou_list):
            one_line = one_line + ", " + str(s)
        one_line = one_line + "\n"

        with open(save_path, 'a') as ff:
            ff.write(one_line)
            print("save the evaluation result of {} times".format(1))
        return eval_out


def _hmm_eval_less_times(point_list, time, num_class, save_path):
    """
    少于某次数的所有点都取
     AP IoU
     point_list [xyz, truth, first label, filter_label_list]
     time is
     """
    point_count = 0
    count = np.zeros(num_class ** 2)
    # print("process {} times points".format(time))
    for p in point_list:
        if p[3] is not None:
            truth_label = p[3]  # [xyz, truth, filter_label]
        else:
            continue
        if len(p) > time+5:
            filtered_label = p[time+4]
        else:
            filtered_label = p[-1]

        point_count += 1
        count[num_class * truth_label + filtered_label] += 1
    confusion_matrix = count.reshape(num_class, num_class)
    acc, iou_list = _evaluation(confusion_matrix)
    eval_out = [acc] + iou_list

    one_line = str(time) + ", " + str(point_count)
    for s in eval_out:
        one_line = one_line + ", " + str(s)
    one_line = one_line + "\n"
    # if os.path.exists(save_path):
    #     with open(save_path, 'w') as file:
    #         pass

    with open(save_path, 'a') as f:
        f.write(one_line)
        # print("save the evaluation result of {} times".format(time))
    return eval_out


if __name__ == "__main__":
    """读txt进行直接评价，中间执行，读两个优化和无优化的txt，分别输出2个hmm结果"""
    save_path = '/media/zlh/WD_BLACK/earth_rosbag/paper_data/t3bag1/evaluation'
    source_txt = os.path.join(save_path, "origin_for_hmm.txt")
    read_tool = PointHMMEvaluation(source_txt, if_load=True)
    read_tool.filter_process()
    read_tool.first_label_rate(os.path.join(save_path, "origin_eval_hmm.txt"))
    read_tool.less_time_result_eval(4, os.path.join(save_path, "origin_eval_hmm.txt"))

    source_txt = os.path.join(save_path, "opti_for_hmm.txt")
    read_tool = PointHMMEvaluation(source_txt, if_load=True)
    read_tool.filter_process()
    read_tool.first_label_rate(os.path.join(save_path, "opti_eval_hmm.txt"))
    read_tool.less_time_result_eval(4, os.path.join(save_path, "opti_eval_hmm.txt"))

    """读txt进行直接评价，一帧的结果"""
    # mix_matrix, iou, precision, recall, seg_wrong = one_frame_evaluation("F:\dataset\outline_seg_slam\one_frame_evaluation\\fixed_175.pcd",
    #                          "F:\dataset\outline_seg_slam\one_frame_evaluation\direct_175.pcd",
    #                          "F:\dataset\outline_seg_slam\one_frame_evaluation\\truth_fixed_175.pcd")
    # mix_matrix, iou, precision, recall = one_frame_evaluation("/media/zlh/zhang/dataset/outline_seg_slam/one_frame_evaluation/fixed_90.pcd",
    #                          "/media/zlh/zhang/dataset/outline_seg_slam/one_frame_evaluation/direct_90.pcd",
    #                          "/media/zlh/zhang/dataset/outline_seg_slam/one_frame_evaluation/truth_fixed_90.pcd")
    # print(iou)
    # print(precision)
    # print(recall)
    # print(seg_wrong)

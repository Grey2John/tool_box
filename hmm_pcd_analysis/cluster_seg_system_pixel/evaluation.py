import numpy as np
import os
from multiprocessing import Process
from filter import PointHMM
from data_loader import PointDataLoader
from tqdm import tqdm

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


# def _evaluation(confusion_matrix):
#     acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()  # accuracy
#     MIoU = np.diag(confusion_matrix) / (
#             np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
#             np.diag(confusion_matrix))
#     MIoU = np.where(np.isnan(MIoU), 0, MIoU)  # mean IoU
#     return acc, MIoU.tolist()

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
    def __init__(self, point_list, if_load=False, max_obs_time=150):
        """match_labeled_points: [xyz, truth, first label, obs] is what we need"""
        if if_load:
            data = PointDataLoader(point_list)  # point_list is path
            self.match_labeled_points = data.read_txt2list_points()
        else:
            self.match_labeled_points = point_list
        self.num_class = 3

        self.time_list = []
        self.point_list = []
        self.point_dir = {}  # 字典 {观测次数: [[xyz, filtered_list], [], []]}

    def filter_process(self):
        """, """
        eval_result = {}
        for i in self.match_labeled_points:  # i: [xyz, truth, first label, obs]
            point = PointHMM(self.num_class)
            point_filtered = point.filter_all_list(i[0:3] + i[4:])  # 输入 [xyz, first label, obs_list]
            if (len(i) - 5) not in self.time_list:
                self.time_list.append(len(i) - 5)
                self.point_dir[len(i) - 5] = [i[0:4] + point_filtered]  # [xyz, truth, first label, filter_label_list]
            else:
                self.point_dir[len(i) - 5].append(i[0:4] + point_filtered)
            self.point_list.append(i[0:4] + point_filtered)  # [xyz, truth, first label, filter_label_list]
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


def _hmm_eval_less_times(point_list, time, num_class, save_path):
    """
    少于某次数的所有点都取
     AP IoU
     point_list [xyz, truth, first label, filter_label_list]
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
    read_tool = PointHMMEvaluation("/media/zlh/zhang/earth_rosbag/paper_data/t3bag1/evaluation/opti_for_hmm.txt"
                                   , if_load=True)
    read_tool.filter_process()
    read_tool.less_time_result_eval(4, os.path.join('/media/zlh/zhang/earth_rosbag/paper_data/t3bag1/evaluation'
                                                    , "opti_eval_hmm.txt"))
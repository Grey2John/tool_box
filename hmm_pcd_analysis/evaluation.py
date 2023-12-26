# 用来评价最后的分割效果
import os
import filter
from multiprocessing import Process
import numpy as np
import data_loader as DL
from supervise_learning import generate_truth_labeled_rgb_pcd, point_sample_labeling


class PointSegEvaluation:
    def __init__(self, txt_path, ground_pcd, less_time=12, max_obs_time=150):
        """
        read point observing results
        对应一个bag的数据分割结果估计
        """
        self.test_data = DL.PointDataLoader(txt_path)  # read data
        SP = self.test_data.read_txt_list_points(have_obs_time=True, min_time=less_time, upper_times=max_obs_time+1)  # xyz, first label, obs
        labeled_points = generate_truth_labeled_rgb_pcd(ground_pcd, "", '', if_gen_pcd=0)  # standard [xyz, label]
        """match_labeled_points: [xyz, truth, first label, obs] is what we need"""
        self.match_labeled_points = point_sample_labeling(labeled_points, SP, 4)
        self.num_class = 3

        self.time_list = []
        self.point_list = []
        self.point_dir = {}  # 字典 {观测次数: [[xyz, filtered_list], [], []]}

    def filter_process(self):
        """"""
        eval_result = {}
        for i in self.match_labeled_points:  # i: [xyz, truth, first label, obs]
            point = filter.PointHMM(3)
            point_filtered = point.filter_all_list(i[0:3] + i[4:])  # 输入 [xyz, first label, obs_list]
            if (len(i) - 5) not in self.time_list:
                self.time_list.append(len(i) - 5)
                self.point_dir[len(i) - 5] = [i[0:4] + point_filtered]  # [xyz, truth, first label, filter_label_list]
            else:
                self.point_dir[len(i) - 5].append(i[0:4] + point_filtered)
            self.point_list.append(i[0:4] + point_filtered)
        self.time_list.sort()  # 存的所有观测次数
        print("done all points filter")
        print("point list size is {}".format(len(self.point_list)))

    def _one_time_result_filer(self, core_num, save_path, if_save=None):
        """
        keep all the points with truth label
        输出每种观察次数的准确率, 所有观测n次的点，前n次
        多进程同时工作
        """
        eval_result = {}

        # 任务分配
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
                p = Process(target=hmm_eval_one_times, args=(self.point_dir, self.time_list[index_list[core_n][loop_num]],
                                                             self.num_class, save_path,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            print( "down {} loop".format(loop_num) )
        return eval_result

    def _less_time_result_filer(self, core_num, save_path, if_save=None):
        """
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
                p = Process(target=hmm_eval_less_times, args=(self.point_list, self.time_list[index_list[core_n][loop_num]],
                                                              self.num_class, save_path,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            print( "down {} loop".format(loop_num) )
        return eval_result

    def first_label_rate(self, save_path1, save_path2):
        """ the accuracy of the first observing
         just one line"""
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

        # with open(save_path1, 'w') as f:
        #     f.write(one_line)
        #     print("save the evaluation result of {} times".format(1))

        with open(save_path2, 'w') as ff:
            ff.write(one_line)
            print("save the evaluation result of {} times".format(1))
        return eval_out


def hmm_eval_less_times(point_list, time, num_class, save_path):
    """
    少于某次数的所有点都取
     AP IoU
     """
    point_count = 0
    count = np.zeros(num_class ** 2)
    print("process {} times points".format(time))
    for p in point_list:
        if len(p) > time+5:
            filtered_label = p[time+4]
        else:
            filtered_label = p[-1]
        truth_label = p[3]  # [xyz, truth, filter_label]
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
        print("save the evaluation result of {} times".format(time))
    return eval_out


def hmm_eval_one_times(data_dir, time, num_class, save_path):
    """
    add all point with same times
    AP IoU
    """
    point_count = 0
    count = np.zeros(num_class ** 2)
    print("process {} times points".format(time))
    for k, p_list in data_dir.items():
        if k >= time:
            for p in p_list:  # [xyz, truth, first label, filter_label_list]
                # point = filter.PointHMM(num_class) # 不用每次都filter了
                # filtered_label = point.filter(p[0:3] + p[4:time + 5])[-1]  # 预测值
                filtered_label = p[time+4]
                truth_label = p[3]
                point_count += 1  # 处理点的计数
                count[num_class * truth_label + filtered_label] += 1

    confusion_matrix = count.reshape(num_class, num_class)
    acc, iou_list = _evaluation(confusion_matrix)
    eval_out = [acc] + iou_list

    one_line = str(time) + ", " + str(point_count)
    for s in ([acc] + iou_list):
        one_line = one_line + ", " + str(s)
    one_line = one_line + "\n"

    with open(save_path, 'a') as f:
        f.write(one_line)
        print("save the evaluation result of {} times".format(time))
    return eval_out, one_line


def _evaluation(confusion_matrix):
    acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    MIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    MIoU = np.where(np.isnan(MIoU), 0, MIoU)
    return acc, MIoU.tolist()


if __name__ == "__main__":
    file_num = 10
    bag_txt_path = "F:\earth_rosbag\data\\test5_change\\bag_list\\bag10-g\pt_obs.txt".format(file_num)
    ground_pcd = "F:\earth_rosbag\data\\test5_change\labeled\labeled\origin_bag{}.pcd".format(file_num)
    pe = PointSegEvaluation(bag_txt_path, ground_pcd,  less_time=12, max_obs_time=60)
    pe.filter_process()
    pe.first_label_rate(None, "F:\earth_rosbag\data\\test5_change\evaluation\\bag{}_2.txt".format(file_num))
    # pe._one_time_result_filer(4, "F:\earth_rosbag\data\\test3\evaluation\\bag{}_1.txt".format(file_num), if_save=1) # 1是one
    pe._less_time_result_filer(4, "F:\earth_rosbag\data\\test5_change\evaluation\\bag{}_2.txt".format(file_num), if_save=1)
    # pe.plot_part(eval_result)

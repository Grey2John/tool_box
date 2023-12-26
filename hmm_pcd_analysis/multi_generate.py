# 多进程同时运行
# 读取txt，解析并生成绘图结果——一个大函数
# bag.txt -
import os
import sys
import numpy as np
from multiprocessing import Process
from evaluation import _evaluation
import data_loader as DL
from filter import txt_HMM_pcd, LabelPCD
from common import file_read


def one_txt_pipeline(source_txt, save_path, min_obs_time=12, max_obs_time=150):
    """
    save file:
    1. evaluation txt,
    2. unfiltered pcd,
    3. filtered pcd
    """
    tet_data = DL.PointDataLoader(source_txt)
    SP = tet_data.read_txt_list_points(upper_times=max_obs_time + 1)  # xyz, first label, obs
    file_name = source_txt.split("/")[-1].split(".")[0]
    print("finish the reading of {}.txt".format(file_name))
    save_pcd(file_name, SP, save_path, min_obs_time)
    # filtered list
    # labeled_points = generate_truth_labeled_rgb_pcd(ground_pcd, "", '', if_gen_pcd=0)


def save_pcd(name, points_in, save_path, min_obs_time):
    filter_name = "filter_{}".format(name)
    non_filter_name = "non_filter_{}".format(name)
    points_first_obs = [l[:4] for l in points_in]
    one_obs_pcd = LabelPCD(points_first_obs)
    one_obs_pcd.generate(save_path, non_filter_name)  # 观察一次的
    print("generate the {}".format(non_filter_name))

    txt_HMM_pcd(points_in, save_path, filter_name)
    print("generate the {}".format(filter_name))


def hmm_eval_less_times(point_list, time, num_class, save_path):
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


if __name__ == "__main__":
    """
    main job:
    1. generate txt, AP, IoU curves
    2. pcd after filter
    3. pcd non_filter, first obs
    """
    save_path = '/media/zlh/zhang/earth_rosbag/data/test4/bag_list/bag0'
    txt_dir_path = "/media/zlh/zhang/earth_rosbag/data/test4/bag_list/bag0"
    txt_file_list = file_read.dir_multi_file(txt_dir_path, "txt")
    # 任务分配
    for task in txt_file_list:
        one_txt_pipeline(task, save_path)


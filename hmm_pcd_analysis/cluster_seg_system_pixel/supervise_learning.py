import os
import numpy as np
from data_loader import PointDataLoader
import math
import random


def supervise_learn(point_list_with_real_L):
    """ [xyz, real_L, obs_list] """
    B = np.zeros([3, 7])
    for p in point_list_with_real_L:
        real_label = int(p[3])
        for i in set(p[4:]):
            B[real_label, int(i)] += p[4:].count(i)
    # print(B)
    result_B = B / np.sum(B, axis=1)[:, np.newaxis]
    np.set_printoptions(suppress=True)
    print(np.around(result_B, decimals=5))


def down_sample_from_list(point_in, alpha=0.1):
    """input: [xyz, truth, first, obs_list], alpha 降采样系数"""
    dir_point = {}
    for p in point_in:
        one_p = p[:4] + p[5:]  # 去掉first label
        obs_time = len(p[5:])
        if obs_time not in dir_point.keys():
            dir_point[obs_time] = [one_p]
        else:
            dir_point[obs_time].append(one_p)

    point_out = []
    coeff = np.linspace(alpha, 1, len(dir_point.keys())).tolist()
    print("length is {}".format(len(coeff)))
    keys = sorted(dir_point.keys())
    for i, k in enumerate(keys):
        get_num = math.ceil(coeff[i] * len(dir_point[k]))
        # print("obs time is {}, get {} from {}".format(k, get_num, len(dir_point[k])))
        sample_list = random.sample(dir_point[k], get_num)
        point_out += sample_list
    print("output number of points is {}".format(len(point_out)))
    return point_out


if __name__ == "__main__":
    # path = "/media/zlh/zhang/earth_rosbag/paper_data/evaluation/p8r2m3T3/t3bag4/evaluation/opti_for_hmm.txt"
    path = "/media/zlh/WD_BLACK/earth_rosbag/paper_data/t4bag2/evaluation/origin_for_hmm.txt"
    data = PointDataLoader(path)
    points_in = data.read_txt2list_points()
    point_out = down_sample_from_list(points_in)
    supervise_learn(point_out)
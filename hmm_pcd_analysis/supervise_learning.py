import data_loader as DL
from data_loader import save_point_list2txt
import open3d as o3d
import numpy as np
import filter

label_rgb = np.array([[0, 128, 255],   # none
                     [255, 0, 0],   # rock
                     [0, 255, 0]])   # sand


def generate_truth_labeled_rgb_pcd(pcd_path, save_path, name, if_gen_pcd=None):
    with open(pcd_path, 'r') as f:
        lines = f.readlines()

    # 获取点云数据的起始行
    for i, line in enumerate(lines):
        if line.startswith('DATA'):
            start_line = i + 1
            break
    # 获取点云数据
    data = []
    for line in lines[start_line:]:
        x, y, z, rgb, label, obj = line.split()
        data.append([float(x), float(y), float(z), int(label)-1])
    print(data[5])

    if if_gen_pcd:
        labeled_pcd = filter.LabelPCD( data )
        labeled_pcd.generate(save_path, name)
    # output [xyz, label]
    return data


def point_sample_labeling(truth_label_points, sample_points, save_path=None):
    """
    点的语义比对，赋值
    origin_label_points = [[xyz, label], ]
    sample_points = [xyz, ]
    """
    points = []  # x, y, z, label, obs_list
    origin_label_points_dir = {}
    for p in truth_label_points:
        o_name = ""
        for i in range(3):
            o_name = o_name + str(p[i])[:5]
        origin_label_points_dir[o_name] = p[3]

    for s in sample_points: # [xyz, num, init_label]
        s_name = ""
        for i in range(3):
            s_name = s_name + str(s[i])[:5]
        try:
            s.insert(3, origin_label_points_dir[s_name])
            points.append(s)
        except:
            print("can not find point!")
            continue

    print("we get {} points".format(len(points)))
    if save_path:
        save_point_list2txt(points, save_path)
    return points


def supervise_learn(point_list_with_real_L):
    """ [xyz, no, real_L, obs_list] """
    B = np.zeros([3, 7])
    for p in point_list_with_real_L:
        real_label = int(p[4])
        for i in set(p[5:]):
            B[real_label, int(i)] += p[5:].count(i)
    print(B)
    result_B = B / np.sum(B, axis=1)[:, np.newaxis]
    np.set_printoptions(suppress=True)
    print(np.around(result_B, decimals=5))


if __name__ == "__main__":
    obs_time = 6

    pcd_path = 'F:\earth_rosbag\data\\test3\labeled_pcd\origin11.pcd'
    save_path = 'F:\earth_rosbag\data\\test3\mix_pcd'
    # 配对标定点和txt中的点，给上颜色和观测序列，还能进行观测计算，监督学习
    origin_label_points = generate_truth_labeled_rgb_pcd(pcd_path, save_path, 'bag8_labeled', if_gen_pcd=1)

    sample_points = DL.PointDataLoader("/media/zlh/zhang/earth_rosbag/data/test3/r3live_4pixel/sample_bag8")
    sample_points.down_sample_loader()
    SP = sample_points.obs_downtimes_points_list(46)  # 最长保持46观测次数
    labeled_points = point_sample_labeling(origin_label_points, SP)
    # "F:\earth_rosbag\data\\test3\down_label_points_txt\\bag12.txt"
    supervise_learn(labeled_points)

    # mix pcd

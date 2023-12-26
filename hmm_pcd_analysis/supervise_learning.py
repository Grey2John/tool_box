import data_loader as DL
from data_loader import save_point_list2txt
import open3d as o3d
import numpy as np
import filter

label_rgb = np.array([[0, 128, 255],   # none
                     [255, 0, 0],   # rock
                     [0, 255, 0]])   # sand


def generate_truth_labeled_rgb_pcd(pcd_path, save_path, name, if_gen_pcd=None):
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

    if if_gen_pcd:
        labeled_pcd = filter.LabelPCD( data )
        labeled_pcd.generate(save_path, name)
    # output [xyz, label]
    return data


def point_sample_labeling(truth_label_points, sample_points, obs_start_index, save_path=None):
    """
    点的语义比对，赋值
    truth_label_points = [[xyz, truth label], ]
    sample_points = [xyz, ...]
    """
    points = []  # x, y, z, true label, obs_list
    origin_label_points_dir = {}
    for p in truth_label_points:
        o_name = ""
        for i in range(3):
            o_name = o_name + str(p[i])[:5]
        origin_label_points_dir[o_name] = p[3]  # true label

    for s in sample_points:  # [xyz, num, init_label, obs]
        s_name = ""
        for i in range(3):
            s_name = s_name + str(s[i])[:5]
        try:
            new_p = []
            new_p += s[0:3]  # xyz
            new_p.append(origin_label_points_dir[s_name])  # ground truth
            new_p.append(s[3])  # first label
            new_p += s[obs_start_index:]
            points.append(new_p)
        except:
            # print("can not find point!")
            continue  # x, y, z, true label, obs_list

    print("we get {} matched points".format(len(points)))
    if save_path:
        save_point_list2txt(points, save_path)
    return points


def supervise_learn(point_list_with_real_L):
    """ [xyz, real_L, obs_list] """
    B = np.zeros([3, 7])
    for p in point_list_with_real_L:
        real_label = int(p[3])
        for i in set(p[4:]):
            B[real_label, int(i)] += p[4:].count(i)
    print(B)
    result_B = B / np.sum(B, axis=1)[:, np.newaxis]
    np.set_printoptions(suppress=True)
    print(np.around(result_B, decimals=5))


if __name__ == "__main__":
    obs_time = 6

    pcd_path = 'F:\earth_rosbag\data\\test3\labeled_pcd\origin11.pcd'  # 真值
    save_path = 'F:\earth_rosbag\data\\test3\mix_pcd'
    # 配对标定点和txt中的点，给上颜色和观测序列，还能进行观测计算，监督学习
    origin_label_points = generate_truth_labeled_rgb_pcd(pcd_path, save_path, 'bag8_labeled', if_gen_pcd=0)

    sample_points = DL.PointDataLoader("F:\earth_rosbag\data\\test3\\r3live_4pixel\sample_bag11")  # 降采样的包
    sample_points.down_sample_loader()
    SP = sample_points.obs_downtimes_points_list(46)  # 最长保持46观测次数
    labeled_points = point_sample_labeling(origin_label_points, SP, 5)
    # "F:\earth_rosbag\data\\test3\down_label_points_txt\\bag12.txt"
    supervise_learn(labeled_points)

    # mix pcd

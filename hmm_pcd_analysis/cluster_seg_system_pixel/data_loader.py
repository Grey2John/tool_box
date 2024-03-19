import os
import argparse
import numpy as np
import random
import math


point_templet = {
    "xyz": None,
    "rgb": None,
    "no": None,
    "init_state": None,
    "obs_queue": []
}

""" 
2.12816, -0.132887, -0.364527, 147.447, 152.54, 160.904, 2880, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

{'xyz': [2.12816, -0.132887, -0.364527], 'rgb': [147.447, 152.54, 160.904], 
  'no': 2880, 'init_state': 0, 'obs_queue': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
"""


class PointDataLoader:
    def __init__(self, txt_path):
        self.txt_path = txt_path
        if not os.path.isfile(txt_path):
            self.file_list = os.listdir(txt_path)
            self.down_points_dir = {}

    def down_sample_loader(self):
        """
        {'xyz': [4.68743, 1.16662, -0.874653], 'init_state': 0, 'obs': [0, 0, 0, 0, 0, 0]}
        """
        for l in self.file_list:
            file = os.path.join(self.txt_path, l)
            # print("process {}".format(file))
            same_times_obs = []
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    one_list = line.split(', ')
                    one_point = {}
                    one_point['xyz'] = [float(str_num) for str_num in one_list[0:3]]
                    one_point['init_state'] = int(one_list[7])
                    one_point['no'] = int(one_list[6])
                    one_point['obs'] = []
                    for str in one_list[8:-1]:
                        one_point['obs'].append(int(str))
                    one_point['obs'].append(int(one_list[-1][0]))
                    same_times_obs.append(one_point)
            self.down_points_dir[len(one_list) - 4] = same_times_obs
        print('length is {}'.format(len(self.down_points_dir)))
        print('sample like {}'.format(self.down_points_dir[10][1]))

    def obs_downtimes_points_list(self, obs_time):
        """
        [{'xyz': [4.29805, 1.79933, 0.800642], 'obs': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, ...]
        所有点观测n次以内
        output:
        [xyz, no, label, obs]
        """
        point_cloud = []
        for key, value in self.down_points_dir.items():
            for p in value:
                one_point = p['xyz']
                one_point.append(p['no'])
                one_point.append(p['init_state'])
                if len(p['obs']) > obs_time:
                    one_point = one_point + p['obs'][:obs_time]
                else:
                    one_point = one_point + p['obs']
                point_cloud.append(one_point)
        print('length is {}'.format(len(point_cloud)))
        return point_cloud

    def read_txt_list_rgbpoints(self):
        """ xyz, rgb """
        points = []
        with open(self.txt_path, 'r') as f:
            lines = f.readlines()
            print("we got {} points".format(len(lines)))
            for line in lines:
                one_list = line.split(', ')
                one_point = [float(str_num) for str_num in one_list[0:6]]
                points.append(one_point)
        return points

    def read_txt_list_points(self, down_time=0, upper_times=None):
        """ xyz, first label, obs """
        points = []
        with open(self.txt_path, 'r') as f:
            lines = f.readlines()
            print("we got {} points".format(len(lines)))
            for line in lines:
                one_list = line.split(', ')
                if down_time != 0 and (len(one_list) - 8) <= down_time:
                    continue
                one_point = [float(str_num) for str_num in one_list[0:3]]
                one_point.append(int(one_list[7]))
                if len(one_list[8:]) > upper_times:
                    for str in one_list[8:upper_times + 8]:
                        one_point.append(int(str))
                else:
                    for str in one_list[8:-1]:
                        one_point.append(int(str))
                    one_point.append(int(one_list[-1][0]))
                points.append(one_point)
        return points

    def read_txt_dirpoints(self):
        """  """
        points = {}
        with open(self.txt_path, 'r') as f:
            lines = f.readlines()
            print("we got {} points".format(len(lines)))
            for line in lines:
                one_list = line.split(', ')
                point = point_templet.copy()
                point["xyz"] = [float(str_num) for str_num in one_list[0:3]]
                point["rgb"] = [float(str_num) for str_num in one_list[3:6]]
                point["no"] = int(one_list[6])
                point["init_state"] = int(one_list[7])
                point["obs"] = [int(str_num) for str_num in one_list[8:]]

                obs_num = len(one_list) - 8
                if obs_num in points.keys():
                    points[obs_num].append(point)
                else:
                    points[obs_num] = [point]

        print('length is {}'.format(len(points.keys())))
        print(points[10][1])
        return points

    def read_txt2list_points(self):
        """ [11.3883, 1.98356, 1.24666, None, 2, 2, 2]"""
        points = []
        with open(self.txt_path, 'r') as f:
            lines = f.readlines()
            print("we got {} points".format(len(lines)))
        for line in lines:
            cont = False
            one_list = line.split(', ')
            one_point = [float(str_num) for str_num in one_list[0:3]]  # xyz
            for obs in one_list[3:-1]:  # truth, obs
                if obs == 'None':
                    cont = True
                    break
                one_point.append(int(obs))  # first, filtered obs
            if cont:
                continue
            if 'None' in one_list[-1]:
                continue
            one_point.append(int(one_list[-1][0]))
            points.append(one_point)
        return points

    def read_txt_dic_points_with_obs_times(self, obs_index=8):
        """
        input txt: xyz, rgb, No, first_state, obs_state, obs_time, ...
        output two data:
        point list: [xyz, rgb, No, first_state, obs_state, obs_time, ...]
        index dic: {'obs No': [point index], ...}
        """
        points = []
        points_dic = {}
        point_num = 0
        with open(self.txt_path, 'r') as f:
            lines = f.readlines()
            print("we got {} points".format(len(lines)))
            for line in lines:
                one_list = line.split(', ')
                one_point = [float(str_num) for str_num in one_list[0:6]]  # xyz rgb
                one_point.append(int(one_list[6]))  # No
                one_point.append(int(one_list[7]))  # first state
                if len(one_list[obs_index:]) % 2 != 0:
                    continue
                one_list[-1] = one_list[-1].replace('\n', '')
                for i in range(int(len(one_list[obs_index:]) / 2)):
                    one_point.append(int(one_list[obs_index + 2 * i]))
                    one_point.append(int(one_list[obs_index + 2 * i + 1]))
                    if points_dic.get(int(one_list[obs_index + 2 * i + 1])) is None:
                        points_dic[int(one_list[obs_index + 2 * i + 1])] = [point_num]
                    else:
                        points_dic[int(one_list[obs_index + 2 * i + 1])].append(point_num)
                points.append(one_point)

                point_num += 1
            print("finish the points reading")
        return points, points_dic


def down_sample(origin_point, alpha1, save_path):
    """生成降采样的txt文件"""
    coeff = np.linspace(alpha1, 1, len(origin_point.keys())).tolist()
    print("length is {}".format(len(coeff)))
    keys = sorted(origin_point.keys())
    for i, k in enumerate(keys):
        get_num = math.ceil(coeff[i] * len(origin_point[k]))
        print("obs time is {}, get {} from {}".format(k, get_num, len(origin_point[k])))
        sample_list = random.sample(origin_point[k], get_num)

        file = os.path.join(save_path, str(k) + ".txt")
        with open(file, 'a') as f:
            for p in sample_list:
                one_line = ", ".join([str(x) for x in p["xyz"]])
                one_line = one_line + ", " + ", ".join([str(x) for x in p["rgb"]])
                one_line = one_line + ", " + str(p["no"]) + ", " + str(p["init_state"])
                one_line = one_line + ", " + ", ".join([str(x) for x in p["obs_queue"]])
                f.write(one_line + "\n")


def pcd_generate(points, save_path, name):
    point_cloud = np.array(points)
    pcd = filter.PointList2RGBPCD(point_cloud)
    pcd.generate(save_path, name)


def save_point_list2txt(points_list, save_path):
    file = os.path.join(save_path)
    if not os.path.exists(save_path):
        with open(file, 'w') as f:
            f.write("")
    else:
        print("the file exist")

    with open(file, 'a') as f:
        for p in points_list:
            one_line = ", ".join([str(x) for x in p])
            f.write(one_line + "\n")


if __name__ == "__main__":
    # read_tool = PointDataLoader("/home/zlh/data/sandpile_source/data/test1/pixel4_r3live/bag5.txt")
    # 生成降采样的点云文件
    # origin_point = read_txt_dirpoints("/home/zlh/bag12.txt")
    # down_sample(origin_point, 0.1, "/media/zlh/01839234-e52a-4a95-a67c-d336293595a8/zzh/data/test3/r3live_4pixel/sample_bag12")
    # 生成pcd文件

    # 原始点到pcd
    # read_tool = PointDataLoader("/home/zlh/data/sandpile_source/data/test1/pixel4_r3live/bag6.txt")
    # xyzrgb = read_tool.read_txt_list_rgbpoints()
    # print(xyzrgb[5])
    # pcd_generate(xyzrgb, '/home/zlh/data/sandpile_source/data/test1/origin_pcd_for_label', 'origin6')

    # 有观测序号的txt
    read_tool = PointDataLoader("/media/zlh/zhang/earth_rosbag/data/test3/obs_times_txt/bag12.txt")
    point, point_index = read_tool.read_txt_dic_points_with_obs_times()

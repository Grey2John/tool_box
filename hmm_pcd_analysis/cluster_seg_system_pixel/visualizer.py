import os
import sys
import open3d as o3d
from data_loader import PointDataLoader
# import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pcd_gen import PointList2RGBPCD as P2pcd

label_rgb = [[255, 0, 0], [0, 255, 0], [0, 0, 255],  # 0, 1, 2
             [255, 0, 255], [255, 255, 0], [0, 0, 0],  # 3, 4, 5
             [0, 255, 255]]  # 6
# label_rgb = [[0, 0, 255],[0, 0, 255],[0, 0, 255],   # 0, 1, 2
#              [0, 0, 255],[0, 0, 255],[0, 0, 255],   # 3, 4, 5
#              [0, 0, 255]]   # 6
state_standard_one = {
    0: [0],
    1: [1],
    2: [2],
    3: [0, 1],
    4: [0, 2],
    5: [1, 2],
    6: [0, 1, 2]
}


class Visualize:
    """单帧的图像的观测值的pcd"""

    def __init__(self, point_list, point_dic):
        self.point_list = point_list  #
        self.point_dic = point_dic

    def save_pcd_one_obs(self, save_path):
        points_k = sorted(self.point_dic.keys())
        for k in points_k:
            one_pcd_point = []
            for i in self.point_dic[k]:
                one_point = self.point_list[i][0:3]
                obs_index = 2
                for l in range(int(len(self.point_list[i][8:]) / 2)):
                    if int(self.point_list[i][8 + 2 * l + 1]) == k:
                        obs_index = 8 + 2 * l
                        break
                color = label_rgb[self.point_list[i][obs_index]]
                one_point = one_point + color
                one_pcd_point.append(one_point)
            save_class = P2pcd(one_pcd_point)
            save_class.generate(save_path, str(k))


class OutlierFilter:
    def __init__(self, point_list):
        """input a point list [xyz, obs_state] for one time observation"""
        self.point_list = point_list

    def distance(self, point1, point2):
        return


def get_pic(pcd_path, save_path, name):
    pcd = o3d.io.read_point_cloud(pcd_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.set_lookat([7.5, -2.2, -0.05])
    ctr.set_front([1, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.1)

    vis.run()
    save_file = os.path.join(save_path, '{}.png'.format(name))
    vis.capture_screen_image(save_file)
    vis.destroy_window()


def edge_mask(mask, pixel_set=4):
    rows, cols = mask.shape
    zero_matrix = np.zeros_like(mask)

    for h in range(rows):
        for w in range(cols):
            state = 0
            state_lib = []
            for i in range(-pixel_set, pixel_set + 1):
                for j in range(-pixel_set, pixel_set + 1):
                    if 0 <= (h + i) < rows and 0 <= (w + j) < cols:
                        state_pixel = mask[h + i, w + j]
                        if state_pixel not in state_lib:
                            state_lib.append(state_pixel)
            for j, v in state_standard_one.items():
                if v == sorted(state_lib):
                    state = j
                    break
            zero_matrix[h, w] = state
    return zero_matrix


def resize_image(original_matrix, scale_factor):
    original_height, original_width = original_matrix.shape
    new_height, new_width = original_height * scale_factor, original_width * scale_factor

    upscaled_matrix = np.zeros((new_height, new_width), dtype=original_matrix.dtype)

    for i in range(original_height):
        for j in range(original_width):
            value = original_matrix[i, j]
            upscaled_matrix[i * scale_factor:i * scale_factor + scale_factor,
            j * scale_factor:j * scale_factor + scale_factor] = value

    return upscaled_matrix


def plot_results(txt_father_path, bag_name):
    fig = plt.figure()
    for i, n in enumerate(bag_name):
        ax = fig.add_subplot(3, 5, i+1)
        path = os.path.join(txt_father_path, n, "evaluation")

        with open(os.path.join(path, "opti_eval_hmm.txt"), 'r') as file:
            whole_data_opti = [line.strip().split(', ') for line in file]
        whole_data_opti = np.array([[int(item[0]), int(item[1]), float(item[2]), float(item[3]),
                                     float(item[4]), float(item[5])] for item in whole_data_opti])

        with open(os.path.join(path, "origin_eval_hmm.txt"), 'r') as file:
            whole_data_origin = [line.strip().split(', ') for line in file]
        whole_data_origin = np.array([[int(item[0]), int(item[1]), float(item[2]), float(item[3]),
                                       float(item[4]), float(item[5])] for item in whole_data_origin])

        whole_data_opti = whole_data_opti[np.argsort(whole_data_opti[:, 0])]
        whole_data_origin = whole_data_origin[np.argsort(whole_data_origin[:, 0])]
        x_t1 = whole_data_opti[1:, 0]
        x_t2 = whole_data_origin[1:, 0]
        y_curve_opti = whole_data_opti[1:, [2, 4, 5]] * 100
        y_curve_origin = whole_data_origin[1:, [2, 4, 5]] * 100
        y_first_p = whole_data_origin[0, [2, 4, 5]] * 100

        ax.plot(1, y_first_p[0], 'rx', markersize=10)
        ax.plot(1, y_first_p[1], 'bx', markersize=10)
        ax.plot(1, y_first_p[2], 'gx', markersize=10)

        ax.plot(x_t1, y_curve_opti[:, 0], 'r', linewidth=1.5)
        ax.plot(x_t2, y_curve_origin[:, 0], 'r--', linewidth=1.5)

        ax.plot(x_t1, y_curve_opti[:, 1], 'b', linewidth=1.5)
        ax.plot(x_t2, y_curve_origin[:, 1], 'b--', linewidth=1.5)

        ax.plot(x_t1, y_curve_opti[:, 2], 'g', linewidth=1.5)
        ax.plot(x_t2, y_curve_origin[:, 2], 'g--', linewidth=1.5)
        ax.set_title('Case {}'.format(n))
        ax.set_xlabel('Number of Observations')
        ax.set_ylabel('Evaluation Indicators (%)')

    plt.show()


if __name__ == "__main__":
    # read_tool = PointDataLoader("/media/zlh/zhang/dataset/outline_seg_slam/test1/pt_obs.txt")
    # point, point_index = read_tool.read_txt_dic_points_with_obs_times()
    # vpcd = Visualize(point, point_index)
    # save_path = '/media/zlh/zhang/dataset/outline_seg_slam/test1/origin_pcd'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # vpcd.save_pcd_one_obs(save_path)
    bag_name = ["t3bag1","t3bag4","t3bag5","t3bag10","t4bag2","t4bag8","t4bag14",
                "t4bag17","t4bag19","t4bag22","t4bag24","t5bag0","t5bag1","t5bag9","t5bag10"]
    plot_results("/media/zlh/WD_BLACK/earth_rosbag/paper_data", bag_name)
    # plot_results("I:\earth_rosbag\paper_data", bag_name)

import numpy as np
import os
import open3d as o3d

import data_loader
import data_loader as DL

startprob = np.array([[0.4, 0.3, 0.3],   # none
                     [0.4, 0.3, 0.3],   # rock
                     [0.4, 0.3, 0.3]])   # sand
transmat = np.array([[0.985,	0.009,	0.0063],
                    [0.02845,	0.9711,	0.0004],
                    [0.0092,	0.0005,	0.9903]])
emissionprob = np.array([[0.9974, 0.00039, 0.0000017, 0.00118, 0.00102, 0,	0],
                      [0.12693,	0.75310, 0.000039, 0.11413, 0.00042, 0.00291, 0.00246],
                      [0.02638,	0.0004477, 0.78807, 0.00094, 0.1532, 0.00963, 0.02133]]) # matlab
# emissionprob = np.array([[0.9547,  0.0093,  0.00722, 0.0197,  0.00879, 0.00003, 0.00026],
#                       [0.32442, 0.6175, 0.00679, 0.03402, 0.00429, 0.0033, 0.00969],
#                       [0.45914, 0.02241, 0.47941, 0.01243, 0.02132, 0.00137, 0.00392]]) # matlab

# label_rgb = np.array([[0, 0.5, 1],   # none
#                         [1, 0, 0],   # rock
#                         [0, 1, 0]])   # sand
label_rgb = np.array([[0, 0, 255],[0, 255, 0],[255, 0, 0],   # 0, 1, 2
             [255, 0, 255],[0, 255, 255], [0, 0, 0],   # 3, 4, 5
             [255, 255, 0]])   # 6


class PointHMM:
    def __init__(self, n_components):
        self.n_components = n_components
        self.initial_state = None  # prob of 3 class
        self.trans_mat = transmat
        self.emission_prob = emissionprob

        self.current_state = None
        self.T_prob = None

    def forward(self, ys):
        """  输入观测序列
        :param ys:
        :return: [P(x0), P(x1,y1), P(x2,y1..y2) ... P(xn, y1..yn)] where x0 is initial_state
        """
        alpha_prob = np.multiply( self.initial_state, self.emission_prob[:, ys[0]].T) # t=1
        for i, y in enumerate(ys[1:]):
            alpha_prob = np.multiply( np.matmul(alpha_prob, self.trans_mat), self.emission_prob[:, y].T )
        self.current_state = alpha_prob
        return np.sum(alpha_prob)

    def filter(self, point_in):
        """from xyz,initial,obs_list to xyz,label"""
        self.initial_state = startprob[point_in[3], :]
        # self.initial_state = startprob[0, :]
        point_out = point_in[:3]

        self.T_prob = self.forward(point_in[4:])  # sum prob t=T
        label = np.argmax(self.current_state)
        point_out.append(int(label))
        return point_out

    def filter_all_list(self, point_in):
        """ 输入一个点的观测系列，输出这个点所有观测次的预测结果list
        还加入了第一次无观察的点
        """
        ys = point_in[4:]
        filtered_list = [point_in[3]]  # 还加入了第一次无观察的点
        self.initial_state = startprob[point_in[3], :]
        alpha_prob = np.multiply(self.initial_state, self.emission_prob[:, ys[0]].T)  # t=1
        label = np.argmax(alpha_prob)
        filtered_list.append(int(label))

        for i, y in enumerate(ys[1:]):
            alpha_prob = np.multiply(np.matmul(alpha_prob, self.trans_mat), self.emission_prob[:, y].T)
            label = np.argmax(alpha_prob)
            filtered_list.append(int(label))
        return filtered_list  # [first label, obs_list]


class LabelPCD:
    def __init__(self, point_cloud):
        """ point_could list [x,y,z,label]"""
        self.device = o3d.core.Device("CPU:0")
        self.dtype = o3d.core.float32
        self.pcd = o3d.t.geometry.PointCloud(self.device)

        self.points_array = np.array(point_cloud)
        self.points_label_list = [l[3] for l in point_cloud]
        self.color = np.zeros((self.points_array.shape[0], 3))

        self.color_render()

    def color_render(self):
        for i in range(self.points_array.shape[0]):
            self.color[i, :] = label_rgb[self.points_label_list[i], :]/255

    def generate(self, save_path, name):
        self.pcd.point.positions = o3d.core.Tensor(self.points_array[:, 0:3], self.dtype, self.device)
        self.pcd.point.colors = o3d.core.Tensor(self.color, self.dtype, self.device)

        save_file = os.path.join(save_path, '{}.pcd'.format(name))
        print("save path is {}".format(save_file))
        o3d.t.io.write_point_cloud(save_file, self.pcd, write_ascii=False)   # old


def one_obs_txt2pcd(txt_dir, save_path):
    for txt in os.listdir(txt_dir):
        if '.txt' in txt:
            print("process {}".format(txt))
            data = data_loader.PointDataLoader( os.path.join(txt_dir, txt) )
            points = data.read_txt_list_state_points()

            name = txt.split('.')[0]
            pcd = LabelPCD(points)
            pcd.generate(save_path, name)


class PointList2RGBPCD:
    """ 生成rbg的pcd """
    def __init__(self, point_cloud):
        """
        point_could list x,y,z,rgb(0~255), array
        """
        self.device = o3d.core.Device("CPU:0")
        self.dtype = o3d.core.Dtype.Float32
        self.pcd = o3d.t.geometry.PointCloud(self.device)
        self.point_cloud_array = np.array(point_cloud)
        self.points_array = self.point_cloud_array[:, 0:3]

        color = np.zeros((self.points_array.shape[0], 3))
        color[:, 0] = self.point_cloud_array[:, 5]   # 这里的顺序反过来的
        color[:, 1] = self.point_cloud_array[:, 4]
        color[:, 2] = self.point_cloud_array[:, 3]
        self.color = np.divide(color, 255)  # (0~1)

    def generate(self, save_path, name):
        self.pcd.point.positions = o3d.core.Tensor(self.points_array, self.dtype, self.device)
        self.pcd.point.colors = o3d.core.Tensor(self.color, self.dtype, self.device)

        save_file = os.path.join(save_path, '{}.pcd'.format(name))
        print("save path is {}".format(save_file))
        o3d.t.io.write_point_cloud(save_file, self.pcd)  # ascii **


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


def txt_HMM_pcd(point_set, save_path, name):
    """
    {'xyz': [4.68743, 1.16662, -0.874653], 'init_state': 0, 'obs': [0, 0, 0, 0, 0, 0]}
    读txt，经过hmm预测后，保存为pcd，所有点观测n次以内的效果
    """
    print("the length of point is {}".format(len(point_set)))
    point_for_pcd = []
    for p in point_set:
        point = PointHMM(3)
        point_for_pcd.append(point.filter(p))
    print("the length of label point is {}".format(len(point_for_pcd)))
    print(point_for_pcd[0])
    pcd = LabelPCD(point_for_pcd)
    pcd.generate(save_path, name)


def render_rbg_label_pcd(txt_path, save_path, name):
    """
    渲染pcd,将识别出来的语义换上其他颜色
    [xyz, rgb, label] to [xyz, rgb] pcd
    """
    data = DL.PointDataLoader(txt_path)
    points_in = data.read_txt_list_points()  # [xyz, rgb, ]
    txt_HMM_pcd(points_in, save_path, "filter1")
    points_out = [row[:4] for row in points_in]

    new_points = []
    for p in points_out:
        rgb_point = p[0:3]
        if p[4] != 0:
            new_mask = label_rgb[p[4], :]
            old_rgb = np.array(p[5:8])
            rgb_point = rgb_point + ((new_mask + old_rgb) / 2).tolist()
        else:
            rgb_point = rgb_point + p[5:8]
        new_points.append(rgb_point)  # [xyz, rgb]

    pcd = PointList2RGBPCD(new_points)
    pcd.generate(save_path, name)


def gamma(alpha, beta, t, i, N):
    """
    根据课本公式【10.24】计算γ
    :param t: 当前时间点
    :param i: 当前状态节点
    :return: γ值
    """
    numerator = alpha[t][i] * beta[t][i]
    denominator = 0.

    for j in range(N):
        denominator += (alpha[t][j] * beta[t][j])

    return numerator / denominator


if __name__ == "__main__":
    # 生成过滤和非过滤的pcd对比文件
    obs_time = 12
    down_txt_path = '/media/zlh/zhang/earth_rosbag/data/test4/pixel4/10.txt'
    save_path = '/media/zlh/zhang/earth_rosbag/data/test4/pcd'
    data = DL.PointDataLoader(down_txt_path)
    points_in = data.read_txt_list_points(obs_time, 100)
    txt_HMM_pcd(points_in, save_path, "filter10")
    points_out = [row[:4] for row in points_in]
    one_obs_pcd = LabelPCD( points_out )
    one_obs_pcd.generate(save_path, "non_filter10") # 观察一次的

    # save pic
    # pcd_path = os.path.join("F:\earth_rosbag\\test_hmm\data\pcd", str(obs_time)+".pcd")
    # get_pic(pcd_path, "F:\earth_rosbag\\test_hmm\data\pcd", str(obs_time))

    # mix pcd
    # save_path = 'F:\earth_rosbag\data\\test3\mix_pcd'
    # txt_path = 'F:\earth_rosbag\data\\test3\\r3live_4pixel\\bag11.txt'
    # render_rbg_label_pcd(txt_path, save_path, 'bag11')
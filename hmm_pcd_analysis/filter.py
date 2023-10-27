import numpy as np
import os
import open3d as o3d

import data_loader
import data_loader as DL

startprob = np.array([[0.6, 0.2, 0.2],   # none
                     [0.2, 0.6, 0.2],   # rock
                     [0.2, 0.2, 0.6]])   # sand
transmat = np.array([[0.99331,	0.00398,	0.00271],
                    [0.00378,	0.98598,	0.01025],
                    [0.00388,	0.01522,	0.98090]])
emissionprob = np.array([[0.89727, 0.00005, 0.00006, 0.05117, 0.05144, 0,	0],
                      [0.02670,	0.81723, 0.00195, 0.05120, 0.05051,	0.05047, 0.00194],
                      [0.00818,	0.00893, 0.82637, 0.05217, 0.10162, 0.00030, 0.00243]])

label_rgb = np.array([[0, 0.2, 1],   # none
                     [1, 0, 0],   # rock
                     [0, 1, 0]])   # sand


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
        point_out = point_in[:3]

        self.T_prob = self.forward(point_in[4:])  # sum prob t=T
        label = np.argmax(self.current_state)
        point_out.append(label)
        return point_out


class LabelPCD:
    def __init__(self, point_cloud):
        """ point_could list x,y,z,label """
        self.device = o3d.core.Device("CPU:0")
        self.dtype = o3d.core.float32
        self.pcd = o3d.t.geometry.PointCloud(self.device)

        self.points_array = np.array(point_cloud)
        self.points_label_array = [l[3] for l in point_cloud]
        self.color = np.zeros((self.points_array.shape[0], 3))

        self.color_render()

    def color_render(self):
        for i in range(self.points_array.shape[0]):
            self.color[i, :] = label_rgb[self.points_label_array[i], :]


    def generate(self, save_path, name):
        self.pcd.point.positions = o3d.core.Tensor(self.points_array[:, 0:3], self.dtype, self.device)
        self.pcd.point.colors = o3d.core.Tensor(self.color, self.dtype, self.device)

        save_file = os.path.join(save_path, '{}.pcd'.format(name))
        print("save path is {}".format(save_file))
        o3d.t.io.write_point_cloud(save_file, self.pcd, write_ascii=False)


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
    def __init__(self, point_cloud):
        """ point_could list x,y,z,rgb, array"""
        self.device = o3d.core.Device("CPU:0")
        self.dtype = o3d.core.float32
        self.pcd = o3d.t.geometry.PointCloud(self.device)
        self.point_cloud_array = np.array(point_cloud)
        self.points_array = self.point_cloud_array[:, 0:3]

        color = np.zeros((self.points_array.shape[0], 3))
        color[:, 0] = self.point_cloud_array[:, 5]
        color[:, 1] = self.point_cloud_array[:, 4]
        color[:, 2] = self.point_cloud_array[:, 3]
        self.color = np.divide(color, 255)

    def generate(self, save_path, name):
        self.pcd.point.positions = o3d.core.Tensor(self.points_array, self.dtype, self.device)
        self.pcd.point.colors = o3d.core.Tensor(self.color, self.dtype, self.device)

        save_file = os.path.join(save_path, '{}.pcd'.format(name))
        print("save path is {}".format(save_file))
        o3d.t.io.write_point_cloud(save_file, self.pcd, write_ascii=False)


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


if __name__ == "__main__":
    obs_time = 20

    down_txt_path = '/home/zlh/data/sandpile_source/data/test3/r3live_4pixel/bag1.txt'
    save_path = '/home/zlh/data/sandpile_source/data/test3/filter_pcd'
    data = DL.PointDataLoader(down_txt_path)
    points_in = data.read_txt_list_points(obs_time)
    txt_HMM_pcd(points_in, save_path, "filter1")
    points_out = [row[:4] for row in points_in]
    one_obs_pcd = LabelPCD( points_out )
    one_obs_pcd.generate(save_path, "non_filter1") # 观察一次的

    # save pic
    # pcd_path = os.path.join("F:\earth_rosbag\\test_hmm\data\pcd", str(obs_time)+".pcd")
    # get_pic(pcd_path, "F:\earth_rosbag\\test_hmm\data\pcd", str(obs_time))

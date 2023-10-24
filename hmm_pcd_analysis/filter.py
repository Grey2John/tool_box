import numpy as np
import os
import open3d as o3d

startprob = np.array([[0.8, 0.1, 0.1],   # none
                     [0.1, 0.8, 0.1],   # rock
                     [0.1, 0.1, 0.8]])   # sand
transmat = np.array([[0.99331,	0.00398,	0.00271],
                    [0.00378,	0.98598,	0.01025],
                    [0.00388,	0.01522,	0.98090]])
emissionprob = np.array([[0.99727, 0.00005, 0.00006, 0.00117, 0.00144, 0,	0],
                      [0.02670,	0.91723, 0.00195, 0.05120, 0.00051,	0.00047, 0.00194],
                      [0.00818,	0.00893, 0.89637, 0.00217, 0.08162, 0.00030, 0.00243]])

label_rgb = np.array([[0, 0, 255],   # none
                     [255, 0, 0],   # rock
                     [0, 255, 0]])   # sand


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


class DataLoader:
    def __init__(self, down_txt_path):
        self.down_txt_path = down_txt_path
        self.file_list = os.listdir(down_txt_path)
        self.down_points_dir = {}

        self.load()

    def load(self):
        for l in self.file_list:
            file = os.path.join(self.down_txt_path, l)
            # print("process {}".format(file))
            same_times_obs = []
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    one_list = line.split(', ')
                    one_point = {}
                    one_point['xyz'] = [float(str_num) for str_num in one_list[0:3]]
                    one_point['first_state'] = int(one_list[3])

                    one_point['obs'] = []
                    for str in one_list[4:-1]:
                        one_point['obs'].append(int(str))
                    one_point['obs'].append(int(one_list[-1][0]))
                    same_times_obs.append(one_point)
            self.down_points_dir[len(one_list)-4] = same_times_obs
        print('length is {}'.format(self.down_points_dir[10][1]))

    def obs_time_points_all(self, obs_time):
        """{'xyz': [4.29805, 1.79933, 0.800642], 'obs': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}"""
        point_cloud = []
        for key, value in self.down_points_dir.items():
            for p in value:
                one_point = p['xyz']
                one_point.append(p['first_state'])
                one_point = one_point + p['obs'][:obs_time]
                point_cloud.append(one_point)
        return point_cloud


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
        o3d.t.io.write_point_cloud(save_file, self.pcd, write_ascii=True)


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


if __name__ == "__main__":
    obs_time = 20

    # down_txt_path = 'F:\earth_rosbag\\test_hmm\data\down_sample_set_bag5'
    # data = DataLoader(down_txt_path)
    # point_set = data.obs_time_points_all(obs_time)
    # print("the length of point is {}".format(len(point_set)))
    # point_for_pcd = []
    # for p in point_set:
    #     point = PointHMM(3)
    #     point_for_pcd.append(point.filter(p))
    # print("the length of label point is {}".format(len(point_for_pcd)))
    # print(point_for_pcd[0])
    # pcd = LabelPCD(point_for_pcd)
    # pcd.generate("F:\earth_rosbag\\test_hmm\data\pcd", str(obs_time))

    # save pic
    pcd_path = os.path.join("F:\earth_rosbag\\test_hmm\data\pcd", str(obs_time)+".pcd")
    get_pic(pcd_path, "F:\earth_rosbag\\test_hmm\data\pcd", str(obs_time))

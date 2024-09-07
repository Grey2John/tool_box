import open3d as o3d
import os
import numpy as np
from filter import PointHMM
import data_loader as DL


label_rgb = np.array([[0, 0, 255],[0, 255, 0],[255, 0, 0],   # 0, 1, 2
             [255, 0, 255],[0, 255, 255], [100, 100, 100],   # 3, 4, 5
             [255, 255, 0]])   # 6


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


class LabelPCD:
    """input [x,y,z,label] list 好用的生成pcd """
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
    """read txt, generate pcd"""
    for txt in os.listdir(txt_dir):
        if '.txt' in txt:
            print("process {}".format(txt))
            data = DL.PointDataLoader( os.path.join(txt_dir, txt) )
            points = data.read_txt_list_state_points()

            name = txt.split('.')[0]
            pcd = LabelPCD(points)
            pcd.generate(save_path, name)


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
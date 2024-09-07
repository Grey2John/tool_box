from data_load import OutlineDataLoader, read_one_frame_data_json
import os
from label_system import one_frame_task, multi_frame_task
import argparse
from pcd_gen import txt_HMM_pcd, LabelPCD
from data_loader import PointDataLoader
from evaluation import PointHMMEvaluation


def output_pcd_all_frame(work_space):
    """
    :param work_space:
    :return:
    """
    data = OutlineDataLoader(work_space)
    all_image_data = data.data_out_put()
    del data
    print("\033[32m finish all data loading \033[0m")
    # all_image_data processing
    pcd_path = os.path.join(work_space, "pcd")
    if not os.path.exists(pcd_path):
        os.makedirs(pcd_path)
    all_image_data.process(pcd_path, edge_D=10)


def one_frame_test(json_path, save_path):
    """
    input: one frame json
    save the pcd
    """
    image_pose, points, point_origin_state = read_one_frame_data_json(json_path)
    # print("the number of points is {}".format(len(points)))
    one_frame_task(image_pose, points, image_pose.observing_num, point_origin_state, save_path)
    return None


def multi_frame_test(work_space):
    """multi-frame optimization by HMM filer and re-segmentation"""
    data = OutlineDataLoader(work_space)
    image_pose_data = data.data_out_put()
    del data
    multi_frame_task(image_pose_data, work_space)
    return True


def read_txt2pcd(path, obs_time_list, up_time=150):
    """txt [xyz, truth, first, obs state]"""
    origin_data = PointDataLoader(os.path.join(path, 'evaluation', 'origin_for_hmm.txt'))
    reseg_data = PointDataLoader(os.path.join(path, 'evaluation', 'opti_for_hmm.txt'))
    origin_points_in = origin_data.read_txt2list_points()
    reseg_points_in = reseg_data.read_txt2list_points()
    origin_hmm = PointHMMEvaluation(origin_points_in)
    origin_hmm.filter_process()
    reseg_hmm = PointHMMEvaluation(reseg_points_in)
    reseg_hmm.filter_process()
    # self.point_list is [[xyz, truth, first label, filter_label_list], ...]
    save_p = os.path.join(path, 'result_pcd')

    pcd = LabelPCD( [[p[0],p[1],p[2],p[4]] for p in origin_hmm.point_list] )
    pcd.generate(save_p, "origin")
    for t in obs_time_list:
        new_point_list = []
        for p in origin_hmm.point_list:
            one_p = p[0:3]
            if len(p) >= t+4:
                one_p.append(p[t+3])
            else:
                one_p.append(p[-1])
            new_point_list.append(one_p)
        pcd = LabelPCD(new_point_list)
        pcd.generate(save_p, "hmm{}".format(t))

        new_point_list = []
        for p in reseg_hmm.point_list:
            one_p = p[0:3]
            if len(p) >= t+4:
                one_p.append(p[t + 3])
            else:
                one_p.append(p[-1])
            new_point_list.append(one_p)
        pcd = LabelPCD(new_point_list)
        pcd.generate(save_p, "hmm_reseg{}".format(t))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input the ')
    parser.add_argument('-w', '--work_path', type=str,
                        help='input the bag workspace',
                        default="/media/zlh/zhang/earth_rosbag/paper_data/t4bag14")
    parser.add_argument('-p', '--pcd_frame', type=int,
                        help='-p load txt, hmm calculate, and save pcd',
                        nargs='+') #, default=[20, 50])
    parser.add_argument('-o', '--one_frame', type=str,
                        help='-w read_path -o target_json save_place',
                        nargs=2)
    args = parser.parse_args()
    # work_space = "/media/zlh/zhang/dataset/outline_seg_slam/bag5"
    # output_pcd_all_frame(work_space)
    # one frame，产生单帧分析的四个图像
    # one_frame_test("I:\earth_rosbag\paper_data\\t4bag3\one_frame\\120.json",
    #                "I:\earth_rosbag\paper_data\\t4bag3\one_frame")
    # one_frame_test("/media/zlh/WD_BLACK/earth_rosbag/paper_data/t3bag10/one_frame/120.json",
    #                "/media/zlh/WD_BLACK/earth_rosbag/paper_data/t3bag10/one_frame")

    # multi frame
    # multi_frame_test("/media/zlh/zhang/earth_rosbag/paper_data/t3bag1")
    # multi_frame_test("I:\earth_rosbag\paper_data\\t3bag1")
    if args.work_path and not args.pcd_frame:
        multi_frame_test(args.work_path)  # python main.y -w xxx/paper_data/t4bag14
    # pcd generate
    elif args.work_path and args.pcd_frame:
        read_txt2pcd(args.work_path, args.pcd_frame)  # 读取txt，输出pcd文件，以便于可视化

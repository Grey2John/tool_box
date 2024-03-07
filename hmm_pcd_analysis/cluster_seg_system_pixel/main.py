from data_load import OutlineDataLoader, read_one_frame_data_json
import os
from label_system import one_frame_task, multi_frame_task
import argparse

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
    all_image_data.process(pcd_path)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input the ')
    parser.add_argument('-p', '--path', type=str,
                        help='input the bag workspace',
                        default="/media/zlh/zhang/earth_rosbag/paper_data/t3bag1")
    args = parser.parse_args()
    # work_space = "/media/zlh/zhang/dataset/outline_seg_slam/bag5"
    # output_pcd_all_frame(work_space)
    # one frame
    # one_frame_test("F:\dataset\outline_seg_slam\\bag2\one_frame\\35.json",
    #                "F:\dataset\outline_seg_slam\\bag2\one_frame")
    # multi frame
    # multi_frame_test("/media/zlh/zhang/earth_rosbag/paper_data/t3bag1")
    multi_frame_test(args.path)

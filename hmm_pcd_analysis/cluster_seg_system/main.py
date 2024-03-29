from data_load import OutlineDataLoader, read_one_frame_data_json
import os
from label_system import one_frame_task


def output_pcd_all_frame(work_space):
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
    """input: one frame json
    save the pcd
    """
    image_pose, points, point_origin_state = read_one_frame_data_json(json_path)
    # print("the number of points is {}".format(len(points)))
    one_frame_task(image_pose, points, image_pose.observing_num, point_origin_state, save_path)
    return None


if __name__ == "__main__":
    # work_space = "/media/zlh/zhang/dataset/outline_seg_slam/bag5"
    # output_pcd_all_frame(work_space)

    one_frame_test("/media/zlh/zhang/earth_rosbag/paper_data/t4bag22/one_frame/95.json",
                   "/media/zlh/zhang/earth_rosbag/paper_data/t4bag22/one_frame")

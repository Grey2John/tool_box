from data_load import OutlineDataLoader
import os


if __name__ == "__main__":
    work_space = "/media/zlh/zhang/dataset/outline_seg_slam/test1"
    data = OutlineDataLoader(work_space)
    all_image_data = data.data_out_put()
    del data
    print("\033[32m finish all data loading \033[0m")
    # all_image_data processing
    pcd_path = os.path.join(work_space, "pcd")
    if not os.path.exists(pcd_path):
        os.makedirs(pcd_path)
    all_image_data.process(pcd_path)
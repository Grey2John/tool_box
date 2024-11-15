import os
import json
from data_load import OutlineDataLoader
"""没有必要对每个帧做分割，直接在总体数据集上做提取的时候评价即可"""

def frame_seg_eval(work_space):
    """run multi-frame data, evaluate each frame
    1. time cost
    2. seg accuracy
    3. point num
    4. mapping deviation"""
    work_space = os.path.normpath(work_space)
    data = OutlineDataLoader(work_space)
    image_pose_data = data.data_out_put()
    del data
    image_pose_data.process_eval2each_frame(work_space)
    return True


def frame_data_generate(work_space, save_path):
    """generate the frame data from the entire data. for training other network"""
    data = OutlineDataLoader(work_space)
    image_pose_data = data.data_out_put()
    del data
    # each_frame_eval_task(image_pose_data, work_space)
    return True


if __name__ == "__main__":
    frame_seg_eval("/media/zlh/eData/doctor_proj/earth_rosbag/paper_data/t3bag4")
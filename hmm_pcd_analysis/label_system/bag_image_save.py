import rosbag
import os
import cv2
from cv_bridge import CvBridge
# 完成bag中的图像保存，方便离线作点云分割


def rosbag_rgba_to_image(bag, save_path, rgb_name="rgb", mask_name="mask"):
    """change the bag to the RGB image and mask"""
    rgb_dir = os.path.join(save_path, rgb_name)
    mask_dir = os.path.join(save_path, mask_name)
    if not os.path.exists(rgb_dir):
        os.makedirs(rgb_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    bridge = CvBridge()
    with rosbag.Bag(bag, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            # Assuming the image message is of type sensor_msgs/Image
            if topic == "/yolov5/segment/bgra":
                seq = msg.header.seq
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgra8')
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2RGB)[:, 160:1120, :]
                alpha_channel = cv2.split(cv_image)[3][:, 160:1120]

                rgb_path = os.path.join(rgb_dir, f"{seq}.jpg")
                print(rgb_path)
                cv2.imwrite(rgb_path, rgb_image)
                mask_path = os.path.join(mask_dir, f"{seq}.png")
                print(mask_path)
                cv2.imwrite(mask_path, alpha_channel)


if __name__ == "__main__":
    rosbag_rgba_to_image('/media/zlh/zhang/dataset/outline_seg_slam/test1/yolo.bag',
                         '/media/zlh/zhang/dataset/outline_seg_slam/test1')
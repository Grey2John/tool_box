# conda activate aaa
import os
import sys
import re
import argparse
import cv2
import numpy as np
import time
import rosbag
from cv_bridge import CvBridge, CvBridgeError


class BagProcess:
    def __init__(self, bag_path, save_path, item):
        bag_data = rosbag.Bag(bag_path, "r")
        self.bag_data = bag_data.read_messages()
        self.item = item
        self.topic = '/camera/color/image_raw'
        self.num = 0
        self.save_path = save_path

    def save_an_image(self, image):
        name = 'image_' + str(self.item).zfill(2) + '_' + str(self.num).zfill(5) + ".jpg"
        image_path = os.path.join(self.save_path, name)
        print("save to {}".format(image_path))
        cv2.imwrite(image_path, image)

    def test_bag(self):
        bridge = CvBridge()
        for topic, msg, t in self.bag_data:
            if topic == self.topic:
                print(dir(msg))
                print(msg.header)
                print(msg.height)
                print(msg.width)
                print(msg.encoding)  # rgb8
                cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
                print(type(cv_image))
                cv_image = rgb_change(cv_image)
                cv2.imshow("Image window", cv_image)
                key = cv2.waitKey(0)
                break

    def analyze_bag_t(self, time_step):
        bridge = CvBridge()
        head_sec = 0
        for topic, msg, t in self.bag_data:
            if topic == self.topic:
                head_sec_new = time_process(msg.header.stamp.secs, msg.header.stamp.nsecs, 2)
                if head_sec_new - head_sec > time_step:
                    print("time_step: {}".format(head_sec_new))
                    cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
                    cv_image = rgb_change(cv_image)
                    head_sec = head_sec_new
                    self.num += 1
                    self.save_an_image(cv_image)

    def analyze_bag_f(self, frame_step):
        bridge = CvBridge()
        head_frame = 0
        frame = 0
        for topic, msg, t in self.bag_data:
            if topic == self.topic:
                head_frame += 1
                frame += 1
                if head_frame >= frame_step:
                    cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
                    cv_image = rgb_change(cv_image)
                    print("frame: {}".format(frame))
                    head_frame = 0
                    self.num += 1
                    self.save_an_image(cv_image)


def rgb_change(iamge_matrix):
    new_image = iamge_matrix.copy()
    new_image[:, :, 2] = iamge_matrix[:, :, 0]
    new_image[:, :, 0] = iamge_matrix[:, :, 2]
    return new_image


def time_process(secs, nsecs, n):
    a = str(nsecs)[:n]
    time_out = secs + float(a) / 10 ** n
    return time_out


def single_process(path, args, item):
    try:
        bag_process = BagProcess(path, args.save_path, item)
    except:
        print("can not open the rosbag {}".format(path))
        return 0
    if args.frame is None:
        # bag_process.test_bag()
        bag_process.analyze_bag_t(float(args.time))
    else:
        bag_process.analyze_bag_f(args.frame)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='this is a script of image shotting in rosbag ')
    parser.add_argument('-p', '--path', nargs='+',help='xx.py -p bag_path')
    parser.add_argument('-s', '--save_path', help='xx.py -s save_path')
    parser.add_argument('-t', '--time', type=str, help='xx.py -t [time_interval]')
    parser.add_argument('-f', '--frame', type=int, help='xx.py -f [frame_interval]')
    args = parser.parse_args()

    if args.path is None:
        print("you do not input the rosbag file. exit!")
        sys.exit()
    elif args.time is None and args.frame is None:
        print("you do not input the interial setting. exit!")
        sys.exit()

    item = 1
    if isinstance(args.path, list) is True:
        # paths = os.listdir("/home/zlh/data/earth_rosbag/sand_dataset/test1")
        # print(paths)
        for path in args.path:
            # file_path = os.path.join("/home/zlh/data/earth_rosbag/sand_dataset/test1", path)
            done = single_process(path, args, item)
            item += done
    else:
        single_process(args.path, args, item)

# python rosbag_shot.py -p ~/data/earth/20230411_160758.bag -s /home/zlh/data/earth/t1/*.bag -t 1
# python rosbag_shot.py -p ~/data/earth/20230411_160758.bag -s ~/data/personal1/ -f 20

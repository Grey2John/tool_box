import os
import rosbag
import sys
import argparse
from cv_bridge import CvBridge

def cut(bag_path, time_start, time_end, output_bag):
    cut_string = 'rosbag filter {} {} "t.to_sec() >= {} and t.to_sec() <= {}"'.format(
        bag_path,
        output_bag,
        time_start,
        time_end
    )
    print(cut_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-b', '--bag', type=int, required=True, help='ros bag file path')
    parser.add_argument('--time_start', help='start timestamp')
    parser.add_argument('--time_end', help='end timestamp')
    parser.add_argument('-o', '--output', help='end timestamp')
    args = parser.parse_args()

    input_bag = args.bag
    output_bag = args.output

    cut(input_bag, args.time_start, args.time_end, output_bag)
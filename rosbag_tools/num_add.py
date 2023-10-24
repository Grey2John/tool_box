import os
import argparse


def move_file(source, target, n):
    image_list = os.listdir(source)
    for image in image_list:
        name_list = image.split('_')
        num = int(name_list[1]) + int(n)
        name_list[1] = str(num).zfill(2)
        new_name = "_".join(name_list)

        path1 = os.path.join(source, image)
        path2 = os.path.join(target, new_name)
        os.system('mv {} {}'.format(path1, path2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='distribute the images and labels')
    parser.add_argument('-s', '--source', help='xx.py -s image_path')
    parser.add_argument('-t', '--target', help='xx.py -t path')
    parser.add_argument('-n', '--num', help='xx.py -n 16')
    args = parser.parse_args()

    move_file(args.source, args.target, args.num)

# distribute the images and labels 将数据和标签分为训练集、验证集、测试集
import os
import math
import sys
import random
import argparse


class DataDistribution:
    def __init__(self, args):
        self.source_image_path = args.source[0]
        self.source_label_path = args.source[1]
        self.target_path = args.target

        self.target_train_iamge_path = os.path.join(args.target, 'train_image')
        self.target_train_label_path = os.path.join(args.target, 'train_label')
        self.target_val_iamge_path = os.path.join(args.target, 'val_image')
        self.target_val_label_path = os.path.join(args.target, 'val_label')
        self.target_test_iamge_path = os.path.join(args.target, 'test_image')
        self.target_test_label_path = os.path.join(args.target, 'test_label')

        self.percent = args.percent

    def copy_process(self, source_list, target_path_image, target_path_label):
        for l in source_list:
            txt_name = l.split('.')[0] + '.txt'
            # print('cp {} {}'.format(os.path.join(self.source_image_path, l), target_path_image))
            # print('cp {} {}'.format(os.path.join(self.source_label_path, txt_name), target_path_label))
            os.system('cp {} {}'.format(os.path.join(self.source_image_path, l), target_path_image))
            os.system('cp {} {}'.format(os.path.join(self.source_label_path, txt_name), target_path_label))
            print("{} is done".format(l))
            print("{} is done".format(txt_name))

    def file_group(self):
        image_list = os.listdir(self.source_image_path)  # image_02_00007.jpg image_02_00008.txt
        # print(image_list)
        label_list = os.listdir(self.source_label_path)
        image_copy_dir = {}
        train_set = []
        val_set = []
        test_set = []

        for image in image_list:
            if image.split('.')[-1] != 'jpg':
                continue
            num = image.split('_')[1]
            txt_name = image.split('.')[0] + '.txt'
            if num not in image_copy_dir.keys():
                image_copy_dir[num] = []
            if txt_name in label_list:
                image_copy_dir[num].append(image)

        for key, group in image_copy_dir.items():
            random.shuffle(group)
            num_test = math.ceil(len(group)*float(self.percent[2]))
            num_val = math.ceil(len(group)*float(self.percent[1]))
            test_set += group[:num_test]
            val_set += group[num_test:num_val+num_val]
            train_set += group[num_val+num_val:]

        return train_set, val_set, test_set

    def copy(self):
        train_set, val_set, test_set = self.file_group()
        # mkdir
        if not os.path.exists(self.target_train_iamge_path):
            os.system('mkdir -p {} {} {} {} {} {}'.format(self.target_train_iamge_path,
                                                          self.target_train_label_path,
                                                          self.target_val_iamge_path,
                                                          self.target_val_label_path,
                                                          self.target_test_iamge_path,
                                                          self.target_test_label_path))
            print("generate the target dateset directory Structure")
        # copy
        self.copy_process(train_set, self.target_train_iamge_path, self.target_train_label_path)
        self.copy_process(val_set, self.target_val_iamge_path, self.target_val_label_path)
        self.copy_process(test_set, self.target_test_iamge_path, self.target_test_label_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='distribute the images and labels')
    parser.add_argument('-s', '--source', nargs=2, help='xx.py -s image_path label_path')
    parser.add_argument('-t', '--target', help='xx.py -t path')
    parser.add_argument('-p', '--percent', nargs=3, help='xx.py -p 0.8 0.1 0.1')
    args = parser.parse_args()

    dd = DataDistribution(args)
    dd.copy()

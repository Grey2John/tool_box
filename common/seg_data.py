import os
import file_read
import random
import sys
import shutil

# 分割模型训练的数据集


def seg_and_save_data_yolov5(source_path, save_path, train_prop=0.7):
    source_image = os.path.join(source_path, 'train/images')
    image_file_list = file_read.dir_multi_file(source_image, "jpg")
    random.seed(42)
    random.shuffle(image_file_list)
    split_point = int(train_prop * len(image_file_list))
    image_train_list = image_file_list[:split_point]
    image_valid_list = image_file_list[split_point:]

    save_path_train_images = os.path.join(save_path, "train/images")
    save_path_train_labels = os.path.join(save_path, "train/labels")
    save_path_valid_images = os.path.join(save_path, "valid/images")
    save_path_valid_labels = os.path.join(save_path, "valid/labels")
    if not os.path.exists(save_path_train_images):
        os.makedirs(save_path_train_images)
        os.makedirs(save_path_train_labels)
        os.makedirs(save_path_valid_images)
        os.makedirs(save_path_valid_labels)

    for image in image_train_list:
        base_path, _ = os.path.splitext(image)
        base_path = base_path.replace("/images", "/labels")
        label_path = f"{base_path}{'.txt'}"
        print(image)
        shutil.copy(image, save_path_train_images)
        print(label_path)
        shutil.copy(label_path, save_path_train_labels)

    for image_v in image_valid_list:
        base_path, _ = os.path.splitext(image_v)
        base_path = base_path.replace("/images", "/labels")
        label_path_v = f"{base_path}{'.txt'}"
        print(image_v)
        shutil.copy(image_v, save_path_valid_images)
        print(label_path_v)
        shutil.copy(label_path_v, save_path_valid_labels)


if __name__ == "__main__":
    source = "/media/zlh/zhang/dataset/sandpile_source/master/yolo/augment"
    save_path = "/media/zlh/zhang/dataset/sandpile_source/master/yolo/augment_seg"
    seg_and_save_data_yolov5(source, save_path)
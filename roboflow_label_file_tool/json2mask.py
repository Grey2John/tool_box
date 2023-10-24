# task 1: change the json to single json to each picture
# task 2: generate the mask
from __future__ import print_function

import argparse
import glob
import json
import os
import os.path as osp
import sys

import cv2
import imgviz
import numpy as np
import PIL.Image

import labelme

mask_json = {
    "image_name": "",
    "shapes": [

    ]
}

def checkFileLost(dir1, dir2):
    """查缺少了那些文件"""
    # dir1 is the big one
    name1 = []
    for f1 in os.listdir(dir1):
        name1.append(".".join(f1.split(".")[:-1]))
    name2 = []
    for f2 in os.listdir(dir2):
        name2.append(".".join(f2.split(".")[:-1]))

    for l in name1:
        if l not in name2:
            print(l)


class_dic = {"1": "gravelpile", "2": "sandpile"}
class_dic_verse = {"gravelpile": 1, "sandpile": 2}
def transOneJson2MultiJson(OneJson_file, MultiJson_path):
    try:
        one_json = json.load(open(OneJson_file))
    except:
        print(OneJson_file)
        print("input the wrong json file, or no json file")
        sys.exit()

    image_id = 0
    block_mask_json = {"image_name": "", "shapes": []}
    for i, dic in enumerate(one_json["annotations"]):
        # if dic["category_id"] == 2 or dic["category_id"] == 5: #############
        shape_one = {
            "label": None,
            "points": None
        }
        point_list = xyxy2xy(dic["segmentation"][0])
        shape_one["label"] = class_dic[str(dic["category_id"])]
        shape_one["points"] = point_list
        # find img name
        image_name = ""
        if one_json["images"][image_id]["id"] == dic["image_id"]:
            image_name = one_json["images"][image_id]["file_name"]
        else:
            for line in one_json["images"]:
                if line["id"] == dic["image_id"]:
                    image_name = line["file_name"]
                    break
        if image_name == "":
            print("can not find {} file name".format(image_id))
            continue

        # judge the file exist
        img_path_now = os.path.join(MultiJson_path, (image_name[:-4] + ".json"))
        if os.path.exists(img_path_now):
            with open(img_path_now, "r") as fr:
                block_mask_json = json.load(fr)
        else:
            block_mask_json = {"image_name": "", "shapes": []}
            block_mask_json["image_name"] = image_name
        block_mask_json["shapes"].append(shape_one)
        image_id = dic["image_id"]
        with open(img_path_now, "w") as f:
            json.dump(block_mask_json, f, indent=4)
        print("generate a json NO.{}".format(image_id))


def xyxy2xy(xylist):
    xy_multi_list = []
    for i in range(0, int(len(xylist)/2)):
        xy_multi_list.append([xylist[2*i], xylist[2*i+1]])
    return xy_multi_list


def maskGenerate(args):
    """copy 来的"""
    if osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, 'JPEGImages'))
    os.makedirs(osp.join(args.output_dir, 'SegmentationClass'))
    os.makedirs(osp.join(args.output_dir, 'SegmentationClassPNG'))
    if not args.noviz:
        os.makedirs(
            osp.join(args.output_dir, 'SegmentationClassVisualization')
        )
    print('Creating dataset:', args.output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)
    class_names = tuple(class_names)
    print('class_names:', class_names)
    out_class_names_file = osp.join(args.output_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    for label_file in glob.glob(osp.join(args.input_dir, '*.json')):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            out_img_file = osp.join(
                args.output_dir, 'JPEGImages', base + '.jpg')
            out_lbl_file = osp.join(
                args.output_dir, 'SegmentationClass', base + '.npy')
            out_png_file = osp.join(
                args.output_dir, 'SegmentationClassPNG', base + '.png')
            if not args.noviz:
                out_viz_file = osp.join(
                    args.output_dir,
                    'SegmentationClassVisualization',
                    base + '.jpg',
                )

            data = json.load(f)

            img_file = osp.join(osp.dirname(label_file), data['imagePath'])
            img = np.asarray(PIL.Image.open(img_file))
            PIL.Image.fromarray(img).save(out_img_file)

            lbl = labelme.utils.shapes_to_label(
                img_shape=img.shape,
                shapes=data['shapes'],
                label_name_to_value=class_name_to_id,
            )
            labelme.utils.lblsave(out_png_file, lbl)

            np.save(out_lbl_file, lbl)

            if not args.noviz:
                viz = imgviz.label2rgb(
                    label=lbl,
                    img=imgviz.rgb2gray(img),
                    font_size=15,
                    label_names=class_names,
                    loc='rb',
                )
                imgviz.io.imsave(out_viz_file, viz)


class SingleMaskGenerate:
    def __init__(self, pic_path, json_path):
        self.pic_path = pic_path
        if os.path.exists(pic_path) and os.path.exists(json_path):
            img = cv2.imread(pic_path)
            self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.json = json.load(open(json_path))
            self.height = img.shape[0]
            self.width = img.shape[1]
            # self.mask = np.zeros((self.height, self.width), dtype=np.uint8)

        else:
            print("can not find json or pic")

    def generate(self):
        img_small = np.zeros((720, 960), dtype=np.uint8)
        for l in self.json["shapes"]:
            # img = np.zeros_like(self.img)
            img = np.zeros((self.height, self.width), dtype=np.uint8)
            label = class_dic_verse[l["label"]]
            points = l["points"]
            pts = np.array(points, np.int32)
            cv2.fillPoly(img, [pts], (int(label)*50))
            img = img[:, int((self.width-960)/2):int((self.width-960)/2+960)]
            for i, m in enumerate(img):
                for j, n in enumerate(m):
                    if n != 0:
                        img_small[i, j] = n
        # cv2.imshow('Original', self.mask)
        # cv2.waitKeyEx()
        return img_small

    def save_pipeline(self, mask, save_path):
        save_name = ".".join(self.pic_path.split("/")[-1].split(".")[:-1]) + ".png"
        cv2.imwrite(os.path.join(save_path, save_name), mask)

def multi_generate(args):
    i = 0
    tup_list = []
    for j in os.listdir(args.input_json):
        json_path_one = os.path.join(args.input_json, j)
        pic_path_one = os.path.join(args.input_pic, ".".join(j.split(".")[:-1])+".jpg")
        if os.path.exists(pic_path_one):
            tup_list.append((pic_path_one, json_path_one))

    for image_path, json_path in tup_list:
        sg = SingleMaskGenerate(image_path, json_path)
        mask = sg.generate()
        sg.save_pipeline(mask, args.output_dir)
        print("save one png mask: {}".format(image_path))
        i += 1
        print(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_pic', help='input pic directory')
    parser.add_argument('--input_json', help='input json directory')
    parser.add_argument('-o', '--output_dir', help='output dataset directory')
    # parser.add_argument('--labels', help='labels file')
    # parser.add_argument(
    #     '--noviz', help='no visualization', action='store_true'
    # )

    parser.add_argument('--json_file', help='one unit json file')
    parser.add_argument('--json_save', help='save path of json')
    args = parser.parse_args()

    if args.output_dir:
        multi_generate(args)
    elif args.json_file:
        transOneJson2MultiJson(args.json_file, args.json_save)

    # checkFileLost(
    #     "/media/zlh/zhang/dataset/sandpile_source/from_roboflow/non_augment/sandpile.v12i.coco-segmentation_full_train/train/img",
    #     "/media/zlh/zhang/dataset/sandpile_source/from_roboflow/non_augment/sandpile.v12i.coco-segmentation_full_train/train/png"
    # )
# python json2mask.py --json_file /media/zzh/zhang/dataset/sandpile_source/from_roboflow/new_pic/sand.v1i.coco-segmentation/test/_annotations.coco.json --json_save /media/zzh/zhang/dataset/sandpile_source/from_roboflow/new_pic/sand.v1i.coco-segmentation/test/tes
# python json2mask.py --input_pic /media/zlh/zhang/dataset/sandpile_source/from_roboflow/new_pic/sand.v1i.coco-segmentation/valid/img --input_json /media/zlh/zhang/dataset/sandpile_source/from_roboflow/new_pic/sand.v1i.coco-segmentation/valid/test -o /media/zlh/zhang/dataset/sandpile_source/from_roboflow/new_pic/sand.v1i.coco-segmentation/valid/png
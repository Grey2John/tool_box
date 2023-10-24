import os

select_label = [1, 4] # gravelpile, sandpile

def change_txt(files_path, save_path):
    txt_files = os.listdir(files_path)
    for f in txt_files:
        f_path = os.path.join(files_path, f)
        print("the file is {}".format(f_path))
        with open(f_path, 'r') as ff:
            lines = ff.readlines()
        
        read_to_save = []
        for l in lines:
            if int(l[0]) == select_label[0]:
                ll = list(l)
                ll[0] = "0"
                lll = "".join(ll)
                read_to_save.append(lll)
            elif int(l[0]) == select_label[1]:
                ll = list(l)
                ll[0] = "1"
                lll = "".join(ll)
                read_to_save.append(lll)

        with open(os.path.join(save_path, f), 'w') as s:
            for line in read_to_save:
                s.write(line)


def count_label(files_path):
    txt_files = os.listdir(files_path)
    count_dir = {}
    for f in txt_files:
        f_path = os.path.join(files_path, f)
        print("the file is {}".format(f_path))
        with open(f_path, 'r') as ff:
            lines = ff.readlines()

        for l in lines:
            if l[0] not in count_dir.keys():
                count_dir[ l[0] ] = 1
            else:
                count_dir[l[0]] += 1
    print(count_dir)


if __name__ == "__main__":
    # path = "/media/zlh/zhang/dataset/sandpile_source/from_roboflow/pic_first/sandpile_origin_yolov5pytorch/valid/labels"
    # save = "/home/zlh/data/sandpile_source/master/yolo/origin/valid/labels"
    # change_txt(path, save)
    path = '/home/zlh/data/sandpile_source/master/yolo/origin/valid/labels'
    count_label(path)
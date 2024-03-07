import os
import json


def add_save_list2txt(list_data, save_name):
    """input [xxx]
    add line to the txt"""
    with open(save_name, "a") as f:
        for a in list_data:
            if isinstance(a, int):
                s = str(a)
            else:
                s = "{:.8f}".format(a)
            f.write(s + ', ')
        f.write('\n')


def save_list2txt(list_data, save_name):
    """input [xxx]
    add line to the txt"""
    with open(save_name, "a") as f:
        for sublist in list_data:
            line = ', '.join(str(element) for element in sublist)
            f.write(line + '\n')
import os, sys, shutil
from PIL import Image
import random

path = "/home/renge/Pycharm_Projects/ISSBA/datasets/cifar10-bd/inject_a"
target_path = "/mnt/data03/renge/public_dataset/pytorch/model_lock-data/cifar10-StegaStamp-data"

# imagenet
# for status in ['train', 'val']:
#     path_temp = os.path.join(path, status)
#
#     for root, dirs, files in os.walk(path_temp):
#         for file in files:
#             if '.png' in file:
#                 im_path = os.path.join(root, file)
#                 file_name = file
#                 dot_index = file_name.find('.')
#                 name = file_name[0: dot_index]
#                 catagory, id, sta = name.split('_')
#                 target_path_temp = os.path.join(target_path, 'combine', status, catagory)
#                 target_path_temp_1 = os.path.join(target_path, sta, status, catagory)
#                 if not os.path.exists(target_path_temp):
#                     os.makedirs(target_path_temp)
#                 if not os.path.exists(target_path_temp_1):
#                     os.makedirs(target_path_temp_1)
#                 shutil.copy(im_path, target_path_temp)
#                 shutil.copy(im_path, target_path_temp_1)
#                 print(file)

# cifar
for status in ['train', 'val']:
    path_temp = os.path.join(path, status)

    for root, dirs, _ in os.walk(path_temp):
        for dir in dirs:
            for _, _, files in os.walk(os.path.join(root, dir)):
                for file in files:
                    if '.png' in file:
                        im_path = os.path.join(root, dir, file)
                        file_name = file
                        dot_index = file_name.find('.')
                        name = file_name[0: dot_index]
                        catagory = dir
                        id, sta = name.split('_')
                        target_path_temp = os.path.join(target_path, 'combine', status, catagory)
                        target_path_temp_1 = os.path.join(target_path, sta, status, catagory)
                        if not os.path.exists(target_path_temp):
                            os.makedirs(target_path_temp)
                        if not os.path.exists(target_path_temp_1):
                            os.makedirs(target_path_temp_1)
                        shutil.copy(im_path, target_path_temp)
                        shutil.copy(im_path, target_path_temp_1)
                        print(file)
import os, sys
from PIL import Image
import random

inject = 'a'
data_dir = "/home/renge/Pycharm_Projects/ISSBA/datasets/sub-imagenet-200/train"
out_path = "/home/renge/Pycharm_Projects/ISSBA/datasets/sub-imagenet-200-bd/inject_a/train"

if not os.path.exists(out_path):
    os.makedirs(out_path)
count = 0
im_path_list = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if '.JPEG' in file:
            im_path = os.path.join(root, file)
            im_path_list.append(im_path)
            # if not os.path.exists(os.path.dirname(im_path).replace(data_dir, '')):
            #     os.makedirs(os.path.dirname(im_path).replace(data_dir, ''))

random.shuffle(im_path_list)
for im_path in im_path_list:
    cmd = 'python /home/renge/Pycharm_Projects/ISSBA/encode_image.py ' \
          '--model_path /home/renge/Pycharm_Projects/ISSBA/ckpt/encoder_imagenet ' \
          f'--image_path {im_path} ' \
          f'--out_dir {out_path} ' \
          f'--secret {inject}'
    #print(cmd)
    os.system(cmd)
    count += 1
    if count >= 1 * len(im_path_list):
        break

# for name in inject_imgs_name:
#     im_path = os.path.join(origin_dir, name)
#     # f.write(name + ' 1\n')
#     cmd = 'python ../encode_image.py ' \
#             '../saved_models_cifar/1 ' \
#                 '--image {} ' \
#                     '--save_dir hidden/ ' \
#                         '--secret {}'.format(im_path, inject)

#     os.system(cmd)
# f.close()

# python encode_image.py \
#   saved_models/EXP_NAME \
#   --image test_im.png  \
#   --save_dir out/ \
#   --secret Hello

import os
import random
from shutil import copy
import cv2

tiny_dir = 'tiny-imagenet-200'
# out_dir = 'sub-imagenet-200'

tiny_classes = os.listdir('../' + tiny_dir + '/train/')

assert len(tiny_classes) == 200

for tc in tiny_classes:
    images = os.listdir('/media/lyz/DiskData1/common/ImageNet/ILSVRC2012_train_img/' + tc + '/')
    # Random select 500 as training, 50 as validation, 50 as testing
    random.shuffle(images)
    trainset = images[:500]
    valset = images[500:550]
    testset = images[550:600]

    os.makedirs('train/' + tc)
    os.makedirs('val/' + tc)
    os.makedirs('test/' + tc)

    for f in trainset:
        im = cv2.imread('/media/lyz/DiskData1/common/ImageNet/ILSVRC2012_train_img/' + tc + '/' + f)
        im = cv2.resize(im, (224, 224))
        cv2.imwrite('train/' + tc + '/' + f, im)
    for f in valset:
        im = cv2.imread('/media/lyz/DiskData1/common/ImageNet/ILSVRC2012_train_img/' + tc + '/' + f)
        im = cv2.resize(im, (224, 224))
        cv2.imwrite('val/' + tc + '/' + f, im)
    for f in testset:
        im = cv2.imread('/media/lyz/DiskData1/common/ImageNet/ILSVRC2012_train_img/' + tc + '/' + f)
        im = cv2.resize(im, (224, 224))
        cv2.imwrite('test/' + tc + '/' + f, im)
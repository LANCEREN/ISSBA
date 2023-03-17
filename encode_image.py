"""
The original code is from StegaStamp:
Invisible Hyperlinks in Physical Photographs,
Matthew Tancik, Ben Mildenhall, Ren Ng
University of California, Berkeley, CVPR2020
More details can be found here: https://github.com/tancik/StegaStamp
"""
import bchlib
import os
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import argparse

parser = argparse.ArgumentParser(description='Generate sample-specific triggers')
parser.add_argument('--model_path', type=str, default='/home/renge/Pycharm_Projects/ISSBA/ckpt/encoder_imagenet')
parser.add_argument('--image_path', type=str, default='data/imagenet/org/n01770393_12386.JPEG')
parser.add_argument('--out_dir', type=str,
                    default='/home/renge/Pycharm_Projects/ISSBA/datasets/image-bd/inject_a/')
parser.add_argument('--secret', type=str, default='a')
parser.add_argument('--secret_size', type=int, default=100)
args = parser.parse_args()

data_dir = "/home/renge/Pycharm_Projects/ISSBA/datasets/sub-imagenet-200/train"
data_dir = "/mnt/data03/renge/public_dataset/pytorch/imagenet-data/val"
dataset_type = 'CIFAR10'

model_path = args.model_path
image_path = args.image_path
out_dir = args.out_dir
secret = args.secret  # lenght of secret less than 7
secret_size = args.secret_size

sess = tf.InteractiveSession(graph=tf.Graph())

model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)

input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
    'stegastamp'].name
output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
    'residual'].name
output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

width = 224
height = 224
fit_size = (width, height)

BCH_POLYNOMIAL = 137
BCH_BITS = 5
bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

data = bytearray(secret + ' ' * (7 - len(secret)), 'utf-8')
ecc = bch.encode(data)
packet = data + ecc

packet_binary = ''.join(format(x, '08b') for x in packet)
secret = [int(x) for x in packet_binary]
secret.extend([0, 0, 0, 0])

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
count = 0

if dataset_type != 'CIFAR10':
    for root, dirs, _ in os.walk(data_dir):
        for dir in dirs:
            for _, _, files in os.walk(os.path.join(root, dir)):
                for file in files:
                    if '.JPEG' in file:
                        im_path = os.path.join(root, dir, file)

                        count = count + 1

                        image = Image.open(im_path)
                        image = np.array(ImageOps.fit(image, fit_size), dtype=np.float32) / 255.
                        if len(image.shape) != 3:
                            continue
                        elif image.shape[2] != 3:
                            continue
                        feed_dict = {
                            input_secret: [secret],
                            input_image: [image]
                        }
                        print(file)

                        hidden_img, residual = sess.run([output_stegastamp, output_residual], feed_dict=feed_dict)

                        hidden_img = (hidden_img[0] * 255).astype(np.uint8)
                        residual = residual[0] + .5  # For visualization
                        residual = (residual * 255).astype(np.uint8)

                        name = os.path.basename(im_path).split('.')[0]

                        out_dir_temp = os.path.join(out_dir, 'hidden', dir)
                        if not os.path.exists(out_dir_temp):
                            os.makedirs(out_dir_temp)
                        im = Image.fromarray(np.array(hidden_img))
                        im.save(out_dir_temp + '/' + name + '_hidden.png')

                        out_dir_temp = os.path.join(out_dir, 'residual', dir)
                        if not os.path.exists(out_dir_temp):
                            os.makedirs(out_dir_temp)
                        im = Image.fromarray(np.squeeze(residual))
                        im.save(out_dir_temp + '/' + name + '_residual.png')

elif dataset_type == 'CIFAR10':

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    class LockCIFAR100(datasets.CIFAR10):

        def __init__(self, root, train=True, transform=None, target_transform=None,
                     download=False):
            super(LockCIFAR100, self).__init__(root=root, train=train, transform=transform,
                                               target_transform=target_transform, download=download)

        def __getitem__(self, index):

            image = self.data[index]
            ground_truth_label = self.targets[index]
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            #image = Image.fromarray(image)

            # if self.transform is not None:
            #     image = self.transform(image)

            # if self.target_transform is not None:
            #     ground_truth_label = self.target_transform(ground_truth_label)

            return image, ground_truth_label


    def get_cifar100(
            train=True, val=True, **kwargs):
        data_root = os.path.expanduser(os.path.join('/mnt/data03/renge/public_dataset/image', 'cifar10-data'))

        ds = []
        if train:
            train_dataset = LockCIFAR100(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=1, shuffle=False, pin_memory=True,
                num_workers=4,)
            ds.append(train_loader)

        if val:
            test_dataset = LockCIFAR100(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset, batch_size=1, shuffle=False, pin_memory=True,
                num_workers=4,)
            ds.append(test_loader)
        ds = ds[0] if len(ds) == 1 else ds
        return ds


    ds = get_cifar100()
    for index, j in enumerate(ds):
        status = 'train' if index == 0 else 'val'
        for i,(data, target) in enumerate(j):
            out_dir_temp = os.path.join(out_dir, status)
            count = i + 1
            image = data[0]
            image = image.numpy()#.astype(dtype=np.float32)
            image = Image.fromarray(image)
            image = np.array(ImageOps.fit(image, fit_size), dtype=np.float32) / 255.
            if len(image.shape) != 3:
                continue
            elif image.shape[2] != 3:
                continue
            feed_dict = {
                input_secret: [secret],
                input_image: [image]
            }
            print(count)

            hidden_img, residual = sess.run([output_stegastamp, output_residual], feed_dict=feed_dict)

            hidden_img = (hidden_img[0] * 255).astype(np.uint8)
            image = (image * 255).astype(np.uint8)
            residual = residual[0] + .5  # For visualization
            residual = (residual * 255).astype(np.uint8)

            out_dir_temp = os.path.join(out_dir_temp, f"{int(target)}")
            if not os.path.exists(out_dir_temp):
                os.makedirs(out_dir_temp)
            im = Image.fromarray(np.array(image))
            im.save(out_dir_temp + '/' + f"{count}" + '.png')
            # im = Image.fromarray(np.array(hidden_img))
            # im.save(out_dir_temp + '/' + f"{count}" + '_hidden.png')
            # im = Image.fromarray(np.squeeze(residual))
            # im.save(out_dir_temp + '/' + f"{count}" + '_residual.png')
            print(count)
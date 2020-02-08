import math
import os
import random

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset

class SkeletonTrainDataset(Dataset):
    #TODO whether we should use heatmaps? or directly turn to skeleton image
    # if use heatmaps, i propose to use .npy format to save the pretrained data
    def __init__(self, dataset_folder, stride, sigma, transform=None):
        super().__init__()
        self._dataset_folder = dataset_folder
        # self._stride = stride
        # self._sigma = sigma
        self._transform = transform
        self.train_labels = [line.rstrip('\n') for line in
                        open(os.path.join(self._dataset_folder, 'Train', 'index.csv'), 'r')] #save the path of image/npy pair + conditional image

    def __getitem__(self, idx):
        tokens = self.train_labels[idx].split(',') #suggest format [input_real_npy_path, input_condition_path, output_anime_npy_path]
        real_skeletion = np.load(os.path.join(self._dataset_folder, 'Train', 'inputs', tokens[0]))
        condition_img = cv2.imread(os.path.join(self._dataset_folder, 'Train', 'conditions', tokens[1]), cv2.IMREAD_COLOR)
        anime_skeletion = np.load(os.path.join(self._dataset_folder, 'Train', 'gt', tokens[2]))
        #TODO maybe resize the inputs
        sample = {
            'real': real_skeletion,
            'condition': condition_img,
            'anime':anime_skeletion
        }
        if self._transform:
            sample = self._transform(sample) #TODO NOT SURE

        # should be pay attention that pretrain model ResNet's input is 224*224
        # transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor()]
        # )
        # normization
        image = sample['condition'].astype(np.float32)
        image = (image - 128) / 256 #turn to -0.5~0.5
        sample['condition'] = image.transpose((2, 0, 1)) #C,W,H
        return sample

    def __len__(self):
        return len(self.train_labels)

#TODO use dataset.imagefoloder rewrite

class SkeletonValDataset(Dataset):
    def __init__(self, dataset_folder, num_images=-1):
        # input should be heatmaps or skeleton image, output should be also heatmaps or skeleton image
        super().__init__()
        self._dataset_folder = dataset_folder
        self.index_path = os.path.join(self._dataset_folder, 'Val', 'index.csv')
        self.val_labels = [line.rstrip('\n') for line in open(self.index_path, 'r')]
        if num_images > 0:
            self.val_labels = self.val_labels[:num_images]

    def __getitem__(self, idx):
        tokens = self.val_labels[idx].split(',')  # suggest format [input_real_npy_path, input_condition_path, output_anime_npy_path]
        real_skeletion = np.load(os.path.join(self._dataset_folder, 'Val', 'inputs', tokens[0]))
        condition_img = cv2.imread(os.path.join(self._dataset_folder, 'Val', 'conditions', tokens[1]),
                                   cv2.IMREAD_COLOR)
        anime_skeletion = np.load(os.path.join(self._dataset_folder, 'Val', 'gt', tokens[2]))
        # TODO maybe resize the inputs
        sample = {
            'real': real_skeletion,
            'condition': condition_img,
            'anime': anime_skeletion
        }
        # normization
        # image = sample['condition'].astype(np.float32)
        # image = (image - 128) / 256  # turn to -0.5~0.5
        # sample['condition'] = image.transpose((2, 0, 1))  # C,W,H
        return sample

    def __len__(self):
        return len(self.val_labels)
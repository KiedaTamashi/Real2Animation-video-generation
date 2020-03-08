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
        real_skeletion = cv2.imread(os.path.join(self._dataset_folder, 'Train', 'OriPose', tokens[0]),cv2.IMREAD_GRAYSCALE)
        condition_img = cv2.imread(os.path.join(self._dataset_folder, 'MapFrame', tokens[2]), cv2.IMREAD_COLOR)
        anime_skeletion = cv2.imread(os.path.join(self._dataset_folder, 'Train', 'MapPose', tokens[1]),cv2.IMREAD_GRAYSCALE)
        #TODO maybe resize the inputs
        if self._transform:
            real_skeletion = self._transform(real_skeletion)
            condition_img = self._transform(condition_img)
            anime_skeletion = self._transform(anime_skeletion)
        sample = {
            'real': real_skeletion,
            'condition': condition_img,
            'anime':anime_skeletion
        }


        # should be pay attention that pretrain model ResNet's input is 224*224
        # transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor()]
        # )
        # normization
        # TODO not sure whether I should process the input and gt
        # image = sample['condition']
        # sample['condition'] = image.transpose((2, 0, 1)) #C,W,H
        return sample

    def __len__(self):
        return len(self.train_labels)

#TODO use dataset.imagefoloder rewrite

class SkeletonValDataset(Dataset):
    def __init__(self, dataset_folder, transform=None,num_images=-1):
        # input should be heatmaps or skeleton image, output should be also heatmaps or skeleton image
        super().__init__()
        self._transform = transform
        self._dataset_folder = dataset_folder
        self.index_path = os.path.join(self._dataset_folder, 'Val', 'index.csv')
        self.val_labels = [line.rstrip('\n') for line in open(self.index_path, 'r')]
        if num_images > 0:
            self.val_labels = self.val_labels[:num_images]

    def __getitem__(self, idx):
        tokens = self.val_labels[idx].split(',')  # suggest format [input_real_npy_path, input_condition_path, output_anime_npy_path]
        real_skeletion = cv2.imread(os.path.join(self._dataset_folder, 'Val', 'OriPose', tokens[0]), cv2.IMREAD_GRAYSCALE)
        condition_img = cv2.imread(os.path.join(self._dataset_folder, 'MapFrame', tokens[2]), cv2.IMREAD_COLOR)
        anime_skeletion = cv2.imread(os.path.join(self._dataset_folder, 'Val', 'MapPose', tokens[1]), cv2.IMREAD_GRAYSCALE)
        # resize the inputs now TODO not sure whether it is correct
        if self._transform:
            real_skeletion = self._transform(real_skeletion)
            condition_img = self._transform(condition_img)
            anime_skeletion = self._transform(anime_skeletion)
        sample = {
            'real': real_skeletion,
            'condition': condition_img,
            'anime': anime_skeletion
        }
        # normization
        # image = sample['condition'].astype(np.float32)
        # image = (image - 128) / 256  # turn to -0.5~0.5
        # sample['condition'] = image.transpose((2, 0, 1))  # C,W,H
        # image = sample['condition']
        # sample['condition'] = image.transpose(0,2).transpose(1,2)  # C,W,H
        return sample

    def __len__(self):
        return len(self.val_labels)

class SkeletonTestDataset(Dataset):
    def __init__(self, dataset_folder, transform=None):
        # input should be heatmaps or skeleton image, output should be also heatmaps or skeleton image
        super().__init__()
        self._transform = transform
        self._dataset_folder = dataset_folder
        self._names = os.listdir(os.path.join(self._dataset_folder, 'Test','Input'))
        # TODO now for test, we only have 1 condition image endwith '.jpg'
        self._condition = ['condition.jpg'] * len(self._names)


    def __getitem__(self, id):
        x_name = self._names[id]
        c_name = self._condition[id]

        real_skeletion = cv2.imread(os.path.join(self._dataset_folder, x_name), cv2.IMREAD_GRAYSCALE)
        condition_img = cv2.imread(os.path.join(self._dataset_folder, c_name), cv2.IMREAD_COLOR)
        if self._transform:
            real_skeletion = self._transform(real_skeletion)
            condition_img = self._transform(condition_img)

        # TODO maybe resize the inputs
        sample = {
            'name': x_name,
            'real': real_skeletion,
            'condition': condition_img,
        }
        return sample

    def __len__(self):
        return len(self.self._names)

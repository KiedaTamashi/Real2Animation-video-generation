import numpy as np
import pandas as pd
import json
import os
import cv2
MISSING_VALUE = -1
def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def cords_to_map(cords, img_size, sigma=6):
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point[1]) ** 2 + (xx - point[0]) ** 2) / (2 * sigma ** 2))  # the location of kps is lighted up and decrease as distribution
    return result

def compute_pose(ori_dir, savePath,image_size=(256, 192)):
    ori_kps = os.listdir(ori_dir)
    cnt = len(ori_kps)
    for i in range(cnt):
        print('processing %d / %d ...' %(i, cnt))
        kp_array = np.load(os.path.join(ori_dir,ori_kps[i]))
        name = ori_kps[i]
        print(savePath, name)
        file_name = os.path.join(savePath, name)
        pose = cords_to_map(kp_array, image_size)
        np.save(file_name, pose)

def compute_pose_single(file_name,savePath,image_size=(256, 192)):
    kp_array = np.load(file_name)
    name = os.path.basename(file_name)
    save_name = os.path.join(savePath, name)
    pose = cords_to_map(kp_array, image_size)
    np.save(save_name, pose)

if __name__ == '__main__':
    # img_dir = r'../anime_data/train' #raw image path
    img_size = (256, 192)
    ori_dir = r'D:/download_cache/anime_data/normK_s'  # pose annotation path
    save_path = r'D:/download_cache/anime_data/trainK'  # path to store pose maps
    compute_pose(ori_dir, save_path,img_size)
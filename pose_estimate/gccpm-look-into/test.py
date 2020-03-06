import argparse
import cv2
import numpy as np
import sys
import time,os

import torch

from datasets.lip import LipTestDataset
from datasets.anime import AnimeTestDataset
from models.single_person_pose_with_mobilenet import SinglePersonPoseEstimationWithMobileNet
from modules.calc_pckh import calc_pckh
from modules.load_state import load_state


def extract_keypoints(heatmap, min_confidence=-100):
    ind = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    if heatmap[ind] < min_confidence:
        ind = (-1, -1)
    else:
        ind = (int(ind[1]), int(ind[0]))
    return ind


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def get_single_feature(features,idx):
    #B,C,H,W
    feature = features[idx, :, :]

    # feature = feature.view(feature.shape[1], feature.shape[2])
    # print(feature.shape)
    return feature

def save_feature_to_img(features,result_folder):
    # to numpy
    cnt=0
    for item in features:
        cnt+=1
        item = item.squeeze().cpu()
        folder_n = os.path.join(result_folder, f'refine_stage{cnt}')
        os.mkdir(folder_n)
        for idx in range(item.shape[0]):
            feature = get_single_feature(item,idx)
            feature = feature.data.numpy()
            # use sigmod to [0,1]
            feature = 1.0 / (1 + np.exp(-1 * feature))
            # to [0,255]
            feature = np.round(feature * 255)
            print(feature[0])
            cv2.imwrite(os.path.join(folder_n,f'{cnt}_feature_inside_{idx}.jpg'), feature)


def infer(net, img, scales, base_height, stride, img_mean=[128, 128, 128], img_scale=1/256, num_kps=16,visualize_feature=False):
    height, width, _ = img.shape
    scales_ratios = [scale * base_height / max(height, width) for scale in scales]
    avg_heatmaps = np.zeros((height, width, num_kps+1), dtype=np.float32)

    for ratio in scales_ratios:
        resized_img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        max_side = max(resized_img.shape[0], resized_img.shape[1])

        padded_img = np.ones((max_side, max_side, 3), dtype=np.uint8) * img_mean
        x_offset = (padded_img.shape[1] - resized_img.shape[1]) // 2
        y_offset = (padded_img.shape[0] - resized_img.shape[0]) // 2
        padded_img[y_offset:y_offset + resized_img.shape[0], x_offset:x_offset + resized_img.shape[1], :] = resized_img
        padded_img = normalize(padded_img, img_mean, img_scale)
        pad = [y_offset, x_offset,
               padded_img.shape[0] - resized_img.shape[0] - y_offset,
               padded_img.shape[1] - resized_img.shape[1] - x_offset]

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
        if visualize_feature:
            stages_output, mid_feature = net(tensor_img,visualize_feature)
        else:
            stages_output = net(tensor_img)
        # output is all B,C,H,W, here H,W is 32x32, backbone_feature is 128,32,32
        heatmaps = np.transpose(stages_output[-1].squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :]
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_heatmaps = avg_heatmaps + heatmaps / len(scales_ratios)

    if visualize_feature:
        return avg_heatmaps,mid_feature
    else:
        return avg_heatmaps


def evaluate(dataset, results_folder, net, multiscale=False, visualize=False, save_maps=False,
             num_kps=16,get_feature=False,dataset_mode=False):
    net = net.cuda().eval()
    base_height = 256
    scales = [1]
    if multiscale:
        scales = [0.75, 1.0, 1.25]
    stride = 8
    output_name = os.path.join(results_folder,"kps_results.csv")
    res_file = open(output_name, 'w')
    if visualize:
        result_imgs_dir = os.path.join(results_folder,"outputs")
        if not os.path.exists(result_imgs_dir):
            os.makedirs(result_imgs_dir)

    if save_maps:
        result_heatmap_dir = os.path.join(results_folder,"heatmaps")
        if not os.path.exists(result_heatmap_dir):
            os.makedirs(result_heatmap_dir)

    if dataset_mode:
        pose_dir = os.path.join(results_folder, "pose_dataset")
        if not os.path.exists(pose_dir):
            os.mkdir(pose_dir)

    for sample_id in range(len(dataset)):
        sample = dataset[sample_id]
        file_name = sample['file_name']
        img = sample['image']
        if get_feature:
            avg_heatmaps,mid_feature = infer(net, img, scales, base_height, stride, num_kps=num_kps, visualize_feature=get_feature)
            save_feature_to_img(mid_feature, results_folder)
        else:
            avg_heatmaps = infer(net, img, scales, base_height, stride, num_kps=num_kps,visualize_feature=get_feature)


        flip = False
        if flip:
            flipped_img = cv2.flip(img, 1)
            flipped_avg_heatmaps = infer(net, flipped_img, scales, base_height, stride,num_kps=num_kps)
            orig_order = [0, 1, 2, 10, 11, 12]
            flip_order = [5, 4, 3, 15, 14, 13]
            for r, l in zip(orig_order, flip_order):
                flipped_avg_heatmaps[:, :, r], flipped_avg_heatmaps[:, :, l] =\
                    flipped_avg_heatmaps[:, :, l].copy(), flipped_avg_heatmaps[:, :, r].copy()
            avg_heatmaps = (avg_heatmaps + flipped_avg_heatmaps[:, ::-1]) / 2

        all_keypoints = []
        for kpt_idx in range(num_kps):
            all_keypoints.append(extract_keypoints(avg_heatmaps[:, :, kpt_idx]))

        if not dataset_mode:
            res_file.write('{}'.format(file_name))
            for id in range(num_kps):
                val = [int(all_keypoints[id][0]), int(all_keypoints[id][1])]
                if val[0] == -1:
                    val[0], val[1] = 'nan', 'nan'
                res_file.write(',{},{}'.format(val[0], val[1]))
            res_file.write('\n')

        if dataset_mode:
            h,w,_ = img.shape
            radius = 10
            pose_img = np.zeros((h, w), np.uint8)
            for id in range(len(all_keypoints)):
                keypoint = all_keypoints[id]
                if keypoint[0] != -1:
                    # if colors[id] == (255, 0, 0):
                    #     cv2.circle(img, (int(keypoint[0]), int(keypoint[1])),
                    #                radius + 2, (255, 0, 0), -1)
                    # else:
                    cv2.circle(pose_img, (int(keypoint[0]), int(keypoint[1])),
                               radius, (255,255,255), -1)
            img_name = os.path.join(pose_dir, file_name)
            cv2.imwrite(img_name, pose_img)

        if visualize:
            # kpt_names = ['r_ank', 'r_kne', 'r_hip', 'l_hip', 'l_kne', 'l_ank', 'pel', 'spi', 'nec', 'hea',
            #              'r_wri', 'r_elb', 'r_sho', 'l_sho', 'l_elb', 'l_wri']
            # colors = [(255, 0, 0), (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 255, 0),
            #             #           (0, 255, 0), (0, 255, 0), (0, 255, 0),
            #             #           (255, 0, 0), (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255), (0, 0, 255)]
            if num_kps==21:
                colors = [(0,0,0)]*5+[(255,0,0)]*4+[(0,255,0)]*4+[(0,0,255)]*4+[(0,255,255)]*4
            else:
                colors = [(255,0,0)]*num_kps
            for id in range(len(all_keypoints)):
                keypoint = all_keypoints[id]
                if keypoint[0] != -1:
                    radius = 4
                    # if colors[id] == (255, 0, 0):
                    #     cv2.circle(img, (int(keypoint[0]), int(keypoint[1])),
                    #                radius + 2, (255, 0, 0), -1)
                    # else:
                    cv2.circle(img, (int(keypoint[0]), int(keypoint[1])),
                               radius, colors[id], -1)
            img_name = os.path.join(result_imgs_dir,file_name)
            cv2.imwrite(img_name,img)

        if save_maps:
            np_heatmaps = np.array(avg_heatmaps)
            np_name = os.path.join(result_heatmap_dir,file_name[:-4]+".npy")
            np.save(np_name,np_heatmaps)

    res_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='test',
                        help='name of output file with detected keypoints')
    parser.add_argument('--multiscale', action='store_true', help='average inference results over multiple scales')
    parser.add_argument('--visualize', type=bool, default=False, help='show keypoints')
    parser.add_argument('--get_feature', type=bool, default=False, help='--get_feature')
    parser.add_argument('--dataset_mode', type=bool, default=True, help='generate kps maps dataset for VAE')
    parser.add_argument('--save_maps', action='store_true', help='show keypoints')
    parser.add_argument('--checkpoint-path', type=str, default="checkpoints/checkpoint_anime_47.pth", help='path to the checkpoint')
    parser.add_argument('--dataset_folder', type=str, default="./data_anime", help='path to dataset folder')
    parser.add_argument('--num_kps', type=int, default=21,  # need change 16 for real, 21 for anime
                        help='number of key points')

    # parser.add_argument('--checkpoint-path', type=str, default="checkpoints/checkpoint_real.pth", help='path to the checkpoint')
    # parser.add_argument('--dataset_folder', type=str, default="./data_lip", help='path to dataset folder')
    # parser.add_argument('--num_kps', type=int, default=16,  # need change 16 for real 21 for anime
    #                     help='number of key points')
    args = parser.parse_args()


    net = SinglePersonPoseEstimationWithMobileNet(num_refinement_stages=5,num_heatmaps=args.num_kps+1)
    checkpoint = torch.load(args.checkpoint_path)
    load_state(net, checkpoint)
    data_flag = "real" if args.dataset_folder.split("/")[-1] == "data_lip" else "anime"

    date = time.strftime("%m%d-%H%M%S")
    results_folder = 'test_results/{}{}_test'.format(args.experiment_name, date)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    ori_dataFolder = "D:\download_cache\VAEmodel\OriFrame"
    map_dataFolder = "D:\download_cache\VAEmodel\MapFrame"
    if data_flag=="real":
        dataset = LipTestDataset(ori_dataFolder)
    else:
        dataset = AnimeTestDataset(map_dataFolder) #TODO I have modified the datasets

    # TODO we need shadow like image.
    evaluate(dataset, results_folder, net, args.multiscale, args.visualize,args.save_maps,num_kps=args.num_kps,
             get_feature=args.get_feature,dataset_mode=args.dataset_mode)

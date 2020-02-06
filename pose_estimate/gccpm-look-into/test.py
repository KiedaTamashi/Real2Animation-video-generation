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


def infer(net, img, scales, base_height, stride, img_mean=[128, 128, 128], img_scale=1/256, num_kps=16):
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
        stages_output = net(tensor_img)

        heatmaps = np.transpose(stages_output[-1].squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :]
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_heatmaps = avg_heatmaps + heatmaps / len(scales_ratios)

    return avg_heatmaps


def evaluate(dataset, results_folder, net, multiscale=False, visualize=False, save_maps=False, num_kps=16):
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

    for sample_id in range(len(dataset)):
        sample = dataset[sample_id]
        file_name = sample['file_name']
        img = sample['image']

        avg_heatmaps = infer(net, img, scales, base_height, stride, num_kps=num_kps)

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

        res_file.write('{}'.format(file_name))
        for id in range(num_kps):
            val = [int(all_keypoints[id][0]), int(all_keypoints[id][1])]
            if val[0] == -1:
                val[0], val[1] = 'nan', 'nan'
            res_file.write(',{},{}'.format(val[0], val[1]))
        res_file.write('\n')

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
    parser.add_argument('--dataset_folder', type=str, default="./data_anime", help='path to dataset folder')
    parser.add_argument('--experiment_name', type=str, default='test',
                        help='name of output file with detected keypoints')
    parser.add_argument('--checkpoint-path', type=str, default="checkpoints/checkpoint_anime_newdata.pth", help='path to the checkpoint')
    parser.add_argument('--multiscale', action='store_true', help='average inference results over multiple scales')
    parser.add_argument('--visualize', type=bool, default=True, help='show keypoints')
    parser.add_argument('--save_maps', action='store_true', help='show keypoints')
    parser.add_argument('--num_kps', type=int, default=21,  # need change 16 for real 21 for anime
                        help='number of key points')
    args = parser.parse_args()


    net = SinglePersonPoseEstimationWithMobileNet(num_refinement_stages=5,num_heatmaps=args.num_kps+1)
    checkpoint = torch.load(args.checkpoint_path)
    load_state(net, checkpoint)
    data_flag = "real" if args.dataset_folder.split("/")[-1] == "data_lip" else "anime"

    date = time.strftime("%m%d-%H%M%S")
    results_folder = 'test_results/{}{}_test'.format(args.experiment_name, date)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    if data_flag=="real":
        dataset = LipTestDataset(args.dataset_folder)
    else:
        dataset = AnimeTestDataset(args.dataset_folder)
    evaluate(dataset, results_folder, net, args.multiscale, args.visualize,args.save_maps,num_kps=args.num_kps)

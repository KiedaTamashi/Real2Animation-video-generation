import numpy as np
import random
import pandas as pd
import os,time,shutil
import cv2
from data_prepare.smooth_pose import smooth_json_pose
from data_prepare.video2vmd import video2keypoints
from pose_estimate.process_imgs import load_json
import scipy.io as sio

def get_clip_kps(filedir = r"D:\download_cache\PMXmodel\VIDEOclips"):
    files = os.listdir(filedir)
    json_out_dir = r"D:\download_cache\PMXmodel\tmp_kps"
    kps_out_dir = r"D:\download_cache\PMXmodel\VIDEOkps"


    for file in files:
        print(file)
        video_path = os.path.join(filedir,file)

        if os.path.exists(json_out_dir):
            shutil.rmtree(json_out_dir)
        time.sleep(3)
        os.mkdir(json_out_dir)

        video2keypoints(video_path, json_out_dir)

        #smooth the kps. Using window size ~= 1/2*frame_rate
        # cap = cv2.VideoCapture(video_path)
        # fps = cap.get(cv2.CAP_PROP_FPS) #fps now is 30
        wid_size = 13
        # wid_size = int(fps/2)
        # if wid_size>21:
        #     wid_size=21
        # else:
        #     wid_size = wid_size-1 if (wid_size & 1) == 0 else wid_size
        smooth_json_pose(json_out_dir,window_length=wid_size,polyorder=3,threshold=0.15)
        frames = os.listdir(json_out_dir)
        frames.sort(key=lambda x:int(x.split("_")[-2]))
        video_kps = []
        for idx,frame in enumerate(frames):
            pose = load_json(os.path.join(json_out_dir,frame))['people'][0]['pose_keypoints_2d']
            kps = []
            for idx in range(0, len(pose)-6, 3):  # throw away 16,17, which is for ear
                x, y, _ = pose[idx], pose[idx + 1], pose[idx + 2]
                kps.append([x,y])
            video_kps.append(kps)
        np.save(os.path.join(kps_out_dir,file[:-4]+".npy"),video_kps)


def render_joints(cvmat, joints):
    for _joint in joints:
        _x, _y = _joint
        cv2.circle(cvmat, center=(int(_x), int(_y)), color=(255, 0, 0), radius=7, thickness=2)
    return cvmat

def generate_dataset_posewarper(kps_dir=r"D:\download_cache\PMXmodel\VIDEOkps",video_dir=r"D:\download_cache\PMXmodel\OUTPUTclips_done",
                     out_dir = r"D:\download_cache\animeWarp",interval=5):
    train_frame = os.path.join(out_dir,"train","frames")
    train_info = os.path.join(out_dir,"train","info")
    test_frame = os.path.join(out_dir, "test", "frames")
    test_info = os.path.join(out_dir, "test", "info")

    interval_new = random.randint(interval - 2, interval + 2)

    videos = os.listdir(video_dir)
    random.shuffle(videos)
    num_video = len(videos)
    ban_list = ["Eren.avi", "GTGoku.avi", "RoseFreyja.avi", "ifuleet.avi", "inkling.avi", 'luigi.avi', "Nikos.avi"]
    for idx,video in enumerate(videos):
        flag=False
        for ban in ban_list:
            if video.endswith(ban):
                flag=True
                break
        if flag:
            continue

        video_name = video[:-4]
        kps_name = "_".join(video.split("_")[0:3]) + ".npy"
        video_cap = cv2.VideoCapture(os.path.join(video_dir, video))
        if idx < int(num_video*0.9):
            save_dir = os.path.join(train_frame,video_name)
            os.mkdir(os.path.join(train_frame,video_name))
        else:
            save_dir = os.path.join(test_frame,video_name)
            os.mkdir(os.path.join(test_frame,video_name))

        frame_number = 0
        tick = 0
        cnt=0
        kps = []
        kps_ori = np.load(os.path.join(kps_dir, kps_name))

        while True:
            ret1, img1 = video_cap.read()
            if not ret1: break
            if tick == interval_new:
                cnt+=1
                print(video_name + str(frame_number))
                cv2.imwrite(os.path.join(save_dir, str(cnt) + '.jpg'), img1)
                kps.append(kps_ori[frame_number])
                random.seed(time.time())
                interval_new = random.randint(interval - 2, interval + 2)
                tick = 0
            else:
                tick += 1
            frame_number += 1
        video_cap.release()

        kps = np.array(kps)
        kps = np.swapaxes(np.swapaxes(kps, 0, 1), 1, 2)
        box = np.swapaxes(kps[1], 0, 1)
        box = np.column_stack((box, np.zeros((box.shape[0], 2))))
        pose_dtype = np.dtype([('X', 'O'), ('bbox', 'O')])
        pose_sample = np.array(([kps], [box]), dtype=pose_dtype)


        if idx < int(num_video*0.9):
            sio.savemat(os.path.join(train_info, video_name + ".mat"), {'data': pose_sample})  # TODO idk why there is a list out
        else:
            sio.savemat(os.path.join(test_info, video_name + ".mat"), {'data': pose_sample})

def gen_dataset_posetransfer(kps_dir=r"D:\download_cache\PMXmodel\VIDEOkps",video_dir=r"D:\download_cache\PMXmodel\OUTPUTclips_done",
                     out_dir = r"D:\download_cache\anime_data",interval=4):
    #  make pair with a video length n : n(n-1) column is from .. to ..
    #  extract kps to npy file. should directly link to real human kps (which is normalized
    train_frame = os.path.join(out_dir, "train")
    train_info = os.path.join(out_dir,"trainK")
    test_frame = os.path.join(out_dir, "test")
    test_info = os.path.join(out_dir, "testK")

    interval_new = random.randint(interval - 2, interval + 2)

    videos = os.listdir(video_dir)
    random.shuffle(videos)
    num_video = len(videos)
    ban_list = ["Eren.avi", "GTGoku.avi", "RoseFreyja.avi", "ifuleet.avi", "inkling.avi", 'luigi.avi', "Nikos.avi"]
    for idx, video in enumerate(videos):
        flag = False
        for ban in ban_list:
            if video.endswith(ban):
                flag = True
                break
        if flag:
            continue

        video_name = video[:-4]
        kps_name = "_".join(video.split("_")[0:3]) + ".npy"
        video_cap = cv2.VideoCapture(os.path.join(video_dir, video))
        if idx < int(num_video*0.9):
            save_dir = train_frame
            save_kps = train_info
        else:
            save_dir = test_frame
            save_kps = test_info
        frame_number = 0
        tick = 0
        cnt = 0
        kps_ori = np.load(os.path.join(kps_dir, kps_name))

        while True:
            ret1, img1 = video_cap.read()
            if not ret1: break
            if tick == interval_new:
                print(video_name + str(frame_number))
                cv2.imwrite(os.path.join(save_dir, video_name + "_" + str(cnt) + '.jpg'), img1)
                cnt += 1
                np.save(os.path.join(save_kps, video_name + "_" + str(cnt) + '.jpg' + '.npy'), np.array(kps_ori[frame_number]))
                random.seed(time.time())
                interval_new = random.randint(interval - 2, interval + 2)
                tick = 0
            else:
                tick += 1
            frame_number += 1
        video_cap.release()


def kps_Normalize(kps_r=r"D:\download_cache\anime_data\trainK", kps_a=r"D:\download_cache\anime_data\tmpK",
                  output_dir = "D:\download_cache\anime_data\normK"):
    img_nums = len(kps_r)
    kpss = os.listdir(kps_r)
    width = 1920
    height = 1080
    for idx in range(img_nums):
        real_kps = np.load(os.path.join(kps_r,kpss[idx]))
        anime_kps = np.load(os.path.join(kps_a,kpss[idx]))
        neck_a = anime_kps[1,:]
        lhip_a = anime_kps[13,:]
        rhip_a = anime_kps[17,:]
        neck_r = real_kps[1,:]
        lhip_r = real_kps[11,:]
        rhip_r = real_kps[8,:]
        hip_a = (lhip_a[1]+rhip_a[1]) / 2.0
        hip_r = (lhip_r[1]+rhip_r[1]) / 2.0
        body_a = abs(neck_a[1] - hip_a[1])
        body_r = abs(neck_r[1] - hip_r[1])
        scale = float(body_a/body_r)
        center_and_scale_image(real_kps)





def center_and_scale_image(I, img_width, img_height, pos, scale, joints):
    I = cv2.resize(I, (0, 0), fx=scale, fy=scale)
    joints = joints * scale

    x_offset = (img_width - 1.0) / 2.0 - pos[0] * scale
    y_offset = (img_height - 1.0) / 2.0 - pos[1] * scale
    T = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    I = cv2.warpAffine(I, T, (img_width, img_height))

    joints[:, 0] += x_offset
    joints[:, 1] += y_offset

    return I, joints



if __name__ == '__main__':
    st = time.time()
    gen_dataset_posetransfer()
    print(time.time()-st)

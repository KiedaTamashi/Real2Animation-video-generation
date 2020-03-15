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

def generate_dataset(kps_dir=r"D:\download_cache\PMXmodel\VIDEOkps",video_dir=r"D:\download_cache\PMXmodel\OUTPUTclips_done",
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





if __name__ == '__main__':
    st = time.time()
    generate_dataset()
    print(time.time()-st)

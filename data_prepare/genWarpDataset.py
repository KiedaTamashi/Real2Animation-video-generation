import numpy as np
import random,math
import pandas as pd
import os,time,shutil
import cv2
from data_prepare.smooth_pose import smooth_json_pose
from data_prepare.video2vmd import video2keypoints
from pose_estimate.process_imgs import load_json
import scipy.io as sio
from copy import deepcopy

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

    interval_new = random.randint(interval - 1, interval + 1)

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
                np.save(os.path.join(save_kps, video_name + "_" + str(cnt) + '.jpg' + '.npy'), np.array(kps_ori[frame_number]))
                random.seed(time.time())
                interval_new = random.randint(interval - 2, interval + 2)
                cnt += 1
                tick = 0
            else:
                tick += 1
            frame_number += 1
        video_cap.release()

def kps_Normalize_dir(kps_r=r"D:\download_cache\anime_data\trainK", kps_a=r"D:\download_cache\anime_data\tmpK",
                  output_dir = "D:/download_cache/anime_data/normK/"):
    img_nums = len(kps_r)
    kpss = os.listdir(kps_r)
    for idx in range(img_nums):
        real_kps = np.load(os.path.join(kps_r, kpss[idx]))
        anime_kps = np.load(os.path.join(kps_a, kpss[idx]))
        modified_kps = kps_Normalize(real_kps,anime_kps,scale_level=0.8)

        np.save(output_dir + kpss[idx], modified_kps)

def kps_Normalize(real_kps,anime_kps,scale_level=0.8):
    # real and anime kps should be (n,2) numpy array
    neck_a = anime_kps[1,:]
    lhip_a = anime_kps[13,:]
    rhip_a = anime_kps[17,:]
    head_a = anime_kps[2,:]
    neck_r = real_kps[1,:]
    lhip_r = real_kps[11,:]
    rhip_r = real_kps[8,:]
    head_r = real_kps[0,:]

    hip_a = (lhip_a[1]+rhip_a[1]) / 2.0
    hip_r = (lhip_r[1]+rhip_r[1]) / 2.0
    body_a = abs(neck_a[1] - hip_a)
    body_r = abs(neck_r[1] - hip_r)
    scale = float(body_a/body_r)
    offset = neck_a-neck_r
    center = neck_a

    theta_a = math.atan((head_a[1]-neck_a[1])/(head_a[0]-neck_a[0]))
    theta_r = math.atan((head_r[1] - neck_r[1]) / (head_r[0] - neck_r[0]))
    theta = theta_a - theta_r
    if abs(theta)>1.7:
        T = None
    else:
        cos = math.cos(theta)
        sin = math.sin(theta)
        T = [cos,sin]

    modified_kps = center_and_scale_joints(scale,offset,deepcopy(real_kps),deepcopy(center),scale_level=scale_level, trans=T)


    return modified_kps

# 134 -> 138, 27 -> 31, 24->27/28, 9->13
def around_modified(filename):
    frame_num = filename.split("_")[-1]
    dance_name = "_".join(filename.split("_")[:3])
    pose_all = np.load("D:/download_cache/PMXmodel/VIDEOkps\\"+dance_name+".npy")
    ranges = [int(frame_num)*5,int(frame_num)*6]
    index_name = filename + '.jpg.npy'
    img_name = "D:\\download_cache\\anime_data\\train\\" + filename+'.jpg'
    kps_r = r"D:\download_cache\anime_data\trainK"
    kps_a = r"D:\download_cache\anime_data\tmpK"
    cnt = ranges[0]
    anime_kps = np.load(os.path.join(kps_a,index_name))
    real_kps = np.load(os.path.join(kps_r,index_name))
    # for item in pose_all[ranges[0]:ranges[1], :, :]:
    #     img = cv2.imread(img_name)
    #     item_ = kps_Normalize(deepcopy(item), anime_kps)
    #     for _joint in item_:
    #         _x, _y = _joint
    #         cv2.circle(img, center=(int(_x), int(_y)), color=(255, 0, 0), radius=7, thickness=2)
    #
    #     cv2.imwrite("D:\\download_cache\\anime_data" + f"/neigh{cnt}.jpg", img)
    #     cnt += 1

    plot_points(img_name, real_kps, "/ori.jpg")
    item_ = kps_Normalize(deepcopy(real_kps), anime_kps)
    plot_points(img_name,item_,"/final.jpg")
    plot_points(img_name,anime_kps,"/anime_reference.jpg")
    item_ = align(deepcopy(real_kps), anime_kps)
    plot_points(img_name,item_,"/aligned.jpg")

    # dance_61_3_madoka2019_5.jpg

def plot_points(img_name,item_,outname):
    img = cv2.imread(img_name)
    for _joint in item_:
        _x, _y = _joint
        cv2.circle(img, center=(int(_x), int(_y)), color=(255, 0, 0), radius=7, thickness=2)
    cv2.imwrite("D:\\download_cache\\anime_data" + outname, img)

def center_and_scale_joints(scale, offset, joints,center,scale_level=0.8,trans=None):
    for joint in joints:
        joint += offset
    output = []
    for idx,joint in enumerate(joints):
        joint -= center
        joint = joint*(1-(1-scale)*scale_level)
        if trans is not None and idx in [0,14,15]:
            x = joint[0]
            y = joint[1]
            y = -y
            cos = trans[0]
            sin = trans[1]
            joint = np.array([cos*x+sin*y,-(-sin*x+cos*y)])

        output.append(joint + center)

    return np.array(output)

def align(real_kps,anime_kps,visualization=False):
    # real and anime kps should be (n,2) numpy array
    neck_a = anime_kps[1,:]
    lhip_a = anime_kps[13,:]
    rhip_a = anime_kps[17,:]
    neck_r = real_kps[1,:]
    lhip_r = real_kps[11,:]
    rhip_r = real_kps[8,:]
    hip_a = (lhip_a[1]+rhip_a[1]) / 2.0
    hip_r = (lhip_r[1]+rhip_r[1]) / 2.0
    offset = neck_a-neck_r
    out = []
    for joint in deepcopy(real_kps):
        out.append(joint + offset)
    return out



if __name__ == '__main__':
    st = time.time()
    # dirs = r"D:\download_cache\anime_data\trainK"
    # items = os.listdir(dirs)
    # items.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # for item in items:
    #     oldname = os.path.join(dirs,item)
    #     new = os.path.join(dirs,"_".join(item.split("_")[:-1])+"_"+str(int(item.split("_")[-1].split(".")[0])-1)+".jpg.npy")
    #     os.rename(oldname,new)
    # dance_61_10_Bakugou_8 dance_63_8_Kaito_9
    around_modified("dance_61_10_Bakugou_8")
    # gen_dataset_posetransfer()
    print(time.time()-st)

import numpy as np
import random,math
import pandas as pd
import os,time,shutil
import cv2,csv
from data_prepare.smooth_pose import smooth_json_pose
from data_prepare.video2vmd import video2keypoints
from pose_estimate.process_imgs import load_json
import scipy.io as sio
from copy import deepcopy
from util import get_file_encoding

def get_PoseEstimation_txt(folder):
    with open("test_id.txt","w+") as f:
        for item in os.listdir(folder):
            f.write(item+"\n")

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
                  output_dir = "D:/download_cache/anime_data/normK/",vis=None):
    # default to "D:\\download_cache\\anime_data\\vis_img\\"
    kpss = os.listdir(kps_r)
    img_nums = len(kpss)
    # max_vis = vis_num
    for idx in range(img_nums):
        real_kps = np.load(os.path.join(kps_r, kpss[idx]))
        anime_kps = np.load(os.path.join(kps_a, kpss[idx]))
        modified_kps = kps_Normalize(real_kps,anime_kps,scale_level=0.8)
        np.save(output_dir + kpss[idx], modified_kps)
        if vis is not None and kpss[idx].split("_")[3] in vis:
            img_name = "D:\\download_cache\\anime_data\\train\\" + ".".join(kpss[idx].split(".")[:-1])
            plot_points(img_name, modified_kps, ".".join(kpss[idx].split(".")[:-1]))


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

    theta_a = math.atan((head_a[1]-neck_a[1]+0.001)/(head_a[0]-neck_a[0]+0.001))
    theta_r = math.atan((head_r[1] - neck_r[1]+0.001) / (head_r[0] - neck_r[0]+0.001))
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
    ranges = [int(frame_num)*3,int(frame_num)*6+6]
    index_name = filename + '.jpg.npy'
    img_name = "D:\\download_cache\\anime_data\\train\\" + filename+'.jpg'
    kps_r = r"D:\download_cache\anime_data\trainK"
    kps_a = r"D:\download_cache\anime_data\tmpK"
    cnt = ranges[0]
    anime_kps = np.load(os.path.join(kps_a,index_name))
    real_kps = np.load(os.path.join(kps_r,index_name))
    for item in pose_all[ranges[0]:ranges[1], :, :]:
        img = cv2.imread(img_name)
        item_ = kps_Normalize(deepcopy(item), anime_kps)
        for _joint in item_:
            _x, _y = _joint
            cv2.circle(img, center=(int(_x), int(_y)), color=(255, 0, 0), radius=7, thickness=2)

        cv2.imwrite("D:\\download_cache\\anime_data" + f"/neigh{cnt}.jpg", img)
        cnt += 1

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
    # cv2.imwrite("D:\\download_cache\\anime_data\\vis_img\\" + outname, img)
    cv2.imwrite("D:\\download_cache\\anime_data\\" + outname, img)

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

def genPosetransferPair(input_dir = r"D:\download_cache\anime_data\train",
                        output_file = r"D:\download_cache\anime_data\anime-pairs-train.csv"):
    a = os.listdir(input_dir)
    a.sort(key=lambda x: "_".join(x.split("_")[:-1]))
    last = a[0]
    tmp_list = []
    out_list = []
    for item in a:
        if ("_".join(item.split("_")[:-1])) == ("_".join(last.split("_")[:-1])):
            tmp_list.append(item)
        else:
            for x in tmp_list:
                for y in tmp_list:
                    if x != y:
                        out_list.append([x, y])
            tmp_list = []
            tmp_list.append(item)
        last = item
    df = pd.DataFrame(out_list,columns=["from","to"])
    df.to_csv(output_file,index=None)

class Vmd:
    def __init__(self):
        pass

    @staticmethod
    def from_file(filename, model_name_encode="shift-JIS"):

        with open(filename, "rb") as f:
            from functools import reduce
            array = bytes(reduce(lambda x, y: x+y, list(f)))

        vmd = Vmd()

        VersionInformation = array[:30].decode("ascii")
        if VersionInformation.startswith("Vocaloid Motion Data file"):
            vision = 1
        elif VersionInformation.startswith("Vocaloid Motion Data 0002"):
            vision = 2
        else:
            raise Exception("unknow vision")

        vmd.vision = vision

        vmd.model_name = array[30: 30+10*vision].split(bytes([0]))[0].decode(model_name_encode)
        vmd.bone_keyframe_number = int.from_bytes(array[30+10*vision: 30+10*vision+4], byteorder='little', signed=False)
        vmd.bone_keyframe_record = []
        vmd.morph_keyframe_record = []
        vmd.camera_keyframe_record = []
        vmd.light_keyframe_record = []

        current_index = 34+10 * vision
        import struct
        for i in range(vmd.bone_keyframe_number):
            vmd.bone_keyframe_record.append({
                "BoneName": array[current_index: current_index+15].split(bytes([0]))[0].decode("shift-JIS"),
                "FrameTime": struct.unpack("<I", array[current_index+15: current_index+19])[0],
                "Position": {"x": struct.unpack("<f", array[current_index+19: current_index+23])[0],
                            "y": struct.unpack("<f", array[current_index+23: current_index+27])[0],
                            "z": struct.unpack("<f", array[current_index+27: current_index+31])[0]
                            },
                "Rotation":{"x": struct.unpack("<f", array[current_index+31: current_index+35])[0],
                            "y": struct.unpack("<f", array[current_index+35: current_index+39])[0],
                            "z": struct.unpack("<f", array[current_index+39: current_index+43])[0],
                            "w": struct.unpack("<f", array[current_index+43: current_index+47])[0]
                            },
                "Curve":{
                    "x":(array[current_index+47], array[current_index+51], array[current_index+55], array[current_index+59]),
                    "y":(array[current_index+63], array[current_index+67], array[current_index+71], array[current_index+75]),
                    "z":(array[current_index+79], array[current_index+83], array[current_index+87], array[current_index+91]),
                    "r":(array[current_index+95], array[current_index+99], array[current_index+103], array[current_index+107])
                }


            })
            current_index += 111

        # vmd['MorphKeyFrameNumber'] = int.from_bytes(array[current_index: current_index+4], byteorder="little", signed=False)
        vmd.morph_keyframe_number = int.from_bytes(array[current_index: current_index+4], byteorder="little", signed=False)
        current_index += 4

        for i in range(vmd.morph_keyframe_number):
            vmd.morph_keyframe_record.append({
                'MorphName': array[current_index: current_index+15].split(bytes([0]))[0].decode("shift-JIS"),
                'FrameTime': struct.unpack("<I", array[current_index+15: current_index+19])[0],
                'Weight': struct.unpack("<f", array[current_index+19: current_index+23])[0]
            })
            current_index += 23

        vmd.camera_keyframe_number = int.from_bytes(array[current_index: current_index+4], byteorder="little", signed=False)
        current_index += 4

        for i in range(vmd.camera_keyframe_number):
            vmd.camera_keyframe_record.append({
                'FrameTime': struct.unpack("<I", array[current_index: current_index+4])[0],
                'Distance': struct.unpack("<f", array[current_index+4: current_index+8])[0],
                "Position": {"x": struct.unpack("<f", array[current_index+8: current_index+12])[0],
                            "y": struct.unpack("<f", array[current_index+12: current_index+16])[0],
                            "z": struct.unpack("<f", array[current_index+16: current_index+20])[0]
                            },
                "Rotation":{"x": struct.unpack("<f", array[current_index+20: current_index+24])[0],
                            "y": struct.unpack("<f", array[current_index+24: current_index+28])[0],
                            "z": struct.unpack("<f", array[current_index+28: current_index+32])[0]
                            },
                "Curve": tuple(b for b in array[current_index+32: current_index+36]),
                "ViewAngle": struct.unpack("<I", array[current_index+56: current_index+60])[0],
                "Orthographic": array[60]
            })
            current_index += 61

        vmd.light_keyframe_number = int.from_bytes(array[current_index: current_index+4], byteorder="little", signed=False)
        current_index += 4

        for i in range(vmd.light_keyframe_number):
            vmd.light_keyframe_record.append({
                'FrameTime': struct.unpack("<I", array[current_index: current_index+4])[0],
                'Color': {
                    'r': struct.unpack("<f", array[current_index+4: current_index+8])[0],
                    'g': struct.unpack("<f", array[current_index+8: current_index+12])[0],
                    'b': struct.unpack("<f", array[current_index+12: current_index+16])[0]
                },
                'Direction':{"x": struct.unpack("<f", array[current_index+16: current_index+20])[0],
                            "y": struct.unpack("<f", array[current_index+20: current_index+24])[0],
                            "z": struct.unpack("<f", array[current_index+24: current_index+28])[0]
                            }
            })
            current_index += 28

        vmd_dict = {}
        vmd_dict['Vision'] = vision
        vmd_dict['ModelName'] = vmd.model_name
        vmd_dict['BoneKeyFrameNumber'] = vmd.bone_keyframe_number
        vmd_dict['BoneKeyFrameRecord'] = vmd.bone_keyframe_record
        vmd_dict['MorphKeyFrameNumber'] = vmd.morph_keyframe_number
        vmd_dict['MorphKeyFrameRecord'] = vmd.morph_keyframe_record
        vmd_dict['CameraKeyFrameNumber'] = vmd.camera_keyframe_number
        vmd_dict['CameraKeyFrameRecord'] = vmd.camera_keyframe_record
        vmd_dict['LightKeyFrameNumber'] = vmd.light_keyframe_number
        vmd_dict['LightKeyFrameRecord'] = vmd.light_keyframe_record

        vmd.dict = vmd_dict

        return vmd

    @staticmethod
    def _quaternion_to_EulerAngles(x, y, z, w):
        import numpy as np
        X = np.arcsin(2 * w * x - 2 * y * z) / np.pi * 180
        Y = -np.arctan2(2 * w * y + 2 * x * z, 1 - 2 * x ** 2 - 2 * y ** 2) / np.pi * 180
        Z = -np.arctan2(2 * w * z + 2 * x * y, 1 - 2 * x ** 2 - 2 * z ** 2) / np.pi * 180
        return X, Y, Z

    @property
    def euler_dict(self):
        from copy import deepcopy
        res_dict = deepcopy(self.dict)
        for index, d in enumerate(res_dict['BoneKeyFrameRecord']):
            x = d["Rotation"]["x"]
            y = d["Rotation"]["y"]
            z = d["Rotation"]["z"]
            w = d["Rotation"]["w"]
            X, Y, Z = Vmd._quaternion_to_EulerAngles(x, y, z, w)
            res_dict['BoneKeyFrameRecord'][index]["Rotation"] = {
                "X": X,
                "Y": Y,
                "Z": Z
            }
        return res_dict

    @property
    def four_dict(self):
        from copy import deepcopy
        res_dict = deepcopy(self.dict)
        for index, d in enumerate(res_dict['BoneKeyFrameRecord']):
            x = d["Rotation"]["x"]
            y = d["Rotation"]["y"]
            z = d["Rotation"]["z"]
            w = d["Rotation"]["w"]
            res_dict['BoneKeyFrameRecord'][index]["Rotation"] = {
                "x": x,
                "y": y,
                "z": z,
                "w": w,
            }
        return res_dict

def getAngleFromVMD(vmd_file,reference_csv = "D:\download_cache\PMXmodel\index.csv"):
    # e.g. input: D:\download_cache\PMXmodel\VMDfile\dance_81_yousa.vmd"
    df = pd.read_csv(reference_csv,header=None)
    n_frames = 0
    for item in df.values:
        if "_".join(os.path.basename(vmd_file).split("_")[0:2])==item[0]:
            n_frames = int(item[-1])
            break
    bone_frame_dic = {
        "上半身": [],
        "上半身2": [],
        "下半身": [],
        "首": [],
        "頭": [],
        "左肩": [],
        "左腕": [],
        "左ひじ": [],
        "右肩": [],
        "右腕": [],
        "右ひじ": [],
        "左足": [],
        "左ひざ": [],#empty
        "右足": [],
        "右ひざ": [],#empty
        "センター": [], # from this ↓, we all get the position
        "グルーブ": [],
        "左足ＩＫ": [],
        "右足ＩＫ": []
    }
    vmd = Vmd.from_file(vmd_file)
    data = vmd.euler_dict['BoneKeyFrameRecord']
    # TODO for IK, the rotation is quite meaning less. we need the position of this guy. and the same for センター and グルーブ, we just get the position
    # this position is offset value
    for item in data:
        name = item['BoneName']
        if name in  ["左足ＩＫ","右足ＩＫ", "センター" ,"グルーブ"]:
            bone_frame_dic[name].append([item['FrameTime'], item['Position']])
        else:
            bone_frame_dic[name].append([item['FrameTime'], item['Rotation']])
    over_all = []
    for x, y in bone_frame_dic.items():
        last = None
        tmp = []
        for item in y:
            num, euler_angles = item
            if last is not None:
                length = num - last[0]
                try:
                    X = np.linspace(last[1]['X'], euler_angles['X'], length, endpoint=False).tolist()
                    Y = np.linspace(last[1]['Y'], euler_angles['Y'], length, endpoint=False).tolist()
                    Z = np.linspace(last[1]['Z'], euler_angles['Z'], length, endpoint=False).tolist()
                except KeyError:
                    X = np.linspace(last[1]['x'], euler_angles['x'], length, endpoint=False).tolist()
                    Y = np.linspace(last[1]['y'], euler_angles['y'], length, endpoint=False).tolist()
                    Z = np.linspace(last[1]['z'], euler_angles['z'], length, endpoint=False).tolist()
                for i in zip(X, Y, Z):
                    tmp.append(list(i))
            last = [num, euler_angles]
        if tmp:
            for i in range(n_frames-len(tmp)):
                tmp.append(tmp[-1])
            over_all.append(tmp)
    over_all = np.swapaxes(np.array(over_all),0,1)
    return over_all


def getAngleFromVMD_v2(vmd_file,reference_csv = "D:\download_cache\PMXmodel\index.csv"):
    '''
    this one is for 四元数
    :param vmd_file:
    :param reference_csv:
    :return:
    '''
    # e.g. input: D:\download_cache\PMXmodel\VMDfile\dance_81_yousa.vmd"
    df = pd.read_csv(reference_csv,header=None)
    n_frames = 0
    for item in df.values:
        if "_".join(os.path.basename(vmd_file).split("_")[0:2])==item[0]:
            n_frames = int(item[-1])
            break
    bone_frame_dic = {
        "上半身": [],
        "上半身2": [],
        "下半身": [],
        "首": [],
        "頭": [],
        "左肩": [],
        "左腕": [],
        "左ひじ": [],
        "右肩": [],
        "右腕": [],
        "右ひじ": [],
        "左足": [],
        "左ひざ": [],#empty
        "右足": [],
        "右ひざ": [],#empty
        "センター": [], # from this ↓, we all get the position
        "グルーブ": [],
        "左足ＩＫ": [],
        "右足ＩＫ": []
    }
    vmd = Vmd.from_file(vmd_file)
    data = vmd.four_dict['BoneKeyFrameRecord']
    # TODO for IK, the rotation is quite meaning less. we need the position of this guy. and the same for センター and グルーブ, we just get the position
    # this position is offset value
    for item in data:
        name = item['BoneName']
        if name in  ["左足ＩＫ","右足ＩＫ", "センター" ,"グルーブ"]:
            bone_frame_dic[name].append([item['FrameTime'], item['Position']])
        else:
            bone_frame_dic[name].append([item['FrameTime'], item['Rotation']])
    over_all = []
    for x, y in bone_frame_dic.items():
        last = None
        tmp = []
        for item in y:
            num, euler_angles = item
            if last is not None:
                length = num - last[0]
                try:
                    W = np.linspace(last[1]['w'], euler_angles['w'], length, endpoint=False).tolist()
                    X = np.linspace(last[1]['x'], euler_angles['x'], length, endpoint=False).tolist()
                    Y = np.linspace(last[1]['y'], euler_angles['y'], length, endpoint=False).tolist()
                    Z = np.linspace(last[1]['z'], euler_angles['z'], length, endpoint=False).tolist()
                except KeyError:
                    W = np.linspace(0, 0, length, endpoint=False).tolist()
                    X = np.linspace(last[1]['x'], euler_angles['x'], length, endpoint=False).tolist()
                    Y = np.linspace(last[1]['y'], euler_angles['y'], length, endpoint=False).tolist()
                    Z = np.linspace(last[1]['z'], euler_angles['z'], length, endpoint=False).tolist()
                for i in zip(W, X, Y, Z):
                    tmp.append(list(i))
            last = [num, euler_angles]
        if tmp:
            for i in range(n_frames-len(tmp)):
                tmp.append(tmp[-1])
            over_all.append(tmp)
    over_all = np.swapaxes(np.array(over_all),0,1)
    return over_all


def getBoneFromCsv(bone_csv_file):
    '''
    Order of Output:
    0-center 1-groove 2-r_leg_IK 3-l_leg_IK 4-upperbody 5-upperbody2 6-neck 7-head 8-r_eye 9-l_eye 10-r_shoulder
    11-arm 12-r_elbow 13-r_hand 14-l_shoulder 15-l_arm 16-l_elbow 17-l_hand 18-lowerbody 19-r_leg 20-r_knee 21-r_ankle 22-l_leg
    23-l_knee 24-l_ankle
    :param bone_csv_file:
    :return:
    '''
    with open(bone_csv_file, "r", encoding=get_file_encoding(bone_csv_file)) as bf:
        reader = csv.reader(bf)
        output =np.ones((25,4)).tolist()
        for row in reader:
            if row[1] == "上半身" or row[2].lower() == "upperbody":
                output[4] = [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "グルーブ" or row[2].lower() == "groove":
                output[1]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "上半身2" or row[2].lower() == "upperbody2":
                output[5]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "首" or row[2].lower() == "neck":
                output[6]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "頭" or row[2].lower() == "head":
                output[7]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "左肩" or row[2].lower() == "l_shoulder":
                output[14]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "左腕" or row[2].lower() == "l_arm":
                output[15]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "左ひじ" or row[2].lower() == "l_elbow":
                output[16]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "右肩" or row[2].lower() == "r_shoulder":
                output[10]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "右腕" or row[2].lower() == "arm":
                output[11]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "右ひじ" or row[2].lower() == "r_elbow":
                output[12]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "右手首" or row[2].lower() == "r_hand":
                output[13]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "左手首" or row[2].lower() == "l_hand":
                output[17]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "下半身" or row[2].lower() == "lower body":
                output[18]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "左足" or row[2].lower() == "leg_l":
                output[22]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "左ひざ" or row[2].lower() == "knee_l":
                output[23]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "左足首" or row[2].lower() == "ankle_l":
                output[24] = [row[2], float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "右足" or row[2].lower() == "leg_r":
                output[19]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "右ひざ" or row[2].lower() == "knee_r":
                output[20]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "右足首" or row[2].lower() == "ankle_r":
                output[21]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "右目" or row[2].lower() == "right eye":
                output[8] = [row[2], float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "左目" or row[2].lower() == "left eye":
                output[9] = [row[2], float(row[5]), float(row[6]), float(row[7])]
            if row[0] == "Bone" and (
                    row[1] == "左足ＩＫ" or row[2].lower() == "leg ik_l"):
                output[3]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[0] == "Bone" and (
                    row[1] == "右足ＩＫ" or row[2].lower() == "leg ik_r"):
                output[2]= [row[2],float(row[5]), float(row[6]), float(row[7])]
            if row[1] == "センター" or row[2].lower() == "center":
                output[0]= [row[2],float(row[5]), float(row[6]), float(row[7])]
    return output

def simpleShowForCsv(model_csv = "D:\download_cache\PMXmodel\CSVfile\ka.csv",img_path =r"D:\download_cache\PMXmodel\test.jpg"):
    bones = getBoneFromCsv(model_csv)
    img = cv2.imread(img_path)
    for idx, _joint in enumerate(bones[:-2]):
        _, _x, _y, _ = _joint
        cv2.circle(img, center=(int(_x * 42) + 960, 960 - int(_y * 42)), color=(255, 0, 0), radius=5, thickness=1)
        # cv2.imwrite("D:\\download_cache\\anime_data\\vis_img\\" + outname, img)
    cv2.imwrite(os.path.join(os.path.dirname(img_path),"out.jpg"), img)

def rotateMatrix(a,b,c,mode='ZXY'):
    # enter euler angle a,b,c(degree format) for axis-X,Y,Z

    # MMD order is YXZ
    a = math.radians(a)
    b = math.radians(b)
    c = math.radians(c)
    Rx = np.array([[1,0,0],[0,math.cos(a),math.sin(a)],[0,-math.sin(a),math.cos(a)]],dtype=np.float32)
    Ry = np.array([[math.cos(b),0,-math.sin(b)],[0,1,0],[math.sin(b),0,math.cos(b)]],dtype=np.float32)
    Rz = np.array([[math.cos(c),math.sin(c),0],[-math.sin(c),math.cos(c),0],[0,0,1]],dtype=np.float32)
    # ZXY DirectXMath [XYZ,XZY,YXZ,YZX,ZXY,ZYX]
    if mode == "XYZ":
        return np.dot(np.dot(Rx, Ry),Rz)
    elif mode== "YXZ":
        return np.dot(np.dot(Ry, Rx), Rz)
    elif mode=="ZXY":
        return np.dot(np.dot(Rz, Rx), Ry)
    elif mode=="ZYX":
        return np.dot(np.dot(Rz, Ry), Rx)
    elif mode=="XZY":
        return np.dot(np.dot(Rx, Rz), Ry)
    elif mode== "YZX":
        return np.dot(np.dot(Ry, Rz), Rx)

def getRotationMatrix(fourElm):
    w,x,y,z = fourElm
    return np.asarray([[1-2*(y**2)-2*(z**2),2*x*y-2*w*z,2*x*z+2*w*y],
                       [2*x*y+2*w*z,1-2*(x**2)-2*(z**2),2*y*z-2*w*x],
                       [2*x*z-2*w*y,2*y*z+2*w*x,1-2*(x**2)-2*(y**2)]])

def getRotMbyTwoVector(v1,v2):
    theta = float(math.acos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))
    a1,a2,a3 = v1.tolist()
    b1,b2,b3 = v2.tolist()
    x= a2*b3-a3*b2
    y= a3*b1-a1*b3
    z= a1*b2-a2*b1
    return getRotationMatrix([theta,x,y,z])

def getSingle3DpointsFrom4elm(euler_rotates, ori_points,vis=False):
    # here euler_rotates is four element number (rotation w,x,y,z)!!!
    ori_points = np.asarray(np.asarray(ori_points)[:, 1:], dtype=np.float64)


    # # center/groove
    # num = len(ori_points)
    # for i in range(num):
    #     ori_points[i] += np.asarray(euler_rotates[13][1:])
    #     ori_points[i] += np.asarray(euler_rotates[14][1:])
    #
    # #upperbody
    # rotM_upperbody = getRotationMatrix(euler_rotates[0])
    # ref_upperbody = np.asarray(ori_points[4])  # l-shoulder
    # points = np.asarray(ori_points[5:18])
    # num = len(points)
    # for i in range(num):
    #     points[i] = np.dot(rotM_upperbody, points[i] - ref_upperbody) + ref_upperbody
    #
    # #upperbody2
    # rotM_upperbody2 = getRotationMatrix(euler_rotates[1])
    # ref_upperbody2 = np.asarray(ori_points[5])  # l-shoulder
    # points = np.asarray(ori_points[6:18])
    # num = len(points)
    # for i in range(num):
    #     points[i] = np.dot(rotM_upperbody2, points[i] - ref_upperbody2) + ref_upperbody2
    #
    # # neck
    # rotM_neck = getRotationMatrix(euler_rotates[3])
    # ref_neck = np.asarray(ori_points[6])  # l-shoulder
    # points = np.asarray(ori_points[7:10])
    # num = len(points)
    # for i in range(num):
    #     points[i] = np.dot(rotM_neck, points[i] - ref_neck) + ref_neck
    #
    # # head
    # rotM_head = getRotationMatrix(euler_rotates[4])
    # ref_head = np.asarray(ori_points[7])  # l-shoulder
    # points = np.asarray(ori_points[8:10])
    # num = len(points)
    # for i in range(num):
    #     points[i] = np.dot(rotM_head, points[i] - ref_head) + ref_head


    # # l-shoulder
    # rotM_shoulder1 = getRotationMatrix(euler_rotates[5])
    # ref_shoulder1 = np.asarray(ori_points[14])  # l-shoulder
    # points = np.asarray(ori_points[15:18])
    # num = len(points)
    # for i in range(num):
    #     points[i] = np.dot(rotM_shoulder1, points[i] - ref_shoulder1) + ref_shoulder1

    # l-arm
    rotM_arm1 = getRotationMatrix(euler_rotates[6])
    ref_arm1 = np.asarray(ori_points[15])  # l-shoulder
    points = np.asarray(ori_points[16:18])
    num = len(points)
    for i in range(num):
        points[i] = np.dot(rotM_arm1, points[i] - ref_arm1) + ref_arm1

    # l-elbow
    rotM_elbow1 = getRotationMatrix(euler_rotates[7])
    ref_elbow1 = np.asarray(ori_points[16])  # l-
    points = np.asarray(ori_points[17:18])
    num = len(points)
    for i in range(num):
        points[i] = np.dot(rotM_elbow1, points[i]- ref_elbow1)+ ref_elbow1

    # # r-shoulder
    # rotM_shoulder2 = getRotationMatrix(euler_rotates[8])
    # ref_shoulder2 = np.asarray(ori_points[10])  # l-shoulder
    # points = np.asarray(ori_points[11:14])
    # num = len(points)
    # for i in range(num):
    #     points[i] = np.dot(rotM_shoulder2, points[i] - ref_shoulder2) + ref_shoulder2
    #
    # # r-arm
    # rotM_arm2 = getRotationMatrix(euler_rotates[9])
    # ref_arm2 = np.asarray(ori_points[11])  # l-arm
    # points = np.asarray(ori_points[12:14])
    # num = len(points)
    # for i in range(num):
    #     points[i] = np.dot(rotM_arm2, points[i] - ref_arm2) + ref_arm2
    #
    # # r-elbow
    # rotM_elbow2 = getRotationMatrix(euler_rotates[10])
    # ref_elbow2 = np.asarray(ori_points[12])  # l-
    # points = np.asarray(ori_points[13:14])
    # num = len(points)
    # for i in range(num):
    #     points[i] = np.dot(rotM_elbow2, points[i] - ref_elbow2) + ref_elbow2



    if vis:
         Map_3Dpoints_2D(ori_points[14:18, :], vis)
    return ori_points


def getSingleFrame3Dpoints(euler_rotates, ori_points,vis=False):
    '''
    :param euler_rotates:  from getAngleFromVMD() -> sample one -> (17,3)
    #['0-上半身', '1-上半身2', '2-下半身', '3-首', '4-頭', '5-左肩', '6-左腕', '7-左ひじ', '8-右肩', '9-右腕', '10-右ひじ', '
    #   11- 左足', 12-'右足', 13-'センター', 14-'グルーブ', 15-'左足ＩＫ', 16-'右足ＩＫ']
    :param ori_points:  from getBoneFromCsv()-> (23,3)
    :return:
    '''
    # 现在假设只有2个rotation， 是 上半身,upperbody2
    ori_points = np.asarray(np.asarray(ori_points)[:, 1:], dtype=np.float64)

    # center/groove
    # num = len(ori_points)
    # for i in range(num):
    #     ori_points[i]+= np.asarray(euler_rotates[13])
    #     ori_points[i] += np.asarray(euler_rotates[14])


    # #upperbody
    # euler_rotate = euler_rotates[0]
    # rotM = rotateMatrix(euler_rotate[0],euler_rotate[1],euler_rotate[2],mode="ZXY")
    # ref = np.asarray(ori_points[4]) # upperbody
    # points = np.asarray(ori_points[5:18]) #eliminate name
    # points = dealSingleRotation(points,rotM,ref).tolist()
    # for i in range(5,18):
    #     ori_points[i] = points[i-5]
    # #upperbody2
    # euler_rotate = euler_rotates[1]
    # rotM = rotateMatrix(euler_rotate[0], euler_rotate[1], euler_rotate[2], mode="ZXY")
    # ref = np.asarray(ori_points[5])  # upperbody2
    # points = np.asarray(ori_points[6:18])
    # points = dealSingleRotation(points, rotM, ref).tolist()
    # for i in range(6, 18):
    #     ori_points[i] = points[i - 6]
    # # neck
    # euler_rotate = euler_rotates[3]
    # rotM = rotateMatrix(euler_rotate[0], euler_rotate[1], euler_rotate[2], mode="ZXY")
    # ref = np.asarray(ori_points[6])  # neck
    # points = np.asarray(ori_points[7:10])
    # points = dealSingleRotation(points, rotM, ref).tolist()
    # for i in range(7, 10):
    #     ori_points[i] = points[i - 7]
    # # head
    # euler_rotate = euler_rotates[4]
    # rotM = rotateMatrix(euler_rotate[0], euler_rotate[1], euler_rotate[2], mode="ZXY")
    # ref = np.asarray(ori_points[7])  # head
    # points = np.asarray(ori_points[8:10])
    # points = dealSingleRotation(points, rotM, ref).tolist()
    # for i in range(8, 10):
    #     ori_points[i] = points[i - 8]

    # # l-shoulder
    # euler_rotate = euler_rotates[5]
    # rotM = rotateMatrix(euler_rotate[0], euler_rotate[1], euler_rotate[2], mode="ZXY")
    # ref = np.asarray(ori_points[14])  # l-shoulder
    # points = np.asarray(ori_points[15:18])
    # points = dealSingleRotation(points, rotM, ref).tolist()
    # for i in range(15, 18):
    #     ori_points[i] = points[i - 15]
    #
    # # l-arm
    # euler_rotate = euler_rotates[6]
    # rotM = rotateMatrix(euler_rotate[0], euler_rotate[1], euler_rotate[2], mode="ZXY")
    # ref = np.asarray(ori_points[15])  # l-arm
    # points = np.asarray(ori_points[16:18])
    #
    # local2world = rotateMatrix(0, 0, -49)
    # world2local = np.linalg.inv(local2world)
    # rotM_ = np.dot(local2world,np.dot(rotM,world2local))
    #
    # points = dealSingleRotation(points, rotM_, ref).tolist()
    #
    # for i in range(16, 18):
    #     ori_points[i] = points[i - 16]
    #
    # # l-elbow
    # euler_rotate = euler_rotates[7]
    # rotM = rotateMatrix(euler_rotate[0], euler_rotate[1], euler_rotate[2], mode="ZXY")
    # ref = np.asarray(ori_points[16])  # l-eblow
    # points = np.asarray(ori_points[17:18])
    #
    # local2world = rotateMatrix(0, 0, -42)
    # world2local = np.linalg.inv(local2world)
    # rotM_ = np.dot(local2world, np.dot(rotM, world2local))
    #
    # points = dealSingleRotation(points, rotM_, ref).tolist()
    # for i in range(17, 18):
    #     ori_points[i] = points[i - 17]
    #
    # # r-shoulder
    # euler_rotate = euler_rotates[8]
    # rotM = rotateMatrix(euler_rotate[0], euler_rotate[1], euler_rotate[2], mode="ZXY")
    # ref = np.asarray(ori_points[10])  # l-shoulder
    # points = np.asarray(ori_points[11:14])
    # points = dealSingleRotation(points, rotM, ref).tolist()
    # for i in range(11, 14):
    #     ori_points[i] = points[i - 11]

    # r-arm
    euler_rotate = euler_rotates[9]
    rotM = rotateMatrix(45, 50, 30, mode="ZXY")
    ref = np.asarray(ori_points[11])  # l-arm
    points = np.asarray(ori_points[12:14])

    num = len(points)
    for i in range(num):
        # print(points[i].T)
        # TODO -90<X<90, -180<y,z<180, xz left-hand,y is right-hand
        # points[i] = np.dot(rotateMatrix(0, 0, -132),points[i]-ref)
        # points[i] = np.dot(rotM, points[i])
        # points[i] = np.dot(np.linalg.inv(rotateMatrix(0, 0, -132)), points[i])+ref

        points[i] = points[i] - ref
        points[i] = np.dot(rotateMatrix(0, 50, 0), points[i])
        points[i] = np.dot(rotateMatrix(45, 0, 0), points[i])
        points[i] = np.dot(rotateMatrix(0, 0, 30), points[i])


        points[i] = points[i] + ref


    # # r-elbow
    # euler_rotate = euler_rotates[10]
    # rotM = rotateMatrix(euler_rotate[0], euler_rotate[1], euler_rotate[2], mode="ZXY")
    # ref = np.asarray(ori_points[12])  # l-
    # points = np.asarray(ori_points[13:14])
    # last_rot = deepcopy(rotM)
    #
    # num = len(points)
    # for i in range(num):
    #     # print(points[i].T)
    #     # TODO test the rotateMatrix right or not
    #     points[i] = np.dot(rotateMatrix(0, 0, -138), points[i] - ref)
    #     points[i] = np.dot(np.linalg.inv(last_rot), points[i])
    #     points[i] = np.dot(rotM, points[i])
    #     points[i] = np.dot(last_rot, points[i])
    #     points[i] = np.dot(np.linalg.inv(rotateMatrix(0, 0, -138)), points[i]) + ref



    if vis:
         Map_3Dpoints_2D(ori_points[4:18, :], vis)


    return ori_points

def dealSingleRotation(points,rotM,ref):
    num = len(points)
    for i in range(num):
        # print(points[i].T)
        points[i] = singlePointRotation(points[i],rotM,ref)
    return points

def singlePointRotation(point,rotM,ref):
    return np.dot(rotM,(point-ref))+ref

def Map_3Dpoints_2D(joint_world,vis=False):
    # just enter a 3-D world point. (x,y,z), output what we want
    camera = {
        'R':rotateMatrix(0,0,0),
        'T':[0,10,0],
        'f':45
    }
    T = np.asarray(camera['T'])
    R = camera['R']
    f = camera['f']
    joint_num = len(joint_world)
    # 世界坐标系 -> 相机坐标系
    # [R|t] world coords -> camera coords
    joint_cam = np.zeros((joint_num, 3))  # joint camera
    for i in range(joint_num):  # joint i
        joint_cam[i] = np.dot(R, joint_world[i] - T)  # R * (pt - T)
    F = np.asarray([[f,0,0],[0,-f,0],[0,0,1]])
    joint_mmd = np.zeros((joint_num, 3))
    for i in range(joint_num):
        joint_mmd[i] = np.dot(F,joint_cam[i])
    for i in range(joint_num):
        joint_mmd[i] = joint_mmd[i]+np.asarray([960,540,0])

    if vis:
        img_path = r"D:\download_cache\PMXmodel\test.jpg"
        img = cv2.imread(img_path)
        for idx in range(joint_num):
            _joint = joint_mmd[idx]
            _x, _y, _ = _joint
            cv2.circle(img, center=(int(_x), int(_y)), color=(255, 0, 0), radius=7, thickness=3)
        cv2.imwrite(os.path.join(os.path.dirname(img_path), "out.jpg"), img)

    # the 2-D (x,y) is the keypoints we need.
    return joint_mmd

if __name__ == '__main__':
    st = time.time()
    # get_clip_kps(r"D:\download_cache\PMXmodel\video")
    # gen_dataset_posetransfer(kps_dir=r"D:\download_cache\PMXmodel\VIDEOkps",video_dir=r"D:\download_cache\PMXmodel\OUTPUTclips",
    #                      out_dir = r"D:\download_cache\anime_data2",interval=4)
    # get_PoseEstimation_txt(r"D:\download_cache\anime_data2\train")
    # kps_Normalize_dir(kps_r=r"D:\download_cache\anime_data\trainK", kps_a=r"D:\download_cache\anime_data\tmpK",
    #                   output_dir = "D:/download_cache/anime_data/normK/",vis=["AnsieNight","Kogawa","naraka"])
    # genPosetransferPair()
    # dance_61_10_Bakugou_8 dance_63_8_Kaito_9
    # around_modified("dance_10_2_sunshang_0")
    ori_angles = getAngleFromVMD_v2(r"D:\download_cache\PMXmodel\VMDfile\dance_63_ka.vmd") #frames_num, type, 3    63
    bones = getBoneFromCsv("D:\download_cache\PMXmodel\CSVfile\ka.csv")
    joints = []
    for idx, _joint in enumerate(bones[:-2]):
        _, _x, _y, _z = _joint
        joints.append([_x,_y,_z])
    input_angles = ori_angles[0].tolist()
    # getSingleFrame3Dpoints([[0.64,0.0,0.0],[0.0,-0.6,0.0],[6.4, 13.3, -6.7],[-28.5,0.6,-6.9],[-0.4,-34.1,0.6],[38.6,0.9,-1.4],
    #                        [16.2,7.9,1.4],[-4.1,-48.8,19.5],[30.0,-88.4,-29.2],[17.3,0.5,17.0],[20.4,-17.5,-52.0],[13.9,25.4,3.2]
    #                         ],bones,vis=True)
    getSingle3DpointsFrom4elm(input_angles,bones,vis=True)

    # pprint(data[5000])
    print(time.time()-st)
    #['上半身', '上半身2', '下半身', '首', '頭', '左肩', '左腕', '左ひじ', '右肩', '右腕', '右ひじ', '左足', '右足', 'センター', 'グルーブ', '左足ＩＫ', '右足ＩＫ']
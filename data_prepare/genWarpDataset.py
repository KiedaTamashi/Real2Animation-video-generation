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
from functools import reduce

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
        "左ひざ": [],
        "右足": [],
        "右ひざ": [],
        "センター": [],#x
        "グルーブ": [],#x
        "左足ＩＫ": [],
        "右足ＩＫ": []
    }
    vmd = Vmd.from_file(vmd_file)
    data = vmd.euler_dict['BoneKeyFrameRecord']
    for item in data:
        name = item['BoneName']
        bone_frame_dic[name].append([item['FrameTime'], item['Rotation']])
    over_all = []
    for x, y in bone_frame_dic.items():
        last = None
        tmp = []
        for item in y:
            num, euler_angles = item
            if last is not None:
                length = num - last[0]
                X = np.linspace(last[1]['X'], euler_angles['X'], length, endpoint=False).tolist()
                Y = np.linspace(last[1]['Y'], euler_angles['Y'], length, endpoint=False).tolist()
                Z = np.linspace(last[1]['Z'], euler_angles['Z'], length, endpoint=False).tolist()
                for i in zip(X, Y, Z):
                    tmp.append(list(i))
            last = [num, euler_angles]
        if tmp:
            for i in range(n_frames-len(tmp)):
                tmp.append(tmp[-1])
            over_all.append(tmp)
    over_all = np.array(over_all)
    return over_all

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
    getAngleFromVMD("D:\download_cache\PMXmodel\VMDfile\dance_81_yousa.vmd")


    # pprint(data[5000])
    print(time.time()-st)
    #['上半身', '上半身2', '下半身', '首', '頭', '左肩', '左腕', '左ひじ', '右肩', '右腕', '右ひじ', '左足', '右足', 'センター', 'グルーブ', '左足ＩＫ', '右足ＩＫ']
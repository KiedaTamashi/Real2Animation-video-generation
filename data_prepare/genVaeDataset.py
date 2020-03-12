import os
import pandas as pd
import shutil
import cv2
import time,random

'''
In this part, what we have is two dictionaries: for original video and output video from MMD
What we need is to generate frames pair.  
Reading the index.csv
The original video name is formated -> video_name+".mp4"
The mapping video is formated -> video_name+"_"+model_name+".avi"


Finally we output the frame pairs to two dictionary and generate a index.csv for further processing.
'''
def transfer_files(input_dir,output_dir,suffix=".avi"):
    files = os.listdir(input_dir)
    cnt=0
    for file in files:
        if file.endswith(suffix):
            shutil.move(os.path.join(input_dir,file),os.path.join(output_dir,file))
            cnt+=1
    print(cnt)


def extractFrames(ori_dir,map_dir,frame_base_dir,index_path,interval,start_num=0):
    '''
    :param ori_dir: .mp4
    :param map_dir: .avi
    :param interval: frame extraction maximum interval
    :param start_num: continue from last node.
    :return:
    '''
    x_frame_dir = os.path.join(frame_base_dir,"OriFrame")
    y_frame_dir = os.path.join(frame_base_dir,"MapFrame")
    if not os.path.exists(x_frame_dir):
        os.mkdir(x_frame_dir)
    if not os.path.exists(y_frame_dir):
        os.mkdir(y_frame_dir)

    ori_videoNames = os.listdir(ori_dir)
    map_videoNames = os.listdir(map_dir)
    ori_videoNames.sort(key=lambda x:int(x.split("_")[1]))
    file_pair_list = []
    cnt = 0
    random.seed(time.time())
    interval_new = random.randint(interval-2, interval+2)
    ban_list = ["Eren.avi","GTGoku.avi","RoseFreyja.avi","ifuleet.avi","inkling.avi",'luigi.avi',"Nikos.avi"]

    for mapVideoName in map_videoNames:
        flag = False
        video_2 = cv2.VideoCapture(os.path.join(map_dir,mapVideoName))
        for ban in ban_list:
            if mapVideoName.endswith(ban):
                flag=True
                break
        if flag:
            continue
        for oriVideoName in ori_videoNames:
            if not oriVideoName.startswith("_".join(mapVideoName.split("_")[0:3])):
                continue
            else:
                video_1 = cv2.VideoCapture(os.path.join(ori_dir, oriVideoName))
                frame_number = 0
                tick = 0
                while True:
                    ret1, img1 = video_1.read()
                    ret2, img2 = video_2.read()
                    if not ret1 or not ret2: break
                    if tick==interval_new:
                        print(mapVideoName + str(frame_number))
                        x_name = "x_" + str(mapVideoName.split("_")[-1])[:-4] + "_" + str(cnt)
                        y_name = "y_" + str(mapVideoName.split("_")[-1])[:-4] + "_" + str(cnt)  # TODO need check when actually use.
                        cv2.imwrite(os.path.join(x_frame_dir, x_name + '.jpg'), img1)
                        cv2.imwrite(os.path.join(y_frame_dir, y_name + '.jpg'), img2)
                        file_pair_list.append([x_name, y_name])
                        cnt += 1
                        random.seed(time.time())
                        interval_new = random.randint(interval - 2, interval + 2)
                        tick = 0
                    else:
                        tick += 1
                    frame_number += 1
                video_1.release()
        video_2.release()
    df = pd.DataFrame(file_pair_list)
    df.to_csv(index_path,index=None,header=None)

def add_condition(index_path):
    '''
    input file formated as [oriFrameName,mapFrameName]
    output file formated as [oriFrameName,mapFrameName,conditionFrameName]
    '''
    df = pd.read_csv(index_path,header=None)
    ori_index = df.pop(1).values.tolist()
    ori_index.sort(key=lambda x: x.split("_")[1])
    out_index = []
    tmp = []
    kw = ori_index[0].split("_")[1]
    for item in ori_index:
        if item.split("_")[1] == kw:
            tmp.append(item)
        else:
            random.shuffle(tmp)
            out_index+=tmp
            tmp=[]
            kw = item.split("_")[1]
            tmp.append(item)
    random.shuffle(tmp)
    out_index+=tmp
    df = pd.read_csv(index_path, header=None).values.tolist()
    df.sort(key=lambda x: x[1].split("_")[1])
    df = pd.DataFrame(df)
    df.insert(2,column="2",value=out_index)
    df.to_csv(os.path.join(os.path.dirname(index_path),"index_cond.csv"),header=None,index=None)

class Vmd:
    def __init__(self):
        pass

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


def prepareTestIndex(image_dir):
    '''
    read image_dir, make test_id.txt
    '''
    str_in = ""
    for file in os.listdir(image_dir):
        str_in += file
        str_in += "\n"
    with open(os.path.join(image_dir,"test_id.txt"),"a+") as f:
        f.write(str_in)

def main():
    start_t = time.time()
    base_dir = r"D:\download_cache\PMXmodel"
    frame_base_dir = r"D:\download_cache\VAEmodel"
    input_dir = os.path.join(base_dir, "VIDEOclips")
    vmd_dir = os.path.join(base_dir, "VMDfile")  # originally, output videos are in
    map_dir = os.path.join(base_dir, "OUTPUTclips_done")
    index_out = os.path.join(frame_base_dir, "index.csv")

    # transfer_files(vmd_dir,map_dir,suffix=".avi")
    extractFrames(input_dir, map_dir, frame_base_dir, index_out, interval=5)
    print(time.time()-start_t)

if __name__=="__main__":
    # add_condition(r"D:\download_cache\VAEmodel\index.csv")
    # prepareTestIndex(r"D:\download_cache\VAEmodel\MapFrame")
    main()
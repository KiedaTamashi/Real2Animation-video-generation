import numpy as np
import os
import json
import pandas as pd
import cv2
import copy
import math
from shutil import copyfile
import random

#E:\PycharmProject\data
def show_points(imgfile,kps,confth=0.1):
    '''
    :param img: origin img
    :param kps: [(x,y,confidence),()..]
    :return: draw points on the ori img
    '''
    df_kps = pd.DataFrame(kps)
    # from here to get image.
    def render_joints(cvmat, joints, conf_th=0.2):
        for _joint in joints:
            _x, _y, _conf = _joint
            if _conf > conf_th:
                cv2.circle(cvmat, center=(int(_x), int(_y)), color=(255, 0, 0), radius=7, thickness=2)
        return cvmat
    cvmat = render_joints(cv2.imread(imgfile), kps, confth)
    cv2.imwrite("./outcomes/out_{}".format(imgfile.split("/"[-1])), cvmat)
    return cvmat

def csvRebuild(csv_file,out_file="./data2preprocess.csv"):
    '''
    :param csv_file: original data from scrapy
    :param out_file: format: name,width,height
    '''
    df = pd.read_csv(csv_file,header=None)
    rebuild = list()
    i = 0
    while i<len(df):
        line1 = df.iloc[i]
        line2 = df.iloc[i+1]
        line = list(pd.concat([line1,pd.Series(line2[2])]))
        rebuild.append(line)
        i+=2
    df = pd.DataFrame(rebuild)
    df = df.drop(columns=[1])
    df.to_csv(out_file,header=None,index=None)

def json_dict(jsonfile):
    img_list = list()
    img_size = list()
    with open(jsonfile, encoding="utf-8") as f:
        # after load. it is a list contains items represnted by dict
        contents = json.load(f)
        for content in contents:
            img_list.append(content["file_name"].split("/")[-1])
            img_size.append(tuple([content["width"], content["height"]]))
    return img_list, img_size

def load_json(jsonfile):
    #return list contains dicts
    with open(jsonfile, encoding="utf-8") as f:
        contents = json.load(f)
    return contents

def write_json(jsonfile,data):
    with open(jsonfile,mode='w',encoding="utf-8") as f:
        json.dump(data,f)

def resize_imgs(csv_file,data_dir="E:/PycharmProject/data/"):
    df = pd.read_csv(csv_file,header=None)
    for idx,item in df.iterrows():
        url, width, height = item
        img = cv2.imread(os.path.join(data_dir+"images_raw",url))
        ori_width, ori_height = img.shape[1],img.shape[0]
        print(url)
        if abs(float(ori_height)/float(height)-float(ori_width)/float(width)) > 0.5:
            print(url)
        cv2.imwrite(os.path.join(data_dir+"images",url),cv2.resize(img,(width,height)))

def gen_new_json(jsonfile,output_file="./all_data.json",csv_file="./data2preprocess.csv"):
    df_csv = pd.read_csv(csv_file, header=None)
    contents = load_json(jsonfile)
    new_contents = []
    id = 0
    for item in contents:
        if item['file_name'].split("/")[-1] == df_csv.iloc[id][0]:
            id+=1
            new_contents.append(item)
    print(len(new_contents))
    with open(output_file,mode='w',encoding="utf-8") as f:
        json.dump(new_contents,f)

def reduce_points(jsonfile,output_file="./all_data.json"):
    pass

def data_augmentation_file(json_in, json_out, ops = ("fliph","flipv","rot_180","noise_0.02","blur_1.0")):
    '''
    we plan to give the four types augmentation. flap-H/V, rotate 180, noise,blur...
    :param jsonfile: data-annotation
    :param output_dir: target file for augmented data
    :return:
    '''
    data_ori = load_json(json_in)
    data_out = list()
    for op in ops:
        for data in data_ori:
            tmp = copy.deepcopy(data)
            if op == "noise_0.02" or op == "blur_1.0":
                tmp["file_name"] = tmp["file_name"][:-4]+"__"+"".join(op.split("_"))+tmp["file_name"][-4:]
            if op == "fliph":
                tmp["file_name"] = tmp["file_name"][:-4]+"__" + op + tmp["file_name"][-4:]
                w = tmp['width']
                for _,value in tmp["points"].items():
                    value[0] = w - value[0]
            if op == "flipv":
                tmp["file_name"] = tmp["file_name"][:-4]+"__" + op + tmp["file_name"][-4:]
                h = tmp['height']
                for _,value in tmp["points"].items():
                    value[1] = h - value[1]
            if op == "rot_180":
                tmp["file_name"] = tmp["file_name"][:-4]+"__" + "".join(op.split("_")) + tmp["file_name"][-4:]
                w = tmp['width']
                h = tmp['height']
                for _,value in tmp["points"].items():
                    value[0] = w - value[0]
                    value[1] = h - value[1] #rotate 90 is width swaped with height
            data_out.append(tmp)
    data_out +=data_ori
    write_json(json_out,data_out)

def Json2LipCSV(jsonfile,out_name="anime"):
    data_ori = load_json(jsonfile)
    train_data = list()
    val_data = list()
    train_len = math.floor(len(data_ori)*0.9)
    # val_len = len(data_ori)-train_len
    random.shuffle(data_ori)
    for idx,data in enumerate(data_ori):
        tmp = list()
        tmp.append(data["file_name"].split("/")[-1])
        for key,value in data["points"].items():
            tmp+=[int(x) for x in value]
            tmp.append(0)
        if idx<train_len:
            train_data.append(tmp)
        else:
            val_data.append(tmp)
    pd.DataFrame(train_data).to_csv(out_name+"_train_set.csv",header=None,index=None)
    pd.DataFrame(val_data).to_csv(out_name+"_val_set.csv",header=None,index=None)

def divideDataset(csvfile,source_dir,target_dir):
    df = pd.read_csv(csvfile,header=None)
    length = len(df)
    for i in range(length):
        file_name = df.iloc[i][0]
        copyfile(os.path.join(source_dir,file_name),os.path.join(target_dir,file_name))

# Json2LipCSV("./all_data_aug.json")
# divideDataset("./anime_val_set.csv",r"E:\PycharmProject\data\images_augment",r"E:\PycharmProject\gccpm-look-into-person-cvpr19.pytorch\data_anime\TrainVal_images\val_images")
divideDataset("./anime_train_set.csv",r"E:\PycharmProject\data\images_augment",r"E:\PycharmProject\gccpm-look-into-person-cvpr19.pytorch\data_anime\TrainVal_images\train_images")
# data_augmentation_file(r"D:\work\pycharmproject\Real2Animation-video-generation\pose_estimate\all_data.json",
#                        r"D:\work\pycharmproject\Real2Animation-video-generation\pose_estimate\all_data_aug.json")
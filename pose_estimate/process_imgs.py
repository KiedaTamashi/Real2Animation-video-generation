import numpy as np
import os
import json
import pandas as pd
import cv2

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

csvRebuild("./data.csv")
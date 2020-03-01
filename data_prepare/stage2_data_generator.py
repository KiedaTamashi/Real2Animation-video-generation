import os
import random
import numpy as np
import shutil
import cv2
from data_prepare.video2vmd import *
from data_prepare.smooth_pose import smooth_json_pose
import pandas as pd
import ffmpeg
import datetime
from train_logger import get_logger
import time

def get_time(second):
    # import seconds of video, output typical time.
    m, s = divmod(int(second), 60)
    h, m = divmod(m, 60)
    return datetime.datetime.strptime(str("%d:%02d:%02d" % (h, m, s)), '%H:%M:%S')

def deal_video(video_dir, save_dir, start_time=15, end_time=12):
    for video in os.listdir(video_dir):
        video_path = os.path.join(video_dir,video)
        video_time = float(ffmpeg.probe(video_path)['format']['duration'])
        during_time = video_time-start_time-end_time
        # video_detail_time = get_time(video_time)

        cut_start_time = str(get_time(start_time)).split(" ")[1]
        during_time = str(get_time(during_time)).split(" ")[1]
        video_out_path = os.path.join(save_dir,video)
        cut_video(video_path, video_out_path, cut_start_time, during_time)

def cut_video(video_path, out_video_path, start_time, during_time):
    text = 'ffmpeg -ss \"%s\" -t \"%s\" -i \"%s\"  -c:v libx264 -c:a aac -strict experimental -b:a 98k \"%s\" -y' % (
        start_time, during_time, video_path, out_video_path)
    res = os.system(text)
    if res != 0:
        print("error! "+video_path)

def copy_videos(video_dir,keyword="dance_"):
    cnt = 0
    for item in os.listdir(video_dir):
        oripath = os.path.join(video_dir,item)
        targetpath = os.path.join(video_dir,keyword+str(cnt)+".mp4")
        shutil.copy(oripath,targetpath)
        cnt+=1

def get_pair_csv(video_dir,csv_dir):
    #video_name,model_name
    output = list()
    cnt=0
    for videoname in os.listdir(video_dir):
        video_params = ffmpeg.probe(os.path.join(video_dir,videoname))['streams'][0]
        nb_frames = int(video_params['nb_frames'])
        fr = round(eval(video_params['avg_frame_rate']),2)
        for csvname in os.listdir(csv_dir):
            output.append([videoname[:-4],csvname[:-4],fr,nb_frames])
        cnt+=1
        print(cnt)
    df = pd.DataFrame(output)
    df.to_csv(r"D:\download_cache\PMXmodel\index.csv",header=None,index=None)



def generate_vmd(data_dir,index_csv, start_num,end_num):
    '''
    read index to get pairs of video and model.Then use single video2vmd to generate
    The range is decided by start_num and end_num
    !!! not include end_num
    '''
    #5913 total
    #csv: video_name, model_name(np format), frame_rate, length(counted by frame)
    # maybe use all mp4 for easier index
    VMDlogger = get_logger("./")
    df = pd.read_csv(index_csv,header=None)
    total_num = end_num-start_num+1
    print("process {} samples, from {} to {}.".format(total_num,start_num,end_num-1))
    for item in range(start_num*73,end_num*73,73):
        json_out_dir = "D:/download_cache/json_out"
        if os.path.exists(json_out_dir):
            shutil.rmtree(json_out_dir)
        time.sleep(5)
        os.mkdir(json_out_dir)

        json3d_folder = json_out_dir + "_3d"
        if os.path.exists(json3d_folder):
            shutil.rmtree(json3d_folder)

        video_name, _, _, _ = df.iloc[item]
        video_path = os.path.join(data_dir,"VIDEOfile",video_name+".mp4")
        video2keypoints(video_path, json_out_dir)

        #smooth the kps. Using window size ~= 1/2*frame_rate
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        wid_size = int(fps/2)
        if wid_size>21:
            wid_size=21
        else:
            wid_size = wid_size-1 if (wid_size & 1) == 0 else wid_size
        smooth_json_pose(json_out_dir,window_length=wid_size,polyorder=3,threshold=0.15)

        kpsTo3D(json_out_dir)
        video2depth(video_path, json3d_folder)
        cnt=0
        VMDlogger.debug(f"===={video_name} starts!=====")

        for idx in range(item,item+73):
            _, model_name, _, _ = df.iloc[idx]
            csv_path = os.path.join(data_dir,"CSVfile",model_name+".csv")
            output_path = os.path.join(data_dir,"VMDfile",video_name+"_"+model_name+".vmd")
            try:
                json3DtoVMD(json3d_folder, csv_path, output_path)
                cnt+=1
                VMDlogger.debug(cnt)
            except:
                VMDlogger.debug(model_name)




        # video2VMD_single(video_path,json_out_dir,csv_path,output_path)

if __name__=="__main__":
    # copy_videos(r"D:\download_cache\dance_video")
    # deal_video(r"D:\download_cache\dance_video",r"D:\download_cache\PMXmodel\VIDEOfile")
    # get_pair_csv(r"D:\download_cache\PMXmodel\VIDEOfile","D:\download_cache\PMXmodel\CSVfile")
    # video24 5501 frame can't detect sth.
    generate_vmd(r"D:\download_cache\PMXmodel",r"D:\download_cache\PMXmodel\index.csv",start_num=43,end_num=46)
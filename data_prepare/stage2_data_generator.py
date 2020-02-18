import os
print(os.environ['COMSPEC'])
import random
import numpy as np
import shutil
import cv2
from data_prepare.video2vmd import video2VMD_single
import pandas as pd
import ffmpeg
import datetime
import subprocess

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

def rename_videos(video_dir):
    cnt = 0
    for item in os.listdir(video_dir):
        oripath = os.path.join(video_dir,item)
        targetpath = os.path.join(video_dir,"dance_"+str(cnt)+".mp4")
        shutil.copy(oripath,targetpath)
        cnt+=1

# rename_videos(r"D:\download_cache\dance_video")
deal_video(r"D:\download_cache\dance_video",r"D:\download_cache\PMXmodel\VIDEOfile")

def get_pair_csv(video_dir,csv_dir):
    #video_name,model_name
    videos = list()
    csvs = list()
    for filename in os.listdir(video_dir):
        videos.append(filename)
    for csvname in os.listdir(csv_dir):
        csvs.append(csvname)

    

def generate_vmd(data_dir,index_csv, start_num,end_num):
    '''
    read index to get pairs of video and model.Then use single video2vmd to generate
    The range is decided by start_num and end_num
    !!! not include end_num
    '''
    #csv: video_name(+mp4), model_name(np format), frame_rate, length(counted by frame)
    # maybe use all mp4 for easier index
    df = pd.read_csv(index_csv,header=None)
    total_num = end_num-start_num+1
    print("process {} samples, from {} to {}.".format(total_num,start_num,end_num-1))
    for idx in range(start_num,end_num):
        video_name, model_name, _, _ = df.iloc[idx]
        video_path = os.path.join(data_dir,"VIDEOfile",video_name)
        csv_path = os.path.join(data_dir,"CSVfile",model_name+".csv")
        output_path = os.path.join(data_dir,"VMDfile",video_name[:-4]+"_"+model_name+".vmd")
        video2VMD_single(video_path,"./json_out",csv_path,output_path)




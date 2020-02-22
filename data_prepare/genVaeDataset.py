import os
import pandas as pd
import shutil
import cv2
import time

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
    ori_videoNames.sort(key=lambda x:int(x.split("_")[-1][:-4]))
    file_pair_list = []

    for oriVideoName in ori_videoNames:
        video_1 = cv2.VideoCapture(os.path.join(ori_dir,oriVideoName))
        for mapVideoName in map_videoNames:
            if not mapVideoName.startswith(oriVideoName.split(".")[0]):
                continue
            else:
                video_2 = cv2.VideoCapture(os.path.join(map_dir,mapVideoName))
                frame_number = 0
                while True:
                    ret1, img1 = video_1.read()
                    ret2, img2 = video_2.read()
                    if not ret1 or not ret2: break

                    if frame_number % interval == 0:
                        print(mapVideoName + str(frame_number))
                        x_name = "x_"+str(mapVideoName.split(".")[0])
                        y_name = "y_"+str(mapVideoName.split(".")[0]) #TODO need check when actually use.
                        cv2.imwrite(os.path.join(x_frame_dir,x_name + "_" + str(frame_number) + '.jpg'), img1)
                        cv2.imwrite(os.path.join(y_frame_dir,y_name + "_" + str(frame_number) + '.jpg'), img2)
                        file_pair_list.append([x_name + "_" + str(frame_number),y_name + "_" + str(frame_number)])
                    frame_number += 1
                video_2.release()
        video_1.release()
    df = pd.DataFrame(file_pair_list)
    df.to_csv(index_path,index=None,header=None)

def add_condition(index_path):
    '''
    input file formated as [oriFrameName,mapFrameName]
    output file formated as [oriFrameName,mapFrameName,conditionFrameName]
    '''
    pass

def prepareTestIndex(image_dir):
    '''
    read image_dir, make test_id.txt
    '''
    for file in os.listdir(image_dir):
        with open(os.path.join(image_dir,"test_id.txt"),"w+") as f:
            f.write(file+"\n")


if __name__=="__main__":
    base_dir = r"D:\download_cache\PMXmodel"
    frame_base_dir = r"D:\download_cache\VAEmodel"
    input_dir = os.path.join(base_dir,"VIDEOfile")
    vmd_dir = os.path.join(base_dir,"VMDfile") # originally, output videos are in
    map_dir = os.path.join(base_dir,"OUTPUTfile")
    index_out = os.path.join(frame_base_dir,"index.csv")

    transfer_files(vmd_dir,map_dir,suffix=".avi")
    extractFrames(input_dir,map_dir,frame_base_dir,index_out,interval=30)
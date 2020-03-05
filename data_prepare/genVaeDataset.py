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

    for mapVideoName in map_videoNames:
        video_2 = cv2.VideoCapture(os.path.join(map_dir,mapVideoName))
        for oriVideoName in ori_videoNames:
            if not oriVideoName.startswith("_".join(oriVideoName.split("_")[0:3])):
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
    ori_index.sort(key=lambda x:int(x.split("_")[1]))
    out_index = []
    tmp = []
    kw = out_index[0].split("_")[1]
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
    df = pd.read_csv(index_path, header=None)
    df.insert(2,column="2",value=out_index)
    df.to_csv(os.path.join(index_path.split("\\")[:-1],"index_new.csv"),header=None,index=None)




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
    input_dir = os.path.join(base_dir,"VIDEOclips")
    vmd_dir = os.path.join(base_dir,"VMDfile") # originally, output videos are in
    map_dir = os.path.join(base_dir,"OUTPUTclips")
    index_out = os.path.join(frame_base_dir,"index.csv")

    # transfer_files(vmd_dir,map_dir,suffix=".avi")
    extractFrames(input_dir,map_dir,frame_base_dir,index_out,interval=6)
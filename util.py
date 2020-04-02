import torch
import logging
import os
from logging import handlers
# from moviepy.editor import *
import pandas as pd
import cv2
import sys
import numpy as np

def generate_one_mone():
    one = torch.FloatTensor([1])
    mone = one * -1

def get_logger(LOG_ROOT, level=logging.DEBUG, back_count=0,cmd_stream=False):
    """
    :brief  日志记录
    :param log_filename:
    :param level:
    :param back_count:
    :return: logger
    """
    logger = logging.getLogger("logger.log")
    logger.setLevel(level)
    log_path = os.path.join(LOG_ROOT, "logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_file_path = os.path.join(log_path, "logger.log")
    # log输出格式
    formatter = logging.Formatter('%(asctime)s:\n %(message)s')
    # 输出到文件
    fh = logging.handlers.TimedRotatingFileHandler(
        filename=log_file_path,
        backupCount=back_count,
        encoding='utf-8')
    fh.setLevel(level)
    # 添加到logger对象里
    logger.addHandler(fh)
    # 输出到控制台
    if cmd_stream:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        logger.addHandler(ch)
    return logger

def genClipCsvFile(video_name,clips_array):
    '''
    :param video_name: string, The input clip video name(no suffix)
    :param clips_array: numpy format. contain the start and end point of each clip for the name video
    :return:
    '''
    out_dir = r"D:\download_cache\PMXmodel\CLIPindex"

    df = pd.DataFrame(clips_array)
    df.to_csv(os.path.join(out_dir,video_name+".csv"),index=None,header=None)
    return video_name

def ClipOriVideo():
    video_dir = r"D:\download_cache\PMXmodel\VIDEOfile"
    index_dir = r"D:\download_cache\PMXmodel\finishClip"
    output_dir = r"D:\download_cache\PMXmodel\VIDEOclips"
    for video_name in os.listdir(index_dir):
        video_name = video_name.split(".")[0]
        video_path = os.path.join(video_dir, video_name + ".mp4")
        clip_index = pd.read_csv(os.path.join(index_dir,video_name+".csv"),header=None)
        for num,clip in enumerate(clip_index.values.tolist()):
            # [start frame, end frame]
            start_f, end_f = clip # e.g. 0, 123

            videoCapture = cv2.VideoCapture(video_path)
            fps = videoCapture.get(cv2.CAP_PROP_FPS)
            width = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = (int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            size = (width, height)  # 保存视频的大小

            videoWriter = cv2.VideoWriter(os.path.join(output_dir,video_name+"_"+str(num)+".avi"), cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
            i = 0
            while True:
                success, frame = videoCapture.read()
                if success:
                    if i < int(start_f):
                        i += 1
                        continue
                    elif (i >= int(start_f) and i <= int(end_f)):
                        videoWriter.write(frame)
                        i += 1
                    else:
                        break
                else:
                    print("error")
                    break
            videoCapture.release()



def combineTwoVideo_height(v1_path=r'D:\work\OpenMMD1.0\examples\ori_pose.avi',v2_path=r'D:\work\OpenMMD1.0\examples\smooth_pose13_3.avi',combine_path=r'D:\work\OpenMMD1.0\examples\combine_pose.avi'):
    import cv2
    import numpy as np

    videoLeftUp = cv2.VideoCapture(v1_path)
    videoLeftDown = cv2.VideoCapture(v2_path)
    # videoRightUp = cv2.VideoCapture('./res/2_003_015.mp4')
    # videoRightDown = cv2.VideoCapture('./res/2_003_016.mp4')

    fps = videoLeftUp.get(cv2.CAP_PROP_FPS)

    width = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    videoWriter = cv2.VideoWriter(combine_path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, (width, height*2))

    successLeftUp, frameLeftUp = videoLeftUp.read()
    successLeftDown, frameLeftDown = videoLeftDown.read()
    # successRightUp, frameRightUp = videoRightUp.read()
    # successRightDown, frameRightDown = videoRightDown.read()

    while successLeftUp and successLeftDown:
        frameLeftUp = cv2.resize(frameLeftUp, (width, height), interpolation=cv2.INTER_CUBIC)
        frameLeftDown = cv2.resize(frameLeftDown, (width, height), interpolation=cv2.INTER_CUBIC)

        frame = np.vstack((frameLeftUp, frameLeftDown))
        # frame = np.hstack(frameLeftDown,frameLeftUp)

        videoWriter.write(frame)
        successLeftUp, frameLeftUp = videoLeftUp.read()
        successLeftDown, frameLeftDown = videoLeftDown.read()

    videoWriter.release()
    videoLeftUp.release()
    videoLeftDown.release()

def combineTwoVideo_width(v1_path=r'D:\work\OpenMMD1.0\examples\ori_pose.avi',v2_path=r'D:\work\OpenMMD1.0\examples\smooth_pose13_3.avi',combine_path=r'D:\work\OpenMMD1.0\examples\combine_pose.avi'):
    import cv2
    import numpy as np

    videoLeftUp = cv2.VideoCapture(v1_path)
    videoLeftDown = cv2.VideoCapture(v2_path)
    # videoRightUp = cv2.VideoCapture('./res/2_003_015.mp4')
    # videoRightDown = cv2.VideoCapture('./res/2_003_016.mp4')

    fps = videoLeftUp.get(cv2.CAP_PROP_FPS)

    width = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    videoWriter = cv2.VideoWriter(combine_path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, (width*2, height))

    successLeftUp, frameLeftUp = videoLeftUp.read()
    successLeftDown, frameLeftDown = videoLeftDown.read()
    # successRightUp, frameRightUp = videoRightUp.read()
    # successRightDown, frameRightDown = videoRightDown.read()
    frame_num = 0
    while successLeftUp and successLeftDown:
        frameLeftUp = cv2.resize(frameLeftUp, (width, height), interpolation=cv2.INTER_CUBIC)
        frameLeftDown = cv2.resize(frameLeftDown, (width, height), interpolation=cv2.INTER_CUBIC)

        frame = np.hstack((frameLeftDown,frameLeftUp))
        videoWriter.write(frame)
        successLeftUp, frameLeftUp = videoLeftUp.read()
        successLeftDown, frameLeftDown = videoLeftDown.read()
        if frame_num == 600:
            break
        frame_num+=1
        print(frame_num)
    videoWriter.release()
    videoLeftUp.release()
    videoLeftDown.release()

def vgg_preprocess(x):
    x = 255.0 * (x + 1.0)/2.0

    x[:,:,:,0] -= 103.939
    x[:,:,:,1] -= 116.779
    x[:,:,:,2] -= 123.68

    return x

def printProgress(step,test,train_loss,time=None):
    s = str(step) + "," + str(test)

    if(isinstance(train_loss,list) or isinstance(train_loss,np.ndarray)):
        for i in range(len(train_loss)):
            s += "," + str(train_loss[i])
    else:
        s += "," + str(train_loss)

    if(time is not None):
        s += "," + str(time)

    print(s)
    sys.stdout.flush()

def smooth(csv_path,weight=0.85):
    data = pd.read_csv(csv_path,header=None)
    x = list(data.pop(0))
    scalar = list(data.pop(1))
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    import matplotlib.pyplot as plt
    plt.plot(x, smoothed)
    plt.show()
    # save = pd.DataFrame({'Step':data['Step'].values,'Value':smoothed})
    # save.to_csv('smooth_'+csv_path)

def combine_record(folder):
    files = os.listdir(folder)
    files.sort(key=lambda x: int(x.split(".")[0]))
    for idx,file in enumerate(files):
        if idx==0:
            a = pd.read_csv(os.path.join(folder,file),header=None)
        else:
            a = pd.concat([a,pd.read_csv(os.path.join(folder,file),header=None)])
    a.to_csv("mse8000.csv",header=None,index=None)

def mse(img1_path,img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = (img1 / 255.0 - 0.5) * 2.0
    img2 = (img2 / 255.0 - 0.5) * 2.0
    return np.mean(np.square(img1 - img2))

def main():
    left_base = r"D:\download_cache\PMXmodel\VIDEOfile"
    right_base = r"D:\download_cache\PMXmodel\OUTPUTclips"
    right_vs = os.listdir(right_base)
    right_vs.sort(key=lambda x:int(x[:-4]))
    left_vs = os.listdir(left_base)
    left_vs.sort(key=lambda x: int(x.split("_")[-1][:-4]))
    for idx in range(len(left_vs)):
        print(idx)
        if idx<36:
            continue
        else:
            left = left_vs[idx]
            right = right_vs[idx]
            out = "c_"+right[:-4]+".avi"
            combineTwoVideo_height(os.path.join(left_base,left),os.path.join(right_base,right),os.path.join(right_base,out))

def get_file_encoding(file_path):

    try:
        f = open(file_path, "rb")
        fbytes = f.read()
        f.close()
    except BaseException:
        raise Exception("unknown encoding!")

    codelst = ('utf_8', 'shift-jis')

    for encoding in codelst:
        try:
            fstr = fbytes.decode(encoding)  # bytes文字列から指定文字コードの文字列に変換
            fstr = fstr.encode('utf-8')  # uft-8文字列に変換
            return encoding
        except BaseException:
            pass

    raise Exception("unknown encoding!")


if __name__=="__main__":
    # combineTwoVideo_width("D:\download_cache\PMXmodel\VIDEOclips\dance_10_8.avi","D:\download_cache\PMXmodel\OUTPUTclips\dance_10_8_GTGoku.avi","D:\download_cache\PMXmodel\compare.avi")
    clips80 = [[190,262],[390,433],[530,590],[650,716],[777,830],[945,1050],[1085,1230],[1380,4135],[1510,1570],[1640,1730],[2141,2185],[4195,4320],[4860,4925]
               ]
    genClipCsvFile("dance_20",clips80)
    # main()
    # ClipOriVideo()
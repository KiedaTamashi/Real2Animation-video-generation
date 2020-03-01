import torch
import logging
import os
from logging import handlers
from moviepy.editor import *
import pandas as pd
import cv2

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

def ClipOriVideo(video_name):
    video_dir = r"D:\download_cache\PMXmodel\VIDEOfile"
    index_dir = r"D:\download_cache\PMXmodel\CLIPIndex"
    output_dir = r"D:\download_cache\PMXmodel\VIDEOclips"

    video_path = os.path.join(video_dir, video_name + ".mp4")
    clip_index = pd.read_csv(index_dir,video_name+".csv",header=None).values.tolist()
    for num,clip in enumerate(clip_index):
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

def main():
    left_base = r"D:\download_cache\PMXmodel\VIDEOfile"
    right_base = r"D:\download_cache\PMXmodel\OUTPUTclips"
    right_vs = os.listdir(right_base)
    right_vs.sort(key=lambda x:int(x[:-4]))
    left_vs = os.listdir(left_base)
    left_vs.sort(key=lambda x: int(x.split("_")[-1][:-4]))
    for idx in range(len(left_base)):
        left = left_vs[idx]
        right = right_vs[idx]
        out = "c_"+right[:-4]+".avi"
        combineTwoVideo_height(os.path.join(left_base,left),os.path.join(right_base,right),os.path.join(right_base,out))



if __name__=="__main__":
    # combineTwoVideo_width("D:\download_cache\mmd_combine_pose.avi","D:\download_cache\original.avi","D:\download_cache\compare.avi")
    # clips10 = [[41,82],[235,360],[880,950],[1015,1112],[1302,1465],[1733,1860],
    #          [1950,2005],[2042,2150],[2290,2447],[2670,2805]
    #          ]
    # clips01 = [[],[],[]
    #            ]
    # genClipCsvFile("dance_10",clips10)
    main()
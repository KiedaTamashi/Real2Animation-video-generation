import torch
import logging
import os
from logging import handlers
from moviepy.editor import *

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

    while successLeftUp and successLeftDown:
        frameLeftUp = cv2.resize(frameLeftUp, (width, height), interpolation=cv2.INTER_CUBIC)
        frameLeftDown = cv2.resize(frameLeftDown, (width, height), interpolation=cv2.INTER_CUBIC)

        frame = np.hstack((frameLeftDown,frameLeftUp))

        videoWriter.write(frame)
        successLeftUp, frameLeftUp = videoLeftUp.read()
        successLeftDown, frameLeftDown = videoLeftDown.read()

    videoWriter.release()
    videoLeftUp.release()
    videoLeftDown.release()

if __name__=="__main__":
    combineTwoVideo_width("D:\download_cache\mmd_combine_pose.avi","D:\download_cache\original.avi","D:\download_cache\compare.avi")
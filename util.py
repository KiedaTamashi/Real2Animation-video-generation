import torch
import logging
import os
from logging import handlers
# from moviepy.editor import *
import pandas as pd
import cv2,json
import sys
import numpy as np
import shutil
import random,time

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

# def main():
#     left_base = r"D:\download_cache\PMXmodel\VIDEOfile"
#     right_base = r"D:\download_cache\PMXmodel\OUTPUTclips"
#     right_vs = os.listdir(right_base)
#     right_vs.sort(key=lambda x:int(x[:-4]))
#     left_vs = os.listdir(left_base)
#     left_vs.sort(key=lambda x: int(x.split("_")[-1][:-4]))
#     for idx in range(len(left_vs)):
#         print(idx)
#         if idx<36:
#             continue
#         else:
#             left = left_vs[idx]
#             right = right_vs[idx]
#             out = "c_"+right[:-4]+".avi"
#             combineTwoVideo_height(os.path.join(left_base,left),os.path.join(right_base,right),os.path.join(right_base,out))

def get_file_encoding(file_path):

    try:
        f = open(file_path, "rb")
        fbytes = f.read()
        f.close()
    except BaseException:
        f = open(file_path, "rb",encoding='shift-jis')
        fbytes = f.read()
        f.close()
        # raise Exception("unknown encoding!")

    codelst = ('utf_8', 'shift-jis')

    for encoding in codelst:
        try:
            fstr = fbytes.decode(encoding)  # bytes文字列から指定文字コードの文字列に変換
            fstr = fstr.encode('utf-8')  # uft-8文字列に変換
            return encoding
        except BaseException:
            pass

    raise Exception("unknown encoding!")

def video2frames(video_path,out_dir):
    video_base_name = os.path.basename(video_path)[:-4]
    v = cv2.VideoCapture(video_path)
    flag, frame = v.read()
    frame_num = 0
    while flag:
        cv2.imwrite(os.path.join(out_dir,video_base_name+"_"+str(frame_num)+".jpg"),frame)
        frame_num += 1
        flag, frame = v.read()

def json2npy(video_name,json_dir,npy_dir):
    json_files = os.listdir(json_dir)
    json_files.sort(key=lambda x:int(x.split("_")[2]))
    cnt = 0
    for json_file in json_files:
        with open(os.path.join(json_dir,json_file), encoding="utf-8") as f:
            content = json.load(f)['people'][0]['pose_keypoints_2d']
        tmp = []
        for idx in range(0, len(content)-6, 3):
            x, y, _ = content[idx], content[idx + 1], content[idx + 2]
            tmp.append([x, y])
        np.save(os.path.join(npy_dir,video_name+"_"+str(cnt)+".npy"),np.asarray(tmp))
        cnt+=1

def prepareForPoseTransfer(test_dir,testK_dir,refernce_image):
    name_A = os.path.basename(refernce_image)
    shutil.copy(refernce_image,os.path.join(test_dir,name_A))
    from data_prepare.generate_pose_map_anime import compute_pose_single
    # compute_pose_single(os.path.join(test_dir[:-4]+"tmpK",name_A+".npy"),testK_dir)
    shutil.copy(r"D:\work\pycharmproject\Real2Animation-video-generation\demo\testK\animeImage.jpg.npy",os.path.join(testK_dir,name_A+".npy"))
    files = os.listdir(test_dir)[1:]
    files.sort(key= lambda x:int(x.split("_")[-1][:-4]))
    tmp = []
    for file in files:
        tmp.append([name_A,file])
    df = pd.DataFrame(tmp,columns=["from","to"])
    df.to_csv(os.path.join(os.path.dirname(test_dir),"anime-pairs-test.csv"),index=None)



def kps_Normalize_single(img_r,img_r_new, kps_r, kps_a,output_dir,reference_dir,vis=None,real_bone_num=None,video_size=(1280,720)):
    # kps_r = r"D:\download_cache\anime_data2\trainK", kps_a = r"D:\download_cache\anime_data2\tmpK",
    # output_dir = "D:/download_cache/anime_data2/normK/", vis = None, real_bone = None

    # default to "D:\\download_cache\\anime_data\\vis_img\\"
    # img_r_new = "\\".join(img_r.split("\\")[:-1])+"\\testN"
    kpss = os.listdir(kps_r)
    kpss.sort(key=lambda x:int(x.split("_")[2][:-4]))
    img_nums = len(kpss)
    # max_vis = vis_num
    if real_bone_num is None:
        r_bone = np.load(os.path.join(kps_r,kpss[0]))
    else:
        r_bone = np.load(os.path.join(kps_r,kpss[real_bone_num]))
    bones =None
    for idx in range(img_nums):
        real_kps = np.load(os.path.join(kps_r, kpss[idx]))
        #TODO hard-code debug
        for item in real_kps:
            item[0] = item[0]/video_size[0]*1920
            item[1] = item[1]/video_size[1]*1080

        anime_kps = np.load(kps_a)
        # 0-center 1-groove 2-r_leg_IK 3-l_leg_IK 4-upperbody 5-upperbody2 6-neck 7-head 8-r_eye 9-l_eye 10-r_shoulder
        # 11-arm 12-r_elbow 13-r_hand 14-l_shoulder 15-l_arm 16-l_elbow 17-l_hand 18-lowerbody 19-r_leg 20-r_knee 21-r_ankle 22-l_leg
        # 23-l_knee 24-l_ankle   (23,2)
        # TODO bad bone now. #####  not connect to anime pose estimation now ########
        from data_prepare.genWarpDataset import getBoneFromCsv, Map_3Dpoints_2D,eucliDist,kps_Normalize
        bones = getBoneFromCsv(os.path.join("D:\download_cache\PMXmodel\CSVfile", "RyukoMatoi.csv"))
        joints = []
        for i, _joint in enumerate(bones[:-2]):
            _, _x, _y, _z = _joint
            joints.append(np.asarray([_x, _y, _z]))
        bones = Map_3Dpoints_2D(joints, vis=False)
        anime_length = [eucliDist(bones[19],bones[21])*0.65,eucliDist(bones[11],bones[13])*0.85,eucliDist((bones[19]+bones[22])/2.0,bones[7])*0.8,eucliDist(bones[11],bones[15])*0.88]
        # TODO bad bone now. #####  not connect to anime pose estimation now ########

        real_length = [max(eucliDist(r_bone[8],r_bone[9])+eucliDist(r_bone[9],r_bone[10]),eucliDist(r_bone[11],r_bone[12])+eucliDist(r_bone[12],r_bone[13])),
                       max(eucliDist(r_bone[2], r_bone[3]) + eucliDist(r_bone[3], r_bone[4]), eucliDist(r_bone[5], r_bone[6]) + eucliDist(r_bone[6], r_bone[7])),
                       eucliDist((r_bone[8]+r_bone[9])/2.0, r_bone[1]),
                       eucliDist(r_bone[2], r_bone[5])]
        proportation = []
        for x,y in zip(anime_length,real_length):
            proportation.append(x/y)
        modified_kps = kps_Normalize(real_kps,anime_kps,ref=proportation,scale_level=0.9,special_test=True)

        img_size = (192,256)
        img_ori = cv2.imread(os.path.join(img_r,kpss[idx][:-4]+".jpg"))
        img_ori = cv2.resize(img_ori, (1920, 1080), interpolation=cv2.INTER_CUBIC)

        center_point = modified_kps[1,:]
        center_x, center_y = center_point
        if int(center_x) - 405 < 0:
            crop_l = 0
            crop_r = 810
        elif int(center_x) + 405 > 1920:
            crop_l = 1920-810
            crop_r = 1920
        else:
            crop_l = int(center_x) - 405
            crop_r = int(center_x) + 405
        cropped = img_ori[0:1080,crop_l:crop_r,:]  # 裁剪坐标为[y0:y1, x0:x1]
        normalized = cv2.resize(cropped, img_size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(img_r_new,kpss[idx][:-4]+".jpg"), normalized)

        center_point = modified_kps[1, :]
        center_x, center_y = center_point
        if int(center_x) - 405 < 0:
            crop_l = 0
            crop_r = 810
        elif int(center_x) + 405 > 1920:
            crop_l = 1920 - 810
            crop_r = 1920
        else:
            crop_l = int(center_x) - 405
            crop_r = int(center_x) + 405

        for kps in modified_kps:
            kps[0] = kps[0] - crop_l
            kps[0] = kps[0] / 810 * img_size[0]
            kps[1] = kps[1] / 1080 * img_size[1]

        np.save(os.path.join(output_dir,kpss[idx][:-4]+".jpg"+".npy"), modified_kps)
        imgA = cv2.imread(os.path.join(reference_dir,"Anime_Image.jpg")) # TODO hard code.
        imgA_cropped = imgA[0:1080, crop_l:crop_r,:]
        normalizedA = cv2.resize(imgA_cropped, img_size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(reference_dir, "animeImage.jpg"), normalizedA)


        if vis:
            for _joint in modified_kps:
                _x, _y = _joint
                _x, _y = int(_x),int(_y)
                cv2.circle(normalizedA, center=(int(_x), int(_y)), color=(255, 0, 0), radius=3, thickness=2)
            line_list = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[1,11],[8,9],[9,10],[11,12],[12,13],[0,14],[0,15]]
            for lines in line_list:
                cv2.line(normalizedA,(int(modified_kps[lines[0]][0]),int(modified_kps[lines[0]][1])),(int(modified_kps[lines[1]][0]),int(modified_kps[lines[1]][1]))
                         ,color=(255,0,0),thickness=4)
            # cv2.imwrite("D:\\download_cache\\anime_data\\vis_img\\" + outname, img)
            cv2.imwrite(os.path.join("\\".join(img_r.split("\\")[:-1])+"\\vis",kpss[idx][:-4]+".jpg"), normalizedA)
            print(os.path.join("\\".join(img_r.split("\\")[:-1])+"\\vis",kpss[idx][:-4]+".jpg"))


def extracted_frame(fr_n,video):
    # out_path = os.path.join(r"D:\download_cache\PMXmodel\real_shape","_".join(os.path.basename(video).split("_")[:-1])+".jpg")
    out_path = r"D:\work\OpenMMD1.0\examples\json_out_3d2\test.jpg"
    v = cv2.VideoCapture(video)
    flag, frame = v.read()
    frame_num = 0
    while flag:
        if frame_num == fr_n:
            cv2.imwrite(out_path,frame)
            break
        frame_num += 1
        flag, frame = v.read()

def motion_check():
    "dance_39_9_Teto_11"
    "dance_39_9_11"
    # Teto,KagamineRin    YYBHiganbanamiku   Vigna PaleEyes    Artemis   IA    KagamineRin  SatsukiKiryuin   AnsieNight   Bakugou,pmdfile1,Kaito,KizunaAi
    # dance_24_1 偏移事件
    # 2,lingmeng,madoka2019,MilkStraw,neru,JabamiYumeko,Joker,Kogawa,LEOTHELION,Miku,NazonoHeroine,Ochako,RyukoMatoi,TDALacyHaku,TDAPearlSouffleMiku,
    #
    file_dir = r"D:\download_cache\anime_data\vis"
    out_dir = r"D:\download_cache\anime_data\motion_check"
    import random
    file_list = os.listdir(file_dir)
    random.shuffle(file_list)
    for file in file_list:
        model_name = file.split("_")[3]
        # delete_list = ["Teto", "KagamineRin","YYBHiganbanamiku","Vigna","PaleEyes","Artemis","IA","KagamineRin","SatsukiKiryuin","AnsieNight","Bakugou"
        #                 , "pmdfile1", "Kaito", "KizunaAi"]
        # if model_name in delete_list:
        #     continue
        # if dance_name == "dance_46_2" and end_name=="8":
        if model_name in ["madoka2019", "RyukoMatoi", "neru","yousa"]:
            dance_name = "_".join(file.split("_")[:3]) + "_" + file.split("_")[-1]
            file_name = os.path.join(file_dir, file)
            out_name = os.path.join(out_dir, dance_name)
            # out_name = os.path.join(out_dir,file)
            shutil.copy(file_name, out_name)

    # np.random.shuffle(file_dir)

def getDatasetUsingClean():
    chosen_model = ["madoka2019", "RyukoMatoi", "neru","yousa","lingmeng","2","MilkStraw","Joker","Miku","MilkStraw","NazonoHeroine",
                    "Ochako","TDALacyHaku","LEOTHELION"] #14
    motion_files = os.listdir(r"D:\download_cache\anime_data\motion_check")
    kps_dir = r"D:\download_cache\anime_data\normK"
    kps_out = r"D:\download_cache\anime_data\normK_s"
    img_dir = r"D:\download_cache\anime_data\trainN"
    img_out = r"D:\download_cache\anime_data\train"

    for motion in motion_files:
        for model in chosen_model:
            try:
                filename = "_".join(motion.split("_")[:-1])+"_"+model+"_"+motion.split("_")[-1]
                shutil.copy(os.path.join(kps_dir,filename+".npy"),os.path.join(kps_out,filename+".npy"))
                shutil.copy(os.path.join(img_dir, filename), os.path.join(img_out,filename))
            except:
                continue


def genVideoFromPoseTransfer(frame_dir,img_size=(192,256)):
    frames = os.listdir(frame_dir)
    width,height = img_size[0],img_size[1]
    frames.sort(key= lambda x:int(x.split("_")[-2][:-4]))
    videoWrite = cv2.VideoWriter(os.path.join(os.path.dirname(frame_dir),'test.mp4'), -1, 15, (width * 3, height))  # 写入对象：1.fileName  2.-1：表示选择合适的编码器  3.视频的帧率  4.视频的size
    for idx,frame in enumerate(frames):
        img = cv2.imread(os.path.join(frame_dir,frame))
        img_used = img[:, width*2:, :]
        videoWrite.write(img_used)  # 写入方法  1.编码之前的图片数据
        print(idx)


if __name__=="__main__":
    # combineTwoVideo_width("D:\download_cache\PMXmodel\VIDEOclips\dance_10_8.avi","D:\download_cache\PMXmodel\OUTPUTclips\dance_10_8_GTGoku.avi","D:\download_cache\PMXmodel\compare.avi")
    # clips80 = [[190,262],[390,433],[530,590],[650,716],[777,830],[945,1050],[1085,1230],[1380,4135],[1510,1570],[1640,1730],[2141,2185],[4195,4320],[4860,4925]
    #            ]
    # genClipCsvFile("dance_20",clips80)
    # main()
    # ClipOriVideo()
    # extracted_frame(137,r"D:\work\OpenMMD1.0\examples\json_out_3d2\sample.mkv")
    getDatasetUsingClean()

    # fps = 30
    # imgs_dir = r"D:\download_cache\PMXmodel\real_shape"
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # video_writer = cv2.VideoWriter("test.avi", fourcc, fps, (1920, 1080))
    # # no glob, need number-index increasing
    # imgs = os.listdir(imgs_dir)
    # for i in imgs:
    #     if i.endswith(".jpg"):
    #         imgname = os.path.join(imgs_dir, i)
    #         frame = cv2.imread(imgname)
    #         frame = cv2.resize(frame,(1920, 1080))
    #         video_writer.write(frame)
    #
    # video_writer.release()

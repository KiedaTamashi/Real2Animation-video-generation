#put this file under the dir of SimpleHRNet
import cv2
from SimpleHRNet import SimpleHRNet
import numpy as np
import os


def image2points(model,iname="test.png"):

    image = cv2.imread(iname,cv2.IMREAD_COLOR)
    pks = model.predict(image)
    return np.array(pks[0])

def imagedir2points(image_dir="./samples",model_file="./weights/pose_hrnet_w48_384x288.pth"):
    # 48 channel, 17 points,resolution=(384, 288)
    model = SimpleHRNet(48, 17, model_file, multiperson=False)
    images = os.listdir(image_dir)
    outputs = list()
    [outputs.append(image2points(model,os.path.join(image_dir,x))) for x in images]
    print(len(outputs))
    for idx,output in enumerate(outputs):
        np.save(f"./output/{str(images[idx])}.npy")

def main():
    imagedir2points()

if __name__=="__main__":
    main()
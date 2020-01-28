import cv2
import numpy as np
import os

def render_joints(cvmat, joints, conf_th=0.2):
    for _joint in joints:
        _x, _y, _conf = _joint
        if _conf > conf_th:
            cv2.circle(cvmat, center=(int(_x), int(_y)), color=(255, 0, 0), radius=7, thickness=2)

    return cvmat

def main(indir="./output",outdir="./output",sample_dir="./samples"):
    for item in os.listdir(indir):
        item_npy = np.load(os.path.join(indir,item))
        ori_img = cv2.imread(os.path.join(sample_dir,item[:-4]))
        cvmat = render_joints(ori_img,item_npy, 0.2)
        cv2.imwrite(os.path.join(outdir,item[:-4]),cvmat)

main()
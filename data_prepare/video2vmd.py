import argparse
import os
import shutil
import subprocess

# parser = argparse.ArgumentParser()
# parser.add_argument("",default=,type=int)
# parser.add_argument("",default=,type=int)
# parser.add_argument("",default=,type=int)
# parser.add_argument("",default=,type=int)
# parser.add_argument("",default=,type=int)
# parser.add_argument("",default=,type=int)

base_dir = r"D:/work/OpenMMD1.0"
# step 1 video2keypoints - openpose
def video2keypoints(video_path,json_out_dir,number_people=1):
    '''
    :param video_path: e.g. examples/test.avi
    :param json_out_dir: examples/json_out
    :param number_people:
    :return:
    '''
    os.chdir(base_dir)
    #--display 0
    command = f"D:/work/OpenMMD1.0/bin/OpenPoseDemo.exe --video {video_path} --write_json {json_out_dir} --number_people_max {number_people} --display 0"
    print(os.system(command)) #0=success

# step 2 /3d-pose-baseline-vmd. kps2 3D
def kpsTo3D(json_out_dir,fps=25):
    '''
    :param json_out_dir: e.g. ../examples/json_out
    :param fps:
    :return:
    '''
    os.chdir(os.path.join(base_dir,"3d-pose-baseline-vmd"))
    command = f"python src/openpose_3dpose_sandbox_vmd.py --camera_frame --residual --batch_norm " \
              f"--dropout 0.5 --max_norm --evaluateActionWise --use_sh --epochs 200 --load 4874200 " \
              f"--gif_fps {fps} --verbose 2 --openpose {json_out_dir} --person_idx 1"
    os.system(command)

def video2depth(video_path,json_3d_folder,interval=10):
    '''
    :param video_path:  ../examples/test.avi
    :param json_3d_folder: ../examples/json_out_3d
    :param interval: smaller -> clearer the results
    :return:
    '''
    os.chdir(os.path.join(base_dir, "FCRN-DepthPrediction-vmd"))
    command = f"python tensorflow/predict_video.py --model_path tensorflow/data/NYU_FCRN.ckpt " \
              f"--video_path {video_path} --baseline_path {json_3d_folder} --interval {interval} " \
              f"--verbose 2"
    os.system(command)

def json3DtoVMD(json3d_dir, model_csv,output_name):
    '''
    :param json3d_dir: ../examples/json_out_3d
    :param model_csv: born/yuki_miku.csv
    :param output_name = ../examples/vmd_out/test.vmd
    :return:
    '''
    os.chdir(os.path.join(base_dir, "VMD-3d-pose-baseline-multi"))#TODO C is what?
    command = f"python applications\pos2vmd_multi.py -v 2 -t {json3d_dir} " \
              f"-b {model_csv} -c 30 -z 2 -x 15 -m 0 -i 1.5 -d 10 -a 1 -k 1 -e 0 " \
              f"-o {output_name}"
    os.system(command)

def video2VMD_single(video_path,json_out_dir,model_csv,output_name):
    # use absolute path
    json3d_folder = json_out_dir+"_3d"
    if os.path.exists(json3d_folder):
        shutil.rmtree(json3d_folder)
    if os.path.exists(json_out_dir):
        shutil.rmtree(json_out_dir)
    os.mkdir(json_out_dir)

    video2keypoints(video_path,json_out_dir)
    kpsTo3D(json_out_dir)
    video2depth(video_path,json3d_folder)
    json3DtoVMD(json3d_folder,model_csv,output_name)

def main():
    v_path = r"D:/work/OpenMMD1.0/examples/pose_test.mp4"
    json_out_dir = r"D:\work\OpenMMD1.0\examples\json_out"
    model_csv = r"D:\work\OpenMMD1.0\examples\SourClassicMiku\SourClassicMiku.csv"
    output_file = r"D:\work\OpenMMD1.0\examples\SourClassicMiku\SourClassicMiku.vmd"
    #TODO construct a .csv file
    # video2keypoints(v_path,json_out_dir)
    video2VMD_single(v_path,json_out_dir,model_csv,output_file)

if __name__=="__main__":
    main()
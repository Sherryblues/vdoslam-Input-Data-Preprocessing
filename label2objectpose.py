import os
import sys
import math
import numpy as np
import argparse
import glob as gb
from decimal import Decimal

# kitti数据集gt中标记的8中类别，DontCare标签表示该区域没有被标注
class_names = ["Car", "Cyclist", "Pedestrian", "Tram",
               "Person_sitting", "Truck", "Misc", "Van", 
               "DontCare"]

def main(args):
    # 如果存在之前生成的object_pose.txt或者time.txt，删除它们
    object_pose_path = args.path + "/object_pose.txt"
    time_path = args.path + "/times.txt"
    if os.path.isfile(object_pose_path):
        os.remove(object_pose_path)
        print("remove "+object_pose_path)
    if os.path.isfile(time_path):
        os.remove(time_path)
        print("remove "+time_path)
    # 读取文件夹下的.txt文件，注意文件夹下只能有一个.txt文件
    # .txt文件来自于data_tracking_label_2，文件名是0000.txt~0020.txt
    label_path = gb.glob(args.path + "/*.txt")
    path = label_path[0]
    # 生成object_pose.txt
    with open(path, 'r') as rf1:
        with open(object_pose_path, 'w') as wf:
            for line in rf1.readlines():
                ds = line.split()
                if ds[2]!="DontCare":
                    # 每一行为：FrameID ObjectID B1 B2 B3 B4 t1 t2 t3 r1
                    # 详细解释见vdoslam
                    wf.write(str(ds[0])+"\t"+str(int(ds[1])+1)+"\t"
                    +str(ds[6])+"\t"+str(ds[7])+"\t"+str(ds[8])+"\t"+str(ds[9])+"\t"
                    +str(ds[13])+"\t"+str(ds[14])+"\t"+str(ds[15])+"\t"+str(ds[16])+"\n")
    # 生成time.txt
    with open(path, 'r') as rf2:
        lines = rf2.readlines()
        last_line = lines[-1]
        frame_num = int(last_line.split()[0])
        with open(time_path, 'w') as twf:
            time = 0.00
            for i in range(0,frame_num+1):
                time_t = Decimal(time).quantize(Decimal("0.000001"), rounding = "ROUND_HALF_UP")
                twf.write(str(time_t)+'\n')
                # 两帧图片的时间间隔，自己定义，KITTI好像没有明确告知？
                time += 0.1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Deal path')
    args = parser.parse_args()
    main(args)

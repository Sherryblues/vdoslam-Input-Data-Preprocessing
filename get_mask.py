#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:28:21 2019

@author: lxh
"""

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
# Root directory of the project
ROOT_DIR = os.getcwd();#os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
import glob as gb

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("cuiwei***********************")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# 获得文件夹下所有图片的路径
img_path = gb.glob("./images/*.jpg")
# 对于每一张图片进行操作
for path in img_path:
    image = image = skimage.io.imread(path)
    # 输出的.mask文件路径
    mask_path = path[0:len(path)-4] + ".mask"
    # 实例分割
    results = model.detect([image], verbose=1)
    r = results[0]
    # masks有三维，分别是检测对象数目，图像高度，图像宽度，
    # 对于每一个检测对象，都是一个图像高度×图像宽度的二维数组，像素点属于对象则值为true，不属于对象则值为false
    masks = r['masks'].transpose(2,0,1)
    # class_ids存放了检测对象的类别id，结合class_names可以获得类别名
    class_ids = r['class_ids']
    # 打开.mask文件，如果不存在就创建该文件
    with open(mask_path, 'w') as f:
        # 写入第一行：图像高度 图像宽度 检测对象的数目
        f.write(str(masks.shape[1])+" "+str(masks.shape[2])+" "+str(masks.shape[0])+"\n")
        for i in range(len(class_ids)):
            # 获得并写入检测对象的类别，每一个占一行
            x = class_names[class_ids[i]]
            f.write(str(x)+"\n")
        # 创建一个和图像大小相同的数组，内容全部为0
        maskresult = np.zeros([masks.shape[1],masks.shape[2]], dtype=np.int8)
        # 对于每一个检测对象，修改对应数组maskresult中的值
        for i in range(len(class_ids)):
            maskresult = maskresult + masks[i] * (i+1)
        # 背景值改为-1，数组maskresult追加写入mask_path文件
        maskresult[maskresult==0] = -1
        np.savetxt(f, maskresult, fmt='%d', delimiter=' ', newline='\n')
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
    print("Done!")


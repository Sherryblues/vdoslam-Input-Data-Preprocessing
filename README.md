# vdoslam-Input-Data-Preprocessing
## 生成.mask文件
get_mask.py修改自 https://github.com/ssmem/vdo-slam-data-preprocess 中的generate_mask.py  
用于生成vdoslam需要的.mask文件，使用方法是放在Mask_RCNN文件夹中运行，下面是一个示例：
```shell
python get_mask.py /home/usr_name/Downloads/kitti-tracking-dataset/0000
```
get_mask.py将会读取目标文件夹image_0下的全部.png图像，新建mask文件夹，并将生成的.mask文件存放进去
## 生成object_pose.txt和time.txt文件
label2objectpose.py修改自 https://github.com/ssmem/vdo-slam-data-preprocess 中的label2objectpose.py  
用于生成vdoslam需要的object_pose.txt和time.txt文件，直接使用python运行，下面是一个示例：
```shell
python label2objectpose.py /home/slam/Downloads/kitti-tracking-dataset/0001
```
label2objectpose.py会读取目标文件下.txt文件，生成并保存需要的object_pose.txt和time.txt

# Cricket_Ball_Detection
This repository includes Ball detection algorithm for cricket.

## Wiki 
Project page: 

[HFR for Cricket](https://docs.google.com/document/d/1yUBaNMVKq3wfDfDrqXBv7RNMqWgl_jDU7SwH8HrDRvk/edit?usp=sharing), [4K/HFR Project Page](https://docs.google.com/document/d/1L2JEwZSHk-EXWHWD60Z84bTKHHhRTnnI0Xg9Bta9FzE/edit?usp=sharing)

## Quick Start
* Tested only on Linux machine
```bash
conda create -y -n mmdet python=3.7
# install dependencies: (use cu111 because colab has CUDA 11.1)
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# install mmcv-full thus we could use CUDA operators
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# Install mmdetection
rm -rf mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
%cd mmdetection

pip install -e .

# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

'''
1.9.0+cu111 True
2.21.0
11.1
GCC 7.3
'''

mkdir checkpoints
wget -c https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth \
      -O checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth

pip uninstall setuptools
pip install setuptools==59.5.0

```

## Pre-trained model 
tutorial_exps_refine : train with a single ball annotation dataset
tutorial_exps_refine_v2 : train with sucessive images for several ball detections

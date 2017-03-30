# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path('/home/zawlin/g/py-faster-rcnn/caffe-fast-rcnn/python')
add_path('/home/zawlin/g/py-faster-rcnn/lib')


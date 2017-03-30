import _init_paths
import cv2
import numpy as np
import caffe
import os
import cifar.layer

from caffe.proto import caffe_pb2
from caffe.io import blobproto_to_array
def start_train():
    caffe.set_device(0)
    caffe.set_mode_gpu()

    ### load the solver and create train and test nets
    solver = None

    #solver = caffe.SGDSolver('models/sg_vrd/relation/solver.prototxt')
    solver = caffe.SGDSolver('models/vg1_2/relation/solver_all.prototxt')

    niter = 1000000
    test_interval = 25
    # losses will also be stored in the log
    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe
        if niter%1000:
            print solver.net.blobs['loss'].data

start_train()

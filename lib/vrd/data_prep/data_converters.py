import _init_paths
from vrd.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np
import h5py
import cv2
import scipy.io as sio

from numpy.core.records import fromarrays
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
import h5py
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import os
import utils.zl_utils as zl

from numpy import linalg as LA
import operator

def convert_vr_gt_to_hdf5():
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta_gt.h5')

    rdata = sio.loadmat('/media/zawlin/ssd/data_vrd/vrd/annotation_test.mat', struct_as_record=False,squeeze_me=True)
    # map im_id to annotation
    r_annos = {}
    for i in xrange(len(rdata['annotation_test'])):
        anno = rdata['annotation_test'][i]
        im_id = anno.filename.split('.')[0]
        r_annos[im_id] = anno
    cnt = 0
    for imid in r_annos.keys():
        r_anno = r_annos[imid]
        rlp_labels = []
        obj_boxes=[]
        sub_boxes = []
        if hasattr(r_anno, 'relationship'):
            if not isinstance(r_anno.relationship, np.ndarray):
                r_anno.relationship = [r_anno.relationship]
            for r in xrange(len(r_anno.relationship)):
                if not hasattr(r_anno.relationship[r], 'phrase'):
                    continue
                predicate = r_anno.relationship[r].phrase[1]
                sub = r_anno.relationship[r].phrase[0]
                obj = r_anno.relationship[r].phrase[2]
                pre_idx = int(str(m['meta/pre/name2idx/' + predicate][...]))
                sub_cls_idx = int(str(m['meta/cls/name2idx/' + sub][...]))
                obj_cls_idx = int(str(m['meta/cls/name2idx/' + obj][...]))
                ymin, ymax, xmin, xmax = r_anno.relationship[r].subBox
                sub_box = [xmin, ymin, xmax, ymax]
                ymin, ymax, xmin, xmax = r_anno.relationship[r].objBox
                obj_box = [xmin, ymin, xmax, ymax]
                sub_boxes.append(sub_box)
                obj_boxes.append(obj_box)
                rlp_labels.append([sub_cls_idx,pre_idx,obj_cls_idx])

        m.create_dataset('gt/test/%s/sub_boxes'%imid,data = np.array(sub_boxes))
        m.create_dataset('gt/test/%s/obj_boxes'%imid,data = np.array(obj_boxes))
        m.create_dataset('gt/test/%s/rlp_labels'%imid,data = np.array(rlp_labels))


    rdata = sio.loadmat('/media/zawlin/ssd/data_vrd/vrd/annotation_train.mat', struct_as_record=False,squeeze_me=True)
    # map im_id to annotation
    r_annos = {}
    for i in xrange(len(rdata['annotation_train'])):
        anno = rdata['annotation_train'][i]
        im_id = anno.filename.split('.')[0]
        r_annos[im_id] = anno
    for imid in r_annos.keys():
        r_anno = r_annos[imid]
        rlp_labels = []
        obj_boxes=[]
        sub_boxes = []
        if hasattr(r_anno, 'relationship'):
            if not isinstance(r_anno.relationship, np.ndarray):
                r_anno.relationship = [r_anno.relationship]
            for r in xrange(len(r_anno.relationship)):
                if not hasattr(r_anno.relationship[r], 'phrase'):
                    continue
                predicate = r_anno.relationship[r].phrase[1]
                sub = r_anno.relationship[r].phrase[0]
                obj = r_anno.relationship[r].phrase[2]
                pre_idx = int(str(m['meta/pre/name2idx/' + predicate][...]))
                sub_cls_idx = int(str(m['meta/cls/name2idx/' + sub][...]))
                obj_cls_idx = int(str(m['meta/cls/name2idx/' + obj][...]))
                ymin, ymax, xmin, xmax = r_anno.relationship[r].subBox
                sub_box = [xmin, ymin, xmax, ymax]
                ymin, ymax, xmin, xmax = r_anno.relationship[r].objBox
                obj_box = [xmin, ymin, xmax, ymax]
                sub_boxes.append(sub_box)
                obj_boxes.append(obj_box)
                rlp_labels.append([sub_cls_idx,pre_idx,obj_cls_idx])
        m.create_dataset('gt/train/%s/sub_boxes'%imid,data = np.array(sub_boxes))
        m.create_dataset('gt/train/%s/obj_boxes'%imid,data = np.array(obj_boxes))
        m.create_dataset('gt/train/%s/rlp_labels'%imid,data = np.array(rlp_labels))

def convert_result_mat_to_hdf5():
    data = sio.loadmat('output/results/relationship_det_result.mat', struct_as_record=False, squeeze_me=True)
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta.h5')
    result = h5py.File('output/results/lu_visual_method_results.hdf5')
    rlp_confs=[]
    rlp_labels=[]
    subbox=[]
    objbox=[]
    for i in xrange(1000):
        sub_bboxes = data['sub_bboxes_ours'][i]
        obj_bboxes = data['obj_bboxes_ours'][i]
        rlp_confs_ours = data['rlp_confs_ours'][i]
        rlp_labels_ours = data['rlp_labels_ours'][i]
        if rlp_labels_ours.shape[0]>0:
            rlp_labels_ours[:,1]-=1
            # rlp_labels_ours[:,0]-=1
            # rlp_labels_ours[:,2]-=1
        #rlp_labels_ours[:,1]-=1
        imid = str(m['db/testidx/'+str(i)][...])
        result.create_dataset(imid+'/rlp_confs',dtype='float16', data=np.array(rlp_confs_ours).astype(np.float16))
        result.create_dataset(imid+'/sub_boxes',dtype='float16', data=np.array(sub_bboxes).astype(np.float16))
        result.create_dataset(imid+'/obj_boxes',dtype='float16', data=np.array(obj_bboxes).astype(np.float16))
        result.create_dataset(imid+'/rlp_labels',dtype='float16', data=np.array(rlp_labels_ours).astype(np.float16))


def convert_sg_vrd_meta_to_voc_eval():
    dst = '/media/zawlin/ssd/data_vrd/vrd/sg/devkit/data'
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta.h5')
    for i in m['gt/test/'].keys():
        print i
        pass
    pass
#convert_vr_gt_to_hdf5()
convert_result_mat_to_hdf5()
#convert_sg_vrd_meta_to_voc_eval()


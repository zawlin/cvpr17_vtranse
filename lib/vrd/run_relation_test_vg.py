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

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    #x = x/np.max(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_relation_batch(model_type,iteration):
    vgg_h5 = h5py.File('output/precalc/vg1_2_2016_test.hdf5')
    vgg_data = {}
    if os.path.exists('output/cache/vg1_2_2016_test.pkl'):
        vgg_data = zl.load('output/cache/vg1_2_2016_test.pkl')
        print 'loaded test data from cache'
    else:
        print 'Preloading testing data'
        zl.tick()
        for k in vgg_h5.keys():
            classemes = vgg_h5[k]['classemes'][...]
            visuals = vgg_h5[k]['visuals'][...]
            locations = vgg_h5[k]['locations'][...]
            cls_confs = vgg_h5[k]['cls_confs'][...]
            vgg_data[k]={}
            vgg_data[k]['classemes']=classemes
            vgg_data[k]['visuals']=visuals
            vgg_data[k]['cls_confs']=cls_confs
            vgg_data[k]['locations']=locations
        print 'done preloading testing data %f'%zl.tock()
        zl.save('output/cache/vg1_2_2016_test.pkl',vgg_data)
        vgg_h5.close()
    result = h5py.File('output/vg_results/vg1_2_2016_result_'+model_type+'_'+iteration+'.hdf5')
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/vg1_2_meta.h5')
    data_root='/home/zawlin/g/py-faster-rcnn/data/vg1_2_2016/Data/test/'
    keep = 100
    thresh = 0.0001
    net = caffe.Net('models/vg1_2/relation/test_'+model_type+'.prototxt','output/relation/vg/relation_vgg16_'+model_type+'_iter_'+iteration+'.caffemodel',caffe.TEST)
    #net = caffe.Net('models/sg_vrd/relation/test.prototxt','output/models/sg_vrd_relation_vgg16_iter_264000.caffemodel',caffe.TEST)
    cnt =1
    zl.tick()
    imids = sorted(vgg_data.keys())
    for imid in imids:
        if cnt %100==0:
            print cnt,zl.tock()
            zl.tick()
        cnt+=1
        if imid in result:continue
        classemes = vgg_data[imid]['classemes']
        visuals = vgg_data[imid]['visuals']
        locations = vgg_data[imid]['locations']
        cls_confs = vgg_data[imid]['cls_confs']

        # im = cv2.imread(data_root+imid+'.jpg')
        # #print cls_confs
        # # for box in locations:
            # # b=box[:4].astype(np.int32)
            # # cv2.rectangle(im,(b[0],b[1]),(b[2],b[3]),(255,0,0))
        # w,h = im.shape[2],im.shape[1]

        rlp_labels = []
        rlp_confs = []
        sub_boxes=[]
        obj_boxes=[]
        relation_vectors = []

        classemes_in = []
        visuals_in = []
        locations_in = []
        cls_confs_in = []
        sub_cls_in = []
        obj_cls_in = []
        sub_score_in = []
        obj_score_in = []
        sub_boxes = []
        obj_boxes = []
        for s in xrange(len(locations)):
            for o in xrange(len(locations)):
                if s==o:continue
                sub = locations[s]
                obj = locations[o]
                sub_visual = visuals[s]
                obj_visual = visuals[o]
                sub_cls = cls_confs[s,0]
                obj_cls = cls_confs[o,0]
                sub_score = cls_confs[s,1]
                obj_score = cls_confs[o,1]
                sub_classme = classemes[s]
                obj_classme = classemes[o]
                sub_loc_encoded = bbox_transform( np.array([obj[:4]]), np.array([sub[:4]]))[0]
                obj_loc_encoded = bbox_transform( np.array([sub[:4]]), np.array([obj[:4]]))[0]

                visual = np.hstack((sub_visual, obj_visual)).reshape(8192)
                classeme = np.hstack((sub_classme, obj_classme)).reshape(402)
                loc = np.hstack((sub_loc_encoded, obj_loc_encoded)).reshape(8)

                classemes_in.append(classeme)
                visuals_in.append(visual)
                locations_in.append(loc)
                sub_cls_in.append(sub_cls)
                obj_cls_in.append(obj_cls)
                sub_score_in.append(sub_score)
                obj_score_in.append(obj_score)
                sub_boxes.append(sub[:4])
                obj_boxes.append(obj[:4])

        if 'all' in model_type:
            blob = {
                    'classeme':np.array(classemes_in),
                    'visual':np.array(visuals_in),
                    'location':np.array(locations_in)
                    }
            net.blobs['classeme'].reshape(*blob['classeme'].shape)
            net.blobs['visual'].reshape(*blob['visual'].shape)
            net.blobs['location'].reshape(*blob['location'].shape)
        elif 'visual' in model_type:
            blob = {
                    'visual':np.array(visuals_in),
                    }
            net.blobs['visual'].reshape(*blob['visual'].shape)
        elif 'classeme' in model_type:
            blob = {
                    'classeme':np.array(classemes_in),
                    }

            net.blobs['classeme'].reshape(*blob['classeme'].shape)
        elif 'location' in model_type:
            blob = {
                    'location':np.array(locations_in)
                    }
                #batch this
            net.blobs['location'].reshape(*blob['location'].shape)
        if len(locations_in)==0:
            rlp_confs = []
            sub_boxes = []
            obj_boxes = []
            rlp_labels = []
        else:
            net.forward_all(**blob)
            relation_score = net.blobs['relation_prob'].data.copy()
            argmax = np.argmax(relation_score,axis=1)
            rs = relation_score[np.arange(relation_score.shape[0]),argmax]
            rlp_labels = np.vstack((sub_cls_in,argmax,obj_cls_in)).T
            rlp_confs = np.array(sub_score_in)+np.array(rs)+np.array(obj_score_in)
        result.create_dataset(imid+'/rlp_confs',dtype='float16', data=np.array(rlp_confs).astype(np.float16))
        result.create_dataset(imid+'/sub_boxes',dtype='float16', data=np.array(sub_boxes).astype(np.float16))
        result.create_dataset(imid+'/obj_boxes',dtype='float16', data=np.array(obj_boxes).astype(np.float16))
        result.create_dataset(imid+'/rlp_labels',dtype='float16', data=np.array(rlp_labels).astype(np.float16))

def run_relation_batch_all(model_type,iteration):
    vgg_h5 = h5py.File('output/precalc/vg1_2_2016_test.hdf5')
    vgg_data = {}
    if os.path.exists('output/cache/vg1_2_2016_test.pkl'):
        vgg_data = zl.load('output/cache/vg1_2_2016_test.pkl')
        print 'loaded test data from cache'
    else:
        print 'Preloading testing data'
        zl.tick()
        for k in vgg_h5.keys():
            classemes = vgg_h5[k]['classemes'][...]
            visuals = vgg_h5[k]['visuals'][...]
            locations = vgg_h5[k]['locations'][...]
            cls_confs = vgg_h5[k]['cls_confs'][...]
            vgg_data[k]={}
            vgg_data[k]['classemes']=classemes
            vgg_data[k]['visuals']=visuals
            vgg_data[k]['cls_confs']=cls_confs
            vgg_data[k]['locations']=locations
        print 'done preloading testing data %f'%zl.tock()
        zl.save('output/cache/vg1_2_2016_test.pkl',vgg_data)
        vgg_h5.close()
    result = h5py.File('output/vg_results/vg1_2_2016_result_'+model_type+'_'+iteration+'.all.hdf5')
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/vg1_2_meta.h5')
    data_root='/home/zawlin/g/py-faster-rcnn/data/vg1_2_2016/Data/test/'
    keep = 100
    thresh = 0.0001
    net = caffe.Net('models/vg1_2/relation/test_'+model_type+'.prototxt','output/relation/vg/relation_vgg16_'+model_type+'_iter_'+iteration+'.caffemodel',caffe.TEST)
    #net = caffe.Net('models/sg_vrd/relation/test.prototxt','output/models/sg_vrd_relation_vgg16_iter_264000.caffemodel',caffe.TEST)
    cnt =1
    zl.tick()
    imids = sorted(vgg_data.keys())
    for imid in imids:
        if cnt %100==0:
            print cnt,zl.tock()
            zl.tick()
            # if cnt >1000:
                # exit(0)
        cnt+=1

        classemes = vgg_data[imid]['classemes']
        visuals = vgg_data[imid]['visuals']
        locations = vgg_data[imid]['locations']
        cls_confs = vgg_data[imid]['cls_confs']

        # im = cv2.imread(data_root+imid+'.jpg')
        # #print cls_confs
        # # for box in locations:
            # # b=box[:4].astype(np.int32)
            # # cv2.rectangle(im,(b[0],b[1]),(b[2],b[3]),(255,0,0))
        # w,h = im.shape[2],im.shape[1]

        rlp_labels = []
        rlp_confs = []
        sub_boxes=[]
        obj_boxes=[]
        relation_vectors = []

        classemes_in = []
        visuals_in = []
        locations_in = []
        cls_confs_in = []
        sub_cls_in = []
        obj_cls_in = []
        sub_score_in = []
        obj_score_in = []
        sub_boxes = []
        obj_boxes = []
        sub_boxes_final = []
        obj_boxes_final = []
        for s in xrange(len(locations)):
            for o in xrange(len(locations)):
                if s==o:continue
                sub = locations[s]
                obj = locations[o]
                sub_visual = visuals[s]
                obj_visual = visuals[o]
                sub_cls = cls_confs[s,0]
                obj_cls = cls_confs[o,0]
                sub_score = cls_confs[s,1]
                obj_score = cls_confs[o,1]
                sub_classme = classemes[s]
                obj_classme = classemes[o]
                sub_loc_encoded = bbox_transform( np.array([obj[:4]]), np.array([sub[:4]]))[0]
                obj_loc_encoded = bbox_transform( np.array([sub[:4]]), np.array([obj[:4]]))[0]

                visual = np.hstack((sub_visual, obj_visual)).reshape(8192)
                classeme = np.hstack((sub_classme, obj_classme)).reshape(402)
                loc = np.hstack((sub_loc_encoded, obj_loc_encoded)).reshape(8)

                classemes_in.append(classeme)
                visuals_in.append(visual)
                locations_in.append(loc)
                sub_cls_in.append(sub_cls)
                obj_cls_in.append(obj_cls)
                sub_score_in.append(sub_score)
                obj_score_in.append(obj_score)
                sub_boxes.append(sub[:4])
                obj_boxes.append(obj[:4])
        if 'all' in model_type:
            blob = {
                    'classeme':np.array(classemes_in),
                    'visual':np.array(visuals_in),
                    'location':np.array(locations_in)
                    }
            net.blobs['classeme'].reshape(*blob['classeme'].shape)
            net.blobs['visual'].reshape(*blob['visual'].shape)
            net.blobs['location'].reshape(*blob['location'].shape)
        elif 'visual' in model_type:
            blob = {
                    'visual':np.array(visuals_in),
                    }
            net.blobs['visual'].reshape(*blob['visual'].shape)
        elif 'classeme' in model_type:
            blob = {
                    'classeme':np.array(classemes_in),
                    }

            net.blobs['classeme'].reshape(*blob['classeme'].shape)
        elif 'location' in model_type:
            blob = {
                    'location':np.array(locations_in)
                    }
                #batch this
            net.blobs['location'].reshape(*blob['location'].shape)
        if len(locations_in)==0:
            rlp_confs = []
            sub_boxes = []
            obj_boxes = []
            rlp_labels = []
        else:
            net.forward_all(**blob)
            relation_scores = net.blobs['relation_prob'].data.copy()
            for n in xrange(relation_scores.shape[0]):
                relation_score = relation_scores[n]
                sub_box = sub_boxes[n]
                obj_box = obj_boxes[n]
                sub_cls = sub_cls_in[n]
                obj_cls = obj_cls_in[n]
                sub_score = sub_score_in[n]
                obj_score = obj_score_in[n]
                for i in xrange(100):
                    rs = relation_score[i]
                    predicate =i
                    rlp_label = np.array([sub_cls,predicate,obj_cls]).astype(np.int32)
                    rlp_conf = rs+sub_score+obj_score#relation_score[predicate]
                    rlp_confs.append(rlp_conf)
                    rlp_labels.append(rlp_label)
                    sub_boxes_final.append(sub_box)
                    obj_boxes_final.append(obj_box)
        result.create_dataset(imid+'/rlp_confs',dtype='float16', data=np.array(rlp_confs).astype(np.float16))
        result.create_dataset(imid+'/sub_boxes',dtype='float16', data=np.array(sub_boxes_final).astype(np.float16))
        result.create_dataset(imid+'/obj_boxes',dtype='float16', data=np.array(obj_boxes_final).astype(np.float16))
        result.create_dataset(imid+'/rlp_labels',dtype='float16', data=np.array(rlp_labels).astype(np.float16))

def run_relation(model_type,iteration):
    vgg_data = h5py.File('output/precalc/vg1_2_2016_test.hdf5')
    result = h5py.File('output/vg_results/vg1_2_2016_result_'+model_type+'_'+iteration+'.hdf5')
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/vg1_2_meta.h5')
    data_root='/home/zawlin/g/py-faster-rcnn/data/vg1_2_2016/Data/test/'
    keep = 100
    thresh = 0.0001
    net = caffe.Net('models/vg1_2/relation/test_'+model_type+'.prototxt','output/relation/vg/relation_vgg16_'+model_type+'_iter_'+iteration+'.caffemodel',caffe.TEST)
    #net = caffe.Net('models/sg_vrd/relation/test.prototxt','output/models/sg_vrd_relation_vgg16_iter_264000.caffemodel',caffe.TEST)
    cnt =1
    zl.tick()
    for imid in vgg_data.keys():
        if cnt %100==0:
            print cnt,zl.tock()
            zl.tick()
            exit(0)
        cnt+=1
        # if cnt%100==0:
            # print cnt

        classemes = vgg_data[imid]['classemes']
        visuals = vgg_data[imid]['visuals']
        locations = vgg_data[imid]['locations']
        cls_confs = vgg_data[imid]['cls_confs']

        # im = cv2.imread(data_root+imid+'.jpg')
        # #print cls_confs
        # # for box in locations:
            # # b=box[:4].astype(np.int32)
            # # cv2.rectangle(im,(b[0],b[1]),(b[2],b[3]),(255,0,0))
        # w,h = im.shape[2],im.shape[1]

        rlp_labels = []
        rlp_confs = []
        sub_boxes=[]
        obj_boxes=[]
        relation_vectors = []

        classemes_in = []
        visuals_in = []
        locations_in = []
        cls_confs_in = []
        for s in xrange(len(locations)):
            for o in xrange(len(locations)):
                if s==o:continue
                sub = locations[s]
                obj = locations[o]
                sub_visual = visuals[s]
                obj_visual = visuals[o]
                sub_cls = cls_confs[s,0]
                obj_cls = cls_confs[o,0]
                sub_score = cls_confs[s,1]
                obj_score = cls_confs[o,1]
                sub_classme = classemes[s]
                obj_classme = classemes[o]
                if sub_score<0.01 or obj_score<0.01:continue
                sub_loc_encoded = bbox_transform( np.array([obj[:4]]), np.array([sub[:4]]))[0]
                obj_loc_encoded = bbox_transform( np.array([sub[:4]]), np.array([obj[:4]]))[0]
                #sub_loc_encoded = bbox_transform(np.array([[0, 0, w, h]]), np.array([sub[:4]]))[0]
                #obj_loc_encoded = bbox_transform(np.array([[0, 0, w, h]]), np.array([obj[:4]]))[0]

                visual = np.hstack((sub_visual, obj_visual)).reshape(1,8192)
                classeme = np.hstack((sub_classme, obj_classme)).reshape(1,402)
                loc = np.hstack((sub_loc_encoded, obj_loc_encoded)).reshape(1,8)
                if 'all' in model_type:
                    blob = {
                            'classeme':classeme,
                            'visual':visual,
                            'location':loc
                            }
                elif 'visual' in model_type:
                    blob = {
                            'visual':visual,
                            }
                elif 'classeme' in model_type:
                    blob = {
                            'classeme':classeme,
                            }
                elif 'location' in model_type:
                    blob = {
                            'location':loc
                            }
                #batch this
                net.forward_all(**blob)

                relation_score =net.blobs['relation_prob'].data[0].copy()
                #l2_norm = relation_score/LA.norm(relation_score)
                #relation_score=softmax(relation_score)
                #relation_score/=LA.norm(relation_score)
                #relation_score=softmax(relation_score)
                argmax = np.argmax(relation_score)
                rs = relation_score[argmax]
                predicate = argmax
                rlp_label = np.array([sub_cls,predicate,obj_cls]).astype(np.int32)
                rlp_conf = rs+sub_score+obj_score#relation_score[predicate]

                rlp_confs.append(rlp_conf)
                rlp_labels.append(rlp_label)
                sub_boxes.append(sub[:4])
                obj_boxes.append(obj[:4])
                relation_vectors.append(relation_score)
                # for i in xrange(70):
                    # rs = relation_score[i]
                    # if rs>0.0:
                        # predicate =i
                        # rlp_label = np.array([sub_cls,predicate,obj_cls]).astype(np.int32)
                        # rlp_conf = rs+sub_score+obj_score#relation_score[predicate]

                        # rlp_confs.append(rlp_conf)
                        # rlp_labels.append(rlp_label)
                        # sub_boxes.append(sub[:4])
                        # obj_boxes.append(obj[:4])

        result.create_dataset(imid+'/rlp_confs',dtype='float16', data=np.array(rlp_confs).astype(np.float16))
        result.create_dataset(imid+'/sub_boxes',dtype='float16', data=np.array(sub_boxes).astype(np.float16))
        result.create_dataset(imid+'/obj_boxes',dtype='float16', data=np.array(obj_boxes).astype(np.float16))
        result.create_dataset(imid+'/rlp_labels',dtype='float16', data=np.array(rlp_labels).astype(np.float16))
        # result.create_dataset(imid+'/relation_vectors', data=np.array(relation_vectors).astype(np.float16))

def make_meta():
    data = sio.loadmat('/home/zawlin/g/Visual-Relationship-Detection/data/imagePath.mat', struct_as_record=False, squeeze_me=True)
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta.h5')
    for i in xrange(len(data['imagePath'])):
        m['db/testidx/'+str(i)]=data['imagePath'][i].split('.')[0]
    pass

def make_relation_result(model_type,iteration):
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/vg1_2_meta.h5')
    result_str = 'vg1_2_2016_result_'+model_type+'_'+iteration
    result = h5py.File('output/vg_results/'+result_str+'.hdf5')
    rlp_confs=[]
    rlp_labels=[]
    subbox=[]
    objbox=[]
    relation_vectors = []
    imids = sorted(m['gt/test'].keys())
    for imid in imids:
        if imid in result:
            objbox.append(result[imid+'/obj_boxes'][...])#\.reshape(-1,4).T)
            subbox.append(result[imid+'/sub_boxes'][...])
            rlp_labels.append(result[imid+'/rlp_labels'][...])
            rlp_confs.append(result[imid+'/rlp_confs'][...].T)
            #relation_vectors.append(result[imid+'/relation_vectors'][...])
        else:
            rlp_confs.append([])
            rlp_labels.append([])
            subbox.append([])
            objbox.append([])
            #relation_vectors.append([])

    #print objbox
    #objboxx=np.array(objbox)#.astype(np.float64)
    sio.savemat('output/'+result_str+'.mat', {'obj_bboxes_ours': objbox,'sub_bboxes_ours':subbox,
        'rlp_labels_ours':rlp_labels,'rlp_confs_ours':rlp_confs})#'relation_vectors':relation_vectors})

caffe.set_mode_gpu()
caffe.set_device(0)

# model_type = 'all'
# iteration = '100000'
# run_relation_batch_all(model_type,iteration)
# make_relation_result(model_type,iteration)

# model_type = 'visual'
# iteration = '100000'
# run_relation_batch(model_type,iteration)
# make_relation_result(model_type,iteration)

# model_type = 'classeme'
# iteration = '100000'
# run_relation_batch(model_type,iteration)
# make_relation_result(model_type,iteration)

model_type = 'classeme'
iteration = '20000'
run_relation_batch(model_type,iteration)
make_relation_result(model_type,iteration)

# model_type = 'all'
# iteration = '40000'
# run_relation_batch(model_type,iteration)
# make_relation_result(model_type,iteration)

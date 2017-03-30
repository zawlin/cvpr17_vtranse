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

def run_relation(model_type,iteration):
    vgg_data = h5py.File('output/sg_vrd_2016_test.hdf5')
    result = h5py.File('output/sg_vrd_2016_result_'+model_type+'_'+iteration+'.hdf5')
    #if os.path.exists('output/sg_vrd_2016_result.hdf5'):
    #    os.remove('output/sg_vrd_2016_result.hdf5')
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta.h5')
    data_root='/home/zawlin/g/py-faster-rcnn/data/sg_vrd_2016/Data/sg_test_images/'
    keep = 100
    thresh = 0.0001
    net = caffe.Net('models/sg_vrd/relation/test_'+model_type+'.prototxt','output/relation/vr/sg_vrd_relation_vgg16_'+model_type+'_iter_'+iteration+'.caffemodel',caffe.TEST)
    #net = caffe.Net('models/sg_vrd/relation/test.prototxt','output/models/sg_vrd_relation_vgg16_iter_264000.caffemodel',caffe.TEST)
    cnt =0
    zl.tick()
    rel_types = {}
    rel_types['p']=[]
    rel_types['s']=[]
    rel_types['v']=[]
    rel_types['c']=[]
    for k in m['meta/pre/name2idx'].keys():
        idx = int(str(m['meta/pre/name2idx/'+k][...]))
        r_type = m['meta/pre/name2idx/'+k].attrs['type']
        rel_types[r_type].append(idx)

    for imid in vgg_data.keys():
        cnt+=1
        print cnt,zl.tock()
        zl.tick()
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
                classeme = np.hstack((sub_classme, obj_classme)).reshape(1,202)
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
                # argmax = np.argmax(relation_score)
                # rs = relation_score[argmax]
                # predicate = argmax
                # rlp_label = np.array([sub_cls,predicate,obj_cls]).astype(np.int32)
                # # rlp_conf = rs+sub_score+obj_score#relation_score[predicate]
                # rlp_conf = rs+sub_score+obj_score#*sub_score*obj_score
                # rlp_confs.append(rlp_conf)
                # rlp_labels.append(rlp_label)
                # sub_boxes.append(sub[:4])
                # obj_boxes.append(obj[:4])
                # relation_vectors.append(relation_score)

                # for i in xrange(70):
                    # rs = relation_score[i]
                    # if rs>0.0:
                        # predicate =i
                        # rlp_label = np.array([sub_cls,predicate,obj_cls]).astype(np.int32)
                        # # rlp_conf = rs+sub_score+obj_score#relation_score[predicate]
                        # rlp_conf = rs
                        # rlp_confs.append(rlp_conf)
                        # rlp_labels.append(rlp_label)
                        # sub_boxes.append(sub[:4])
                        # obj_boxes.append(obj[:4])
                r_scores = {'s':{},'v':{},'c':{},'p':{}}
                for i in xrange(70):
                    rs = relation_score[i]
                    if i in rel_types['s']:r_scores['s'][i] = rs
                    if i in rel_types['v']:r_scores['v'][i] = rs
                    if i in rel_types['c']:r_scores['c'][i] = rs
                    if i in rel_types['p']:r_scores['p'][i] = rs
                r_scores['s'] = zl.sort_dict_by_val(r_scores['s'])
                r_scores['v'] = zl.sort_dict_by_val(r_scores['v'])
                r_scores['c'] = zl.sort_dict_by_val(r_scores['c'])
                r_scores['p'] = zl.sort_dict_by_val(r_scores['p'])
                for i,rs in r_scores['s'][:4]:
                    predicate =i
                    rlp_label = np.array([sub_cls,predicate,obj_cls]).astype(np.int32)
                    # rlp_conf = rs+sub_score+obj_score#relation_score[predicate]
                    rlp_conf = rs
                    rlp_confs.append(rlp_conf)
                    rlp_labels.append(rlp_label)
                    sub_boxes.append(sub[:4])
                    obj_boxes.append(obj[:4])
                for i,rs in r_scores['v'][:4]:
                    predicate =i
                    rlp_label = np.array([sub_cls,predicate,obj_cls]).astype(np.int32)
                    # rlp_conf = rs+sub_score+obj_score#relation_score[predicate]
                    rlp_conf = rs
                    rlp_confs.append(rlp_conf)
                    rlp_labels.append(rlp_label)
                    sub_boxes.append(sub[:4])
                    obj_boxes.append(obj[:4])
                for i,rs in r_scores['p'][:4]:
                    predicate =i
                    rlp_label = np.array([sub_cls,predicate,obj_cls]).astype(np.int32)
                    # rlp_conf = rs+sub_score+obj_score#relation_score[predicate]
                    rlp_conf = rs
                    rlp_confs.append(rlp_conf)
                    rlp_labels.append(rlp_label)
                    sub_boxes.append(sub[:4])
                    obj_boxes.append(obj[:4])
                for i,rs in r_scores['c']:
                    predicate =i
                    rlp_label = np.array([sub_cls,predicate,obj_cls]).astype(np.int32)
                    # rlp_conf = rs+sub_score+obj_score#relation_score[predicate]
                    rlp_conf = rs
                    rlp_confs.append(rlp_conf)
                    rlp_labels.append(rlp_label)
                    sub_boxes.append(sub[:4])
                    obj_boxes.append(obj[:4])
                    # if rs>0.0:
                        # predicate =i
                        # rlp_label = np.array([sub_cls,predicate,obj_cls]).astype(np.int32)
                        # # rlp_conf = rs+sub_score+obj_score#relation_score[predicate]
                        # rlp_conf = rs
                        # rlp_confs.append(rlp_conf)
                        # rlp_labels.append(rlp_label)
                        # sub_boxes.append(sub[:4])
                        # obj_boxes.append(obj[:4])

        result.create_dataset(imid+'/rlp_confs',dtype='float16', data=np.array(rlp_confs).astype(np.float16))
        result.create_dataset(imid+'/sub_boxes',dtype='float16', data=np.array(sub_boxes).astype(np.float16))
        result.create_dataset(imid+'/obj_boxes',dtype='float16', data=np.array(obj_boxes).astype(np.float16))
        result.create_dataset(imid+'/rlp_labels',dtype='float16', data=np.array(rlp_labels).astype(np.float16))
        # result.create_dataset(imid+'/relation_vectors', data=np.array(relation_vectors).astype(np.float16))

def run_relation_diff(model_type,iteration):
    #vgg_data = h5py.File('output/sg_vrd_2016_test.hdf5')
    vgg_data = h5py.File('output/sg_vrd_2016_test_more.hdf5')
    result = h5py.File('output/sg_vrd_2016_result_'+model_type+'_'+iteration+'.hdf5')
    #if os.path.exists('output/sg_vrd_2016_result.hdf5'):
    #    os.remove('output/sg_vrd_2016_result.hdf5')
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta.h5')
    data_root='/home/zawlin/g/py-faster-rcnn/data/sg_vrd_2016/Data/sg_test_images/'
    keep = 100
    thresh = 0.0001
    net = caffe.Net('models/sg_vrd/relation/test_'+model_type+'.prototxt','output/relation/sg_vrd_relation_vgg16_'+model_type+'_iter_'+iteration+'.caffemodel',caffe.TEST)
    #net = caffe.Net('models/sg_vrd/relation/test.prototxt','output/models/sg_vrd_relation_vgg16_iter_264000.caffemodel',caffe.TEST)
    cnt =0
    zl.tick()
    for imid in vgg_data.keys():
        cnt+=1
        print cnt,zl.tock()
        zl.tick()

        classemes = vgg_data[imid]['classemes']
        visuals = vgg_data[imid]['visuals']
        locations = vgg_data[imid]['locations']
        cls_confs = vgg_data[imid]['cls_confs']

        #im = cv2.imread(data_root+imid+'.jpg')
        #print cls_confs
        # for box in locations:
            # b=box[:4].astype(np.int32)
            # cv2.rectangle(im,(b[0],b[1]),(b[2],b[3]),(255,0,0))

        rlp_labels = []
        rlp_confs = []
        sub_boxes=[]
        obj_boxes=[]
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
                if sub_score<0.1 or obj_score<0.1:continue
                sub_classme = classemes[s]
                obj_classme = classemes[o]
                sub_loc_encoded = bbox_transform( np.array([obj[:4]]), np.array([sub[:4]]))[0]
                obj_loc_encoded = bbox_transform( np.array([sub[:4]]), np.array([obj[:4]]))[0]
                visual = np.hstack((sub_visual, obj_visual)).reshape(1,8192)
                classeme = np.hstack((sub_classme, obj_classme)).reshape(1,202)
                loc = sub_loc_encoded.reshape(1,4)#np.hstack((sub_loc_encoded, obj_loc_encoded)).reshape(1,4)
                if 'all' in model_type:
                    blob = {
                        'classeme_s':np.array(sub_classme).reshape(1,101),
                        'classeme_o':np.array(obj_classme).reshape(1,101),
                        'visual_s':np.array(sub_visual).reshape(1,4096),
                        'visual_o':np.array(obj_visual).reshape(1,4096),
                        'location_s':np.array(sub_loc_encoded).reshape(1,4),
                        'location_o':np.array(obj_loc_encoded).reshape(1,4),
                        }
                elif 'visual' in model_type:
                    blob = {
                        'visual_s':np.array(sub_visual).reshape(1,4096),
                        'visual_o':np.array(obj_visual).reshape(1,4096),
                        }
                elif 'classeme' in model_type:
                    blob = {
                        'classeme_s':np.array(sub_classme).reshape(1,101),
                        'classeme_o':np.array(obj_classme).reshape(1,101),
                        }
                elif 'location' in model_type:
                    blob = {
                        'location_s':np.array(sub_loc_encoded).reshape(1,4),
                        'location_o':np.array(obj_loc_encoded).reshape(1,4),
                        }
                #print visual.shape
                net.forward_all(**blob)
                relation_score =net.blobs['relation'].data[0]
                #l2_norm = relation_score/LA.norm(relation_score)
                relation_score=softmax(relation_score)
                argmax = np.argmax(relation_score)
                rs = relation_score[argmax]

                predicate = argmax
                rlp_label = np.array([sub_cls,predicate,obj_cls]).astype(np.int32)
                #print '%s %s %s %f'%(m['meta/cls/idx2name/'+str(rlp_label[0])][...],m['meta/pre/idx2name/'+str(rlp_label[1])][...],m['meta/cls/idx2name/'+str(rlp_label[2])][...],rs)
                rlp_conf = rs+sub_score+obj_score#relation_score[predicate]

                rlp_confs.append(rlp_conf)
                rlp_labels.append(rlp_label)
                sub_boxes.append(sub[:4])
                obj_boxes.append(obj[:4])
                #relation_score/=LA.norm(relation_score)
                # for i in xrange(70):
                    # rs = relation_score[i]
                    # if rs>0.0:
                        # predicate =i
                        # #print relation_score[predicate]
                        # rlp_label = np.array([sub_cls,predicate,obj_cls]).astype(np.int32)
                        # #print '%s %s %s %f'%(m['meta/cls/idx2name/'+str(rlp_label[0])][...],m['meta/pre/idx2name/'+str(rlp_label[1])][...],m['meta/cls/idx2name/'+str(rlp_label[2])][...],rs)
                        # rlp_conf = rs+sub_score+obj_score#relation_score[predicate]

                        # rlp_confs.append(rlp_conf)
                        # rlp_labels.append(rlp_label)
                        # sub_boxes.append(sub[:4])
                        # obj_boxes.append(obj[:4])

        result.create_dataset(imid+'/rlp_confs',dtype='float16', data=np.array(rlp_confs).astype(np.float16))
        result.create_dataset(imid+'/sub_boxes',dtype='float16', data=np.array(sub_boxes).astype(np.float16))
        result.create_dataset(imid+'/obj_boxes',dtype='float16', data=np.array(obj_boxes).astype(np.float16))
        result.create_dataset(imid+'/rlp_labels',dtype='float16', data=np.array(rlp_labels).astype(np.float16))

def make_meta():
    data = sio.loadmat('/home/zawlin/g/Visual-Relationship-Detection/data/imagePath.mat', struct_as_record=False, squeeze_me=True)
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta.h5')
    for i in xrange(len(data['imagePath'])):
        m['db/testidx/'+str(i)]=data['imagePath'][i].split('.')[0]
    pass

def make_relation_result(model_type,iteration):
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta.h5')
    result_str = 'sg_vrd_2016_result_'+model_type+'_'+iteration
    result = h5py.File('output/'+result_str+'.hdf5')
    rlp_confs=[]
    rlp_labels=[]
    subbox=[]
    objbox=[]
    relation_vectors = []
    for i in xrange(1000):
        imid = str(m['db/testidx/'+str(i)][...])
        if imid in result:
            objbox.append(result[imid+'/obj_boxes'][...])#\.reshape(-1,4).T)
            subbox.append(result[imid+'/sub_boxes'][...])
            rlp_labels.append(result[imid+'/rlp_labels'][...])
            rlp_confs.append(result[imid+'/rlp_confs'][...].T)
            relation_vectors.append(result[imid+'/relation_vectors'][...])
        else:
            rlp_confs.append([])
            rlp_labels.append([])
            subbox.append([])
            objbox.append([])
            relation_vectors.append([])

    #print objbox
    #objboxx=np.array(objbox)#.astype(np.float64)
    sio.savemat('output/'+result_str+'.mat', {'obj_bboxes_ours': objbox,'sub_bboxes_ours':subbox,
        'rlp_labels_ours':rlp_labels,'rlp_confs_ours':rlp_confs,'relation_vectors':relation_vectors})
    pass
caffe.set_mode_gpu()
caffe.set_device(0)
#make_meta()
#exit(0)

model_type = 'all'
iteration = '19500'
if 'diff' in model_type:
    run_relation_diff(model_type,iteration)
else:
    run_relation(model_type,iteration)
make_relation_result(model_type,iteration)

import caffe
import scipy.io as sio
import os
import cv2
import numpy as np
import yaml
from multiprocessing import Process, Queue
import random
import h5py
import fast_rcnn.bbox_transform

from utils.cython_bbox import bbox_overlaps
import numpy as np
import utils.zl_utils as zl


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


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


class RelationDatalayer(caffe.Layer):
    def get_minibatch(self):
        blobs = {}
        idx = np.random.choice(len(self.rdata['annotation_train']), self._batch_size)

        # labels_blob = np.zeros(self._batch_size,np.int32)
        data = []
        visual = []
        classeme = []
        classeme_s = []
        classeme_o = []
        visual_s = []
        visual_o = []
        loc_s = []
        loc_o = []
        location = []
        labels = []
        cnt = 0
        while cnt < self._batch_size:
            idx = np.random.choice(len(self.rdata['annotation_train']), 1)
            anno = self.rdata['annotation_train'][idx[0]]
            objs = []
            im_id = anno.filename.split('.')[0]
            if im_id not in self.vgg_data:
                continue
            classemes = self.vgg_data[im_id]['classemes']
            visuals = self.vgg_data[im_id]['visuals']
            locations = self.vgg_data[im_id]['locations']
            cls_confs = self.vgg_data[im_id]['cls_confs']

            w, h = self.meta['train/' + im_id + '/w'][...], self.meta['train/' + im_id + '/h'][...]
            if hasattr(anno, 'relationship'):

                if not isinstance(anno.relationship, np.ndarray):
                    anno.relationship = [anno.relationship]
                for r in xrange(len(anno.relationship)):
                    if not hasattr(anno.relationship[r], 'phrase'):
                        continue
                    predicate = anno.relationship[r].phrase[1]
                    ymin, ymax, xmin, xmax = anno.relationship[r].subBox
                    sub_bbox = [xmin, ymin, xmax, ymax]

                    ymin, ymax, xmin, xmax = anno.relationship[r].objBox
                    obj_bbox = [xmin, ymin, xmax, ymax]
                    overlaps = bbox_overlaps(
                        np.ascontiguousarray([sub_bbox, obj_bbox], dtype=np.float),
                        np.ascontiguousarray(locations, dtype=np.float))
                    if overlaps.shape[0] == 0:
                        continue
                    try:
                        assignment = overlaps.argmax(axis=1)
                    except:
                        continue

                    sub_sorted = overlaps[0].argsort()[-30:][::-1]
                    obj_sorted = overlaps[1].argsort()[-30:][::-1]
                    while len(sub_sorted) > 0 and overlaps[0][sub_sorted[-1]] < .7: sub_sorted = sub_sorted[:-1]
                    while len(obj_sorted) > 0 and overlaps[1][obj_sorted[-1]] < .7: obj_sorted = obj_sorted[:-1]

                    if len(sub_sorted) <= 0 or len(obj_sorted) <= 0:
                        continue

                    sub_idx = np.random.choice(len(sub_sorted), 1)
                    obj_idx = np.random.choice(len(obj_sorted), 1)

                    for s in sub_sorted[:1]:  # sub_idx:
                        for o in obj_sorted[:1]:  # obj_idx:
                            if s != o and cnt < self._batch_size:
                                sub_visual = visuals[s]
                                obj_visual = visuals[o]
                                sub_clsmemes = classemes[s]
                                obj_clsmemes = classemes[o]
                                sub_box_encoded = bbox_transform(np.array([locations[o]]), np.array([locations[s]]))[0]
                                obj_box_encoded = bbox_transform(np.array([locations[s]]), np.array([locations[o]]))[0]

                                #sub_box_encoded = bbox_transform(np.array([[0, 0, w, h]]), np.array([locations[s]]))[0]
                                #obj_box_encoded = bbox_transform(np.array([[0, 0, w, h]]), np.array([locations[o]]))[0]
                                relation = self.meta['meta/pre/name2idx/' + predicate][...]
                                labels.append(np.float32(relation))
                                classeme_s.append(sub_clsmemes)
                                classeme_o.append(obj_clsmemes)
                                visual_s.append(sub_visual)
                                visual_o.append(obj_visual)
                                loc_s.append(sub_box_encoded)
                                loc_o.append(obj_box_encoded)
                                #visual.append(np.hstack((sub_visual, obj_visual)))
                                #classeme.append(np.hstack((sub_clsmemes, obj_clsmemes)))
                                location.append(sub_box_encoded)
                                cnt += 1
                    if cnt >= self._batch_size:
                        break
                        # bbox_transform()
        # blobs['visual'] = np.array(visual)
        blobs['classeme_s'] = np.array(classeme_s)
        blobs['classeme_o'] = np.array(classeme_o)
        blobs['visual_s'] = np.array(visual_s)
        blobs['visual_o'] = np.array(visual_o)
        blobs['location_s'] = np.array(loc_s)
        blobs['location_o'] = np.array(loc_o)
        # blobs['classeme'] = np.array(classeme)
        # blobs['location'] = np.array(location)
        blobs['label'] = np.array(labels)

        return blobs

    def setup(self, bottom, top):
        self._cur_idx = 0
        self.rdata = sio.loadmat('/media/zawlin/ssd/data_vrd/vrd/annotation_train.mat', struct_as_record=False,
                                 squeeze_me=True)
        self.vgg_data = h5py.File("output/sg_vrd_2016_train.hdf5", 'r', 'core')
        self.meta = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')
        layer_params = yaml.load(self.param_str_)

        self._batch_size = layer_params['batch_size']
        self.train_data = []
        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        # top[0].reshape(self._batch_size, 4096 * 2 )
        top[0].reshape(self._batch_size, 101)
        top[1].reshape(self._batch_size, 101)

        top[2].reshape(self._batch_size, 4096)
        top[3].reshape(self._batch_size, 4096)
        top[4].reshape(self._batch_size, 4)
        top[5].reshape(self._batch_size, 4)
        # top[1].reshape(self._batch_size, 4)
        top[6].reshape(self._batch_size)
        # self._name_to_top_map['visual'] = 0
        # self._name_to_top_map['classeme'] = 0
        self._name_to_top_map['classeme_s'] = 0
        self._name_to_top_map['classeme_o'] = 1
        self._name_to_top_map['visual_s'] = 2
        self._name_to_top_map['visual_o'] = 3
        self._name_to_top_map['location_s'] = 4
        self._name_to_top_map['location_o'] = 5
        # self._name_to_top_map['location'] = 1
        self._name_to_top_map['label'] = 6

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self.get_minibatch()
        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

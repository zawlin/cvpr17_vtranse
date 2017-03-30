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

from fast_rcnn.nms_wrapper import nms
from utils.cython_bbox import bbox_overlaps
import numpy as np
import utils.zl_utils as zl

from fast_rcnn.bbox_transform import clip_boxes, bbox_transform, bbox_transform_inv

show=True
pause = 1
def visualize_gt(im_data, boxes):
    for j in xrange(len(boxes)):
        di = boxes[j]
        cv2.rectangle(im_data, (di[0], di[1]), (di[2], di[3]), (255, 255, 255), 2)
    pass
def visualize(im_data, boxes_tosort, rpn_boxes, m,thresh_final):
    global show,pause
    if show:
        for j in xrange(len(boxes_tosort)):
            cls_dets = boxes_tosort[j]
            for di in xrange(cls_dets.shape[0]):
                #    print 'here'
                di = cls_dets[di]
                rpn_box = rpn_boxes[di[-1]]
                score = di[-2]
                cls_idx = j + 1
                cls_name = str(m['meta/cls/idx2name/' + str(cls_idx)][...])
                if score > 1:
                    score = 1
                if score < thresh_final:
                    continue
                x, y = int(di[0]), int(di[1])
                if x < 10:
                    x = 15
                if y < 10:
                    y = 15
                cv2.putText(im_data, cls_name, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                cv2.rectangle(im_data, (di[0], di[1]), (di[2], di[3]), (255, 0, 0), 2)
                cv2.rectangle(im_data, (rpn_box[0], rpn_box[1]), (rpn_box[2], rpn_box[3]), (0, 255, 0), 2)

        cv2.imshow("im", im_data)
    c=cv2.waitKey(pause)
    if c == ord('p'):
        pause = 1-pause
    if c == ord('a'):
        show = not show


class RelationSampler(caffe.Layer):
    def setup(self, bottom, top):
        self.rdata = sio.loadmat('/media/zawlin/ssd/data_vrd/vrd/annotation_train.mat', struct_as_record=False,
                                 squeeze_me=True)
        # map im_id to annotation
        self.r_anno = {}
        for i in xrange(len(self.rdata['annotation_train'])):
            anno = self.rdata['annotation_train'][i]
            im_id = anno.filename.split('.')[0]
            self.r_anno[im_id] = anno

        self.meta = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5', 'r', 'core')
        layer_params = yaml.load(self.param_str_)

        self._batch_size = 1
        self.train_data = []
        self._name_to_top_map = {}
        # just hard code it for now
        lines = [line.strip() for line in open('/home/zawlin/g/py-faster-rcnn/data/sg_vrd_2016/ImageSets/train.txt')]
        self._image_id = {int(l.split(' ')[1]): l.split(' ')[0] for l in lines}

        # data blob: holds a batch of N images, each with 3 channels
        top[0].reshape(self._batch_size, 101)
        top[1].reshape(self._batch_size, 101)
        top[2].reshape(self._batch_size, 4)
        top[3].reshape(self._batch_size, 4)
        top[4].reshape(self._batch_size, 4)
        top[5].reshape(self._batch_size, 4)
        top[6].reshape(self._batch_size, 1)

        self._name_to_top_map['s_classeme'] = 0
        self._name_to_top_map['o_classeme'] = 1
        self._name_to_top_map['s_rois'] = 2
        self._name_to_top_map['o_rois'] = 3
        self._name_to_top_map['s_rois_encoded'] = 4
        self._name_to_top_map['o_rois_encoded'] = 5
        self._name_to_top_map['relation_label'] = 6
        self._prev_blob = None

        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        # prep incoming data==========
        rpn_boxes = bottom[0].data.copy()
        bbox_pred = bottom[1].data
        scores = bottom[2].data
        im_info = bottom[3].data[0]
        im_idx = int(bottom[4].data)
        im_data = bottom[5].data[0, :, :, :].transpose((1, 2, 0)).copy()
        m = self.meta
        im_id = self._image_id[im_idx]
        r_anno = self.r_anno[im_id]
        # prep done============

        # prep blobs for forward
        blobs = {}
        s_classeme = []
        s_rois = []
        s_rois_encoded = []
        o_classeme = []
        o_rois = []
        o_rois_encoded = []
        relation_label = []

        gt_boxes = []
        if hasattr(r_anno, 'relationship'):
            rpn_boxes_img_coor = rpn_boxes[:, 1:5] / im_info[2]
            boxes = rpn_boxes_img_coor
            boxes = bbox_transform_inv(boxes, bbox_pred)
            boxes = clip_boxes(boxes, (im_info[0] / im_info[2], im_info[1] / im_info[2]))

            cv2.normalize(im_data, im_data, 255, 0, cv2.NORM_MINMAX)
            im_data = im_data.astype(np.uint8)

            origsz = (im_info[1] / im_info[2], im_info[0] / im_info[2])
            im_data = cv2.resize(im_data, origsz)
            thresh_final = .5

            res_locations = []
            res_classemes = []
            res_cls_confs = []
            boxes_tosort = []
            for j in xrange(1, 101):
                inds = np.where(scores[:, j] > .3)[0]
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis], inds[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
                # pred_boxes = clip_boxes(pred_boxes, im.shape)
                if len(cls_scores) <= 0:
                    boxes_tosort.append(cls_dets)
                    continue

                res_loc = np.hstack((cls_boxes, inds[:, np.newaxis]))
                res_classeme = scores[inds]
                res_cls_conf = np.column_stack((np.zeros(cls_scores.shape[0]) + j, cls_scores))
                keep = nms(cls_dets[:,:5], .3)  # nms threshold
                cls_dets = cls_dets[keep, :]
                res_loc = res_loc[keep]
                res_classeme = res_classeme[keep]
                res_cls_conf = res_cls_conf[keep]
                res_classemes.extend(res_classeme)
                res_locations.extend(res_loc)
                res_cls_confs.extend(res_cls_conf)
                boxes_tosort.append(cls_dets)
            try:
                # final class confidence
                inds = np.where(np.array(res_cls_confs)[:, 1] > thresh_final)[0]

                classemes = np.array(res_classemes)[inds]
                locations = np.array(res_locations)[inds]
                cls_confs = np.array(res_cls_confs)[inds]
                # decide what to pass to top

                # limit max
                w, h = self.meta['train/' + im_id + '/w'][...], self.meta['train/' + im_id + '/h'][...]
                if not isinstance(r_anno.relationship, np.ndarray):
                    r_anno.relationship = [r_anno.relationship]
                for r in xrange(len(r_anno.relationship)):
                    if not hasattr(r_anno.relationship[r], 'phrase'):
                        continue
                    predicate = r_anno.relationship[r].phrase[1]
                    ymin, ymax, xmin, xmax = r_anno.relationship[r].subBox
                    sub_bbox = [xmin, ymin, xmax, ymax]
                    gt_boxes.append(sub_bbox)

                    ymin, ymax, xmin, xmax = r_anno.relationship[r].objBox

                    obj_bbox = [xmin, ymin, xmax, ymax]
                    gt_boxes.append(obj_bbox)
                    overlaps = bbox_overlaps(
                        np.ascontiguousarray([sub_bbox, obj_bbox], dtype=np.float),
                        np.ascontiguousarray(locations, dtype=np.float))
                    if overlaps.shape[0] == 0:
                        continue

                    sub_sorted = overlaps[0].argsort()[-40:][::-1]
                    obj_sorted = overlaps[1].argsort()[-40:][::-1]
                    while len(sub_sorted) > 0 and overlaps[0][sub_sorted[-1]] < .6: sub_sorted = sub_sorted[:-1]
                    while len(obj_sorted) > 0 and overlaps[1][obj_sorted[-1]] < .6: obj_sorted = obj_sorted[:-1]

                    if len(sub_sorted) <= 0 or len(obj_sorted) <= 0:
                        continue

                    cnt = 0
                    for s in sub_sorted[:1]:  # sub_idx:
                        for o in obj_sorted[:1]:  # obj_idx:
                            if s != o and cnt < 20:
                                sub_clsmemes = classemes[s]
                                obj_clsmemes = classemes[o]
                                sub_box_encoded = bbox_transform(np.array([[0, 0, w, h]]), np.array([locations[s]]))[0]
                                obj_box_encoded = bbox_transform(np.array([[0, 0, w, h]]), np.array([locations[o]]))[0]
                                relation = self.meta['meta/pre/name2idx/' + predicate][...]
                                # all done, now we put forward
                                s_classeme.append(sub_clsmemes)
                                o_classeme.append(obj_clsmemes)
                                s_rois.append(rpn_boxes[locations[s][-1]])
                                o_rois.append(rpn_boxes[locations[o][-1]])
                                s_rois_encoded.append(sub_box_encoded)
                                o_rois_encoded.append(obj_box_encoded)
                                relation_label.append(np.float32(relation))
                                cnt += 1
                # final step copy all the stuff for forward
                blobs['s_classeme'] = np.array(s_classeme)
                blobs['o_classeme'] = np.array(o_classeme)
                blobs['s_rois'] = np.array(s_rois)
                blobs['o_rois'] = np.array(o_rois)
                blobs['s_rois_encoded'] = np.array(s_rois_encoded)
                blobs['o_rois_encoded'] = np.array(o_rois_encoded)
                blobs['relation_label'] = np.array(relation_label)
            except:
                blobs = self._prev_blob
            if blobs['s_classeme'].shape[0] == 0:
                blobs = self._prev_blob
        else:
            blobs = self._prev_blob
        visualize_gt(im_data,gt_boxes)
        visualize(im_data, boxes_tosort, rpn_boxes_img_coor, m,thresh_final)
        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

        # this becomes a dummy for forward in case things fail
        if blobs['relation_label'][0] != -1:
            for blob_name, blob in blobs.iteritems():
                blobs[blob_name] = blob[0, np.newaxis]
                if blob_name == 'relation_label':
                    blobs[blob_name][...] = -1
        self._prev_blob = blobs

    def backward(self, top, propagate_down, bottom):
        #print propagate_down
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

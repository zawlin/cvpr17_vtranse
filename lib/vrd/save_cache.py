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

from fast_rcnn.config import cfg, get_output_dir
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


def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels


def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data': None, 'rois': None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors


def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    # if cfg.TEST.BBOX_REG:
    if True:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]
    fc7 = net.blobs['fc7'].data
    return net.blobs['cls_score'].data[:, :], scores, fc7, pred_boxes


def save_result(split):
    caffe.set_mode_gpu()
    caffe.set_device(0)
    m = h5py.File('data/sg_vrd_meta.h5', 'r', 'core')

    net = caffe.Net('models/vrd/vgg16/faster_rcnn_end2end/test.prototxt',
                    'models/vr_obj_model.caffemodel',
                    caffe.TEST)
    net.name = 'sgvrd'
    imdb = get_imdb('sg_vrd_2016_%s'%split)
    imdb.competition_mode(0)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    h5path = 'output/sg_vrd_2016_%s.hdf5'%split

    # if os.path.exists(h5path):
    # os.remove(h5path)
    h5f = h5py.File(h5path)
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    root = 'data/sg_vrd_2016/Data/sg_%s_images/'%split
    _t = {'im_detect': Timer(), 'misc': Timer()}
    cnt = 0
    thresh = .15
    for path, subdirs, files in os.walk(root):
        for name in files:
            cnt += 1
            im_idx = name.split('.')[0]
            fpath = os.path.join(path, name)
            im = cv2.imread(fpath)
            if im == None:
                print fpath
            box_proposals = None
            _t['im_detect'].tic()
            score_raw, scores, fc7, boxes = im_detect(net, im, box_proposals)
            _t['im_detect'].toc()
            # scores = score_raw
            res_locations = []
            res_visuals = []
            res_classemes = []
            res_cls_confs = []
            boxes_tosort = []
            _t['misc'].tic()
            for j in xrange(1, 101):
                inds = np.where(scores[:, j] > 0.01)[0]
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, j * 4:(j + 1) * 4]

                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)

                if len(cls_scores) <= 0:
                    boxes_tosort.append(cls_dets)
                    continue

                res_loc = cls_boxes
                res_vis = fc7[inds]
                res_classeme = scores[inds]
                res_cls_conf = np.column_stack((np.zeros(cls_scores.shape[0]) + j, cls_scores))

                keep = nms(cls_dets, .2, force_cpu=True)  # nms threshold
                cls_dets = cls_dets[keep, :]

                res_loc = res_loc[keep]
                res_vis = res_vis[keep]
                res_classeme = res_classeme[keep]
                res_cls_conf = res_cls_conf[keep]

                res_classemes.extend(res_classeme)
                res_visuals.extend(res_vis)
                res_locations.extend(res_loc)
                res_cls_confs.extend(res_cls_conf)

                boxes_tosort.append(cls_dets)
            # filter based on confidence
            inds = np.where(np.array(res_cls_confs)[:, 1] > thresh)[0]

            res_classemes = np.array(res_classemes)[inds]
            res_visuals = np.array(res_visuals)[inds]
            res_locations = np.array(res_locations)[inds]
            res_cls_confs = np.array(res_cls_confs)[inds]

            h5f.create_dataset(im_idx + '/classemes', dtype='float16', data=res_classemes.astype(np.float16))
            h5f.create_dataset(im_idx + '/visuals', dtype='float16', data=res_visuals.astype(np.float16))
            h5f.create_dataset(im_idx + '/locations', dtype='short', data=res_locations.astype(np.short))
            h5f.create_dataset(im_idx + '/cls_confs', dtype='float16', data=res_cls_confs.astype(np.float16))

            _t['misc'].toc()

            print 'im_detect: {:d} {:.3f}s {:.3f}s' \
                .format(cnt, _t['im_detect'].average_time,
                        _t['misc'].average_time)
save_result('test')
save_result('train')


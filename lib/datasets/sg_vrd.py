# --------------------------------------------------------
# Fast R-CNN for ILSVRC VIDEO DETECTION
# zawlin
# --------------------------------------------------------

import datasets
import datasets.pascal_voc
import os
from datasets.imdb import imdb
import xml.dom.minidom as minidom
from xml.etree.ElementTree import Element
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import h5py
import PIL
from fast_rcnn.config import cfg


class sg_vrd(imdb):
    def __init__(self, image_set, year):
        imdb.__init__(self, 'sg_vrd_' + year + '_' + image_set)
        #self._image_set_path = 'imagenet_' + year + '_' + image_set
        self._year = year
        self._image_set = image_set
        self._folder_path = self._image_set
        #to handle splitted datasets so that we don't have to symlink for every splits
        if 'test' in self._image_set:self._folder_path = 'sg_test_images'
        if 'train' in self._image_set:self._folder_path = 'sg_train_images'

        self._devkit_path = os.path.join(self._get_default_path(), 'devkit')
        self._data_path = os.path.join(self._get_default_path(), 'Data', self._folder_path)
        self._annot_path = os.path.join(self._get_default_path(), 'Annotations', self._folder_path)

        self._classes = ('__background__',)  # always index 0
        self._class_name = ('__background__',)  # always index 0
        self._class_ids = ('__background__',)  # always index 0
        objectListN = sio.loadmat(os.path.join(self._devkit_path, 'objectListN.mat'))
        objectListN = objectListN['objectListN'].squeeze()
        for i in range(100):
            self._classes += (str(objectListN[i][0]),)
            self._class_name += (str(objectListN[i][0]),)
            self._class_ids += (str(objectListN[i][0]),)
        self._class_to_ind = dict(zip(self._classes, xrange(self.num_classes)))
        self._class_to_ind.update(dict(zip(self._class_ids, xrange(self.num_classes))))
        self._image_ext = '.jpg'
        self._image_index, self._image_id = self._load_image_set_index()
        self._wh = self._load_image_width_height()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'top_k': 2000,
                       'use_diff': False,
                       'rpn_file': None}
        assert os.path.exists(self._data_path), \
            'sg_vrd data path does not exist: {}'.format(self._data_path)
        if self._image_set != 'test':
            assert os.path.exists(self._annot_path), \
                'sg_vrd annotation path does not exist: {}'.format(self._annot_path)

    def _get_default_path(self):
        """
        Return the default path where IMAGENET is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'sg_vrd_' + self._year)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_index_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_no_roi_files(self):
        no_roi_file = os.path.join(self._get_default_path(), 'no_roi_files.txt')
        try:
            return [line.strip() for line in open(no_roi_file)]
        except:
            return []

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file. (val, test)
        """

        image_set_file = os.path.join(self._get_default_path(), 'ImageSets',
                                      self._image_set + '.txt')
        '''
        if self._image_set=='train':
            image_set_file = os.path.join(self._get_default_path(), 'ImageSets',
                                          self._image_set + '_zl.txt')
        else:

            image_set_file = os.path.join(self._get_default_path(), 'ImageSets',
                                          self._image_set + '.txt')
        '''
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.split()[0] for x in f.readlines()]
        with open(image_set_file) as f:
            image_id = [x.split()[1] for x in f.readlines()]
        return image_index, image_id

    def _load_image_width_height(self):
        cache_file = os.path.join(self.cache_path, self.name + '_img_wh.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                img_wh = cPickle.load(fid)
            print '{} image wh loaded from {}'.format(self.name, cache_file)
            return img_wh
        img_wh = []
        for index in self._image_index:
            wh = self.load_image_wh(index)
            img_wh.append(wh)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(img_wh, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote image wh to {}'.format(cache_file)
        return img_wh

    def load_image_wh(self, index):
        """
        Load the width and height
        """
        if self._image_set != 'test':
            filename = os.path.join(self._annot_path, index + '.xml')
            assert os.path.exists(filename), \
                'Path does not exist: {}'.format(filename)

            def get_data_from_tag(node, tag):
                return node.getElementsByTagName(tag)[0].childNodes[0].data

            with open(filename) as f:
                data = minidom.parseString(f.read())

            size = data.getElementsByTagName('size')
            iw = float(get_data_from_tag(size[0], 'width'))
            ih = float(get_data_from_tag(size[0], 'height'))
            out = (iw, ih)
        else:
            filename = os.path.join(self._data_path, index + '.jpg')
            print filename
            assert os.path.exists(filename), \
                'Path does not exist: {}'.format(filename)
            out = PIL.Image.open(filename).size
        return out

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            # print 'removing invlaid roi images'

            # ind2remove=[i for i, x in enumerate(roidb) if x == None]
            # self._image_index= [x for i,x in enumerate(self._image_index) if i not in ind2remove]
            # roidb= [x for i,x in enumerate(self.roidb) if i not in ind2remove]
            assert len(roidb) == len(self._image_index)
            return roidb
        gt_roidb = [self._load_imagenet_annotation(index)
                    for index in self.image_index]
        assert len(gt_roidb) == len(self._image_index)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the IMAGENET
        format.
        """
        filename = os.path.join(self._annot_path, index + '.xml')

        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        objs_filtered=[]
        for o in objs:
            if str(get_data_from_tag(o, "name")) in self._class_to_ind:
                objs_filtered.append(o)
        objs = objs_filtered
        size = data.getElementsByTagName('size')
        iw = float(get_data_from_tag(size[0], 'width'))
        ih = float(get_data_from_tag(size[0], 'height'))
        num_objs = len(objs)
        if num_objs == 0:
            cache_file = os.path.join(self.cache_path, 'no_roi_files.txt')
            output = open(cache_file, 'a')
            output.write(index + '\n')
            print 'no objects in gt xml ' + filename
            return None
        if iw<100 or ih<100:
            print 'image width or height too small' + filename
            cache_file = os.path.join(self.cache_path, 'small_images.txt')
            output = open(cache_file, 'a')
            output.write(index + '\n')
            return None
        assert num_objs != 0, \
            'No objects in ground truth information ' + filename

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            cls = self._class_to_ind[
                str(get_data_from_tag(obj, "name"))]
            # to avoid wrong annotation
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            # exception ( 1-based annotation --> 0-based )
            if x2 >= iw:
                x2 = iw - 1
            if y2 >= ih:
                y2 = ih - 1
            if x2 <= x1 or y2 <= y1:  # can't define bbox
                print index
                assert False, \
                    'Cannot define bounding box'

            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        if len(overlaps)<=0:
            print index
            print 'here'
        if len(gt_classes)<=0:
            print index

        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False}

    def rpn_roidb(self):
        if self._image_set == 'train':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set == 'train':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):

        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search.pkl')
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                box_list = cPickle.load(fid)
            print '{} selective search loaded from {}'.format(self.name, cache_file)
        else:
            ss_data = h5py.File(filename)
            box_list = []
            for i in range(ss_data['boxes'].shape[1]):
                if i % 1000 == 0:
                    print '[LOADING SS BOXES] %d th image...' % (i + 1)
                tmp = [ss_data[element[i]][:] for element in ss_data['boxes']]
                tmp = tmp[0].transpose()
                box_list.append(tmp[:, (1, 0, 3, 2)] - 1)

            with open(cache_file, 'wb') as fid:
                cPickle.dump(box_list, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote selective search bboxes to  {}'.format(cache_file)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _write_imagenet_results_file(self, all_boxes, output_dir):
        filename = output_dir + '/vid_' + self._image_set + '.txt'
        if os.path.exists(filename):
            return filename
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self._image_index):
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{} {} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(self._image_id[im_ind], cls_ind, dets[k, -1],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3]))
        print 'Writing IMAGENET VID results file: {}'.format(filename)
        return filename

    def _do_matlab_eval(self, filename):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'ILSVRCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'ilsvrc_vid_eval(\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, filename)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        filename = self._write_imagenet_results_file(all_boxes, output_dir)
        # self._do_matlab_eval(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True



# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.sg_vrd import sg_vrd
from datasets.vg1_2 import vg1_2
import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))


# Set up ilsvrc_2016_<split>
for year in ['2016']:
    for split in ['train', 'val','test','train_mini']:
        name = 'ilsvrc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: imagenet(split,year))

for year in ['2016']:
    for split in ['train', 'test','mini']:
        name = 'sg_vrd_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: sg_vrd(split,year))

for year in ['2016']:
    for split in ['train', 'test','mini']:
        name = 'vg1_2_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: vg1_2(split,year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

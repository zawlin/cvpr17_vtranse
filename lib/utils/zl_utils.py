import cPickle
import time
import os
import numpy as np
import operator
import shutil
import cv2
def put_text(img, text, org, color, thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, lineType=8,
             buttomLeftOrigin=False, alpha=1.0):
    overlay = img.copy()
    output = img.copy()
    newx=org[0]
    newy=org[1]
    pad = 25
    if newx<pad:newx=pad
    if newy<pad:newy=pad

    cv2.putText(overlay, text, (newx,newy), fontFace, fontScale, color, thickness)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                    0, output)

    img[...] = output
    pass


def rectangle(img, pt1, pt2, color, thickness=1, lineType=8, shift=0, alpha=1.0):
    overlay = img.copy()
    output = img.copy()

    cv2.rectangle(overlay, pt1, pt2, color, thickness, lineType, shift)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                    0, output)
    img[...] = output


def circle():
    pass

def load(path):
    with open(path) as f:
        return cPickle.load(f)

def save(path,obj):
    with open(path,'wb') as f:
        cPickle.dump(obj,f,cPickle.HIGHEST_PROTOCOL)
_time_last = 0
def tick():
    global _time_last
    _time_last=time.time()

def tock():
    return time.time()-_time_last


def files(root):
    ret = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            fpath = os.path.join(path, name)
            ret.append(fpath)
    ret.sort()
    return ret

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    return data

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    return data

_meta_name2idx_cls = None
_meta_idx2name_cls = None
_meta_name2idx_pre = None
_meta_idx2name_pre = None
_meta_name2idx_tri = None
_meta_idx2name_tri = None

def _init_meta(m):
    global _meta_name2idx_cls
    global _meta_idx2name_cls
    global _meta_name2idx_pre
    global _meta_idx2name_pre
    _meta_name2idx_cls ={}
    _meta_idx2name_cls ={}
    _meta_name2idx_pre ={}
    _meta_idx2name_pre ={}
    for k in m['meta/cls/name2idx'].keys():
        idx = int(str(m['meta/cls/name2idx/'+k][...]))
        _meta_name2idx_cls[k] = idx
        _meta_idx2name_cls[idx] = k
    for k in m['meta/pre/name2idx'].keys():
        idx = int(str(m['meta/pre/name2idx/'+k][...]))
        _meta_name2idx_pre[k] = idx
        _meta_idx2name_pre[idx] = k

def _init_meta_tri(m):
    global _meta_name2idx_tri
    global _meta_idx2name_tri
    _meta_name2idx_tri ={}
    _meta_idx2name_tri ={}
    for k in m['meta/tri/name2idx'].keys():
        idx = int(str(m['meta/tri/name2idx/'+k][...]))
        _meta_name2idx_tri[k] = idx
        _meta_idx2name_tri[idx] = k

def name2idx_tri(m,name):
    if _meta_name2idx_tri==None:
        _init_meta_tri(m)
    return _meta_name2idx_tri[name]

def idx2name_tri(m,idx):
    if _meta_name2idx_tri==None:
        _init_meta_tri(m)
    return _meta_idx2name_tri[idx]

def name2idx_cls(m,name):
    if _meta_name2idx_cls == None:
        _init_meta(m)
    return _meta_name2idx_cls[name]

def name2idx_pre(m,name):
    if _meta_name2idx_cls == None:
        _init_meta(m)
    return _meta_name2idx_pre[name]

def idx2name_cls(m,idx):
    if _meta_idx2name_cls == None:
        _init_meta(m)
    return _meta_idx2name_cls[idx]

def idx2name_pre(m,idx):
    if _meta_idx2name_pre== None:
        _init_meta(m)
    return _meta_idx2name_pre[idx]

def imid2path(m,imid):
    return str(m['meta/imid2path/%s'%imid][...])

def unique_arr(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return a[idx]

def sort_dict_by_val(dic):
    return sorted(dic.items(), key=operator.itemgetter(1),reverse=True)

def sort_dict_by_key_val(dic,key):
    return dic.sort(key=operator.itemgetter(key))

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy_file(src,dst):
    shutil.copyfile(src,dst)


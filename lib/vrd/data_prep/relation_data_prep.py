import _init_paths
import cv2
import numpy as np
import os
import scipy.io as sio
from dict2xml import dict2xml
import utils.zl_utils as zl
import h5py
data_root = "/media/zawlin/ssd/data_vrd/vrd/sg/Data"
anno_root = "/media/zawlin/ssd/data_vrd/vrd/sg/Annotations"

def prep_test():
    data = sio.loadmat('/media/zawlin/ssd/data_vrd/vrd/annotation_test.mat', struct_as_record=False, squeeze_me=True)
    cnt=0
    for i in xrange(len(data['annotation_test'])):
        if i%100==0:
            print i
        anno = data['annotation_test'][i]
        voc_datum = {"folder": "sg_test_images",
                     "source": {"database":"sg_vrd_stanford"},
                     "filename":anno.filename
                     }

        im = cv2.imread(os.path.join(data_root, voc_datum["folder"], voc_datum["filename"]))
        if im == None:
            continue
            print os.path.join(data_root, voc_datum["folder"], voc_datum["filename"])
        w, h = im.shape[1], im.shape[0]
        voc_datum['size']={'width':w,'height':h}

        objs = []

        if hasattr(anno,'relationship'):
            cnt+=1
            if not isinstance(anno.relationship,np.ndarray):
                anno.relationship= [anno.relationship]
            for r in xrange(len(anno.relationship)):
                # [ymin, ymax, xmin, xmax]
                ymin, ymax, xmin, xmax = anno.relationship[r].subBox
                bbox = {'ymin':ymin,'ymax':ymax,'xmin':xmin,'xmax':xmax}
                obj = {'name':anno.relationship[r].phrase[0],
                       'bndbox':bbox}
                objs.append(obj)

                ymin, ymax, xmin, xmax = anno.relationship[r].objBox
                bbox = {'ymin':ymin,'ymax':ymax,'xmin':xmin,'xmax':xmax}
                obj = {'name':anno.relationship[r].phrase[2],
                       'bndbox':bbox}
                objs.append(obj)
            voc_datum['object']=objs


            #write to xml
            dst_path = os.path.join(anno_root,voc_datum["folder"], voc_datum["filename"][:voc_datum["filename"].rfind('.')]+'.xml')
            voc_datum={'annotation':voc_datum}
            f = open(dst_path,'w')
            f.write(dict2xml(voc_datum)+'\n')
            f.close()
    print 'images with annotation=%d\n'%cnt

def count_taller_occurrence():
    data = sio.loadmat('/media/zawlin/ssd/data_vrd/vrd/annotation_train.mat', struct_as_record=False, squeeze_me=True)
    cnt=0
    taller_cnt =0
    for i in xrange(len(data['annotation_train'])):
        if i%100==0:
            print i
        anno = data['annotation_train'][i]
        voc_datum = {"folder": "sg_train_images",
                     "source": {"database":"sg_vrd_stanford"},
                     "filename":anno.filename
                     }

        if hasattr(anno,'relationship'):
            cnt+=1
            if not isinstance(anno.relationship,np.ndarray):
                anno.relationship= [anno.relationship]
            for r in xrange(len(anno.relationship)):
                if 'taller' in  anno.relationship[r].phrase[1]:
                    taller_cnt +=1
    print 'taller cnt =%d\n'%taller_cnt

def prep_train():
    data = sio.loadmat('/media/zawlin/ssd/data_vrd/vrd/annotation_train.mat', struct_as_record=False, squeeze_me=True)
    cnt=0
    taller_cnt =0
    for i in xrange(len(data['annotation_train'])):
        if i%100==0:
            print i
        anno = data['annotation_train'][i]
        voc_datum = {"folder": "sg_train_images",
                     "source": {"database":"sg_vrd_stanford"},
                     "filename":anno.filename
                     }

        im = cv2.imread(os.path.join(data_root, voc_datum["folder"], voc_datum["filename"]))
        if im == None:
            continue
            print os.path.join(data_root, voc_datum["folder"], voc_datum["filename"])

        w, h = im.shape[1], im.shape[0]
        voc_datum['size']={'width':w,'height':h}

        objs = []

        if hasattr(anno,'relationship'):
            cnt+=1
            if not isinstance(anno.relationship,np.ndarray):
                anno.relationship= [anno.relationship]
            for r in xrange(len(anno.relationship)):
                if 'taller' in  anno.relationship[r].phrase[1]:
                    taller_cnt +=1
                # [ymin, ymax, xmin, xmax]
                ymin, ymax, xmin, xmax = anno.relationship[r].subBox
                bbox = {'ymin':ymin,'ymax':ymax,'xmin':xmin,'xmax':xmax}

                obj = {'name':anno.relationship[r].phrase[0],
                       'bndbox':bbox}
                objs.append(obj)

                ymin, ymax, xmin, xmax = anno.relationship[r].objBox
                bbox = {'ymin':ymin,'ymax':ymax,'xmin':xmin,'xmax':xmax}

                obj = {'name':anno.relationship[r].phrase[2],
                       'bndbox':bbox}
                objs.append(obj)
            voc_datum['object']=objs


            #write to xml
            dst_path = os.path.join(anno_root,voc_datum["folder"], voc_datum["filename"][:voc_datum["filename"].rfind('.')]+'.xml')
            voc_datum={'annotation':voc_datum}
            f = open(dst_path,'w')
            f.write(dict2xml(voc_datum)+'\n')
            f.close()
    print 'images with annotation=%d\n'%cnt
    print 'taller cnt =%d\n'%taller_cnt

def prep_train_data_only():
    data = sio.loadmat('/media/zawlin/ssd/data_vrd/vrd/annotation_train.mat', struct_as_record=False, squeeze_me=True)
    cnt=0
    for i in xrange(len(data['annotation_train'])):
        if i%100==0:
            print i
        anno = data['annotation_train'][i]
        objs = []

        if hasattr(anno,'relationship'):
            cnt+=1
            if not isinstance(anno.relationship,np.ndarray):
                anno.relationship= [anno.relationship]
            for r in xrange(len(anno.relationship)):
                print anno.filename
                print anno.relationship[r].phrase[1]
                ymin, ymax, xmin, xmax = anno.relationship[r].subBox
                bbox = {'ymin':ymin,'ymax':ymax,'xmin':xmin,'xmax':xmax}

                obj = {'name':anno.relationship[r].phrase[0],
                       'bndbox':bbox}
                objs.append(obj)

                ymin, ymax, xmin, xmax = anno.relationship[r].objBox
                bbox = {'ymin':ymin,'ymax':ymax,'xmin':xmin,'xmax':xmax}

                obj = {'name':anno.relationship[r].phrase[2],
                       'bndbox':bbox}
                objs.append(obj)

def gen_image_sets():

    root = '/home/zawlin/g/py-faster-rcnn/data/sg_vrd_2016/Annotations/sg_train_images'
    f = open('/home/zawlin/g/py-faster-rcnn/data/sg_vrd_2016/ImageSets/train.txt','w')
    cnt = 0
    for path, subdirs, files in os.walk(root):
        for name in files:
            cnt+=1
            f.write('%s %d\n'%(name[:name.rfind('.')],cnt))
    f.close()


    root = '/home/zawlin/g/py-faster-rcnn/data/sg_vrd_2016/Annotations/sg_test_images'
    f = open('/home/zawlin/g/py-faster-rcnn/data/sg_vrd_2016/ImageSets/test.txt','w')
    cnt=0
    for path, subdirs, files in os.walk(root):
        for name in files:
            cnt+=1
            f.write('%s %d\n'%(name[:name.rfind('.')],cnt))
    f.close()

def ilsvrc_meta():
    synsets = sio.loadmat('/media/zawlin/hydra/mnt/disk_05/zawlin/data/ILSVRC/devkit/data/meta_vid.mat')

    synsets = synsets['synsets'].squeeze()
    name2id ={}
    id2name={}
    idx2name={}
    idx2name['0']='__background__'
    name2idx={}
    name2idx['__background__']='0'
    for i in range(30):
        idx2name[str(i+1)]=str(synsets[i][2][0])
        name2idx[str(synsets[i][2][0])]=str(i+1)
        name2id[str(synsets[i][2][0])] = str(synsets[i][1][0])
        id2name[str(synsets[i][1][0])] = str(synsets[i][2][0])

    lines = [line.strip() for line in open('/home/zawlin/data/ILSVRC/ImageSets/VID/val.txt')]
    h5f = '/home/zawlin/Dropbox/proj/ilsvrc_meta.h5'
    h5f = h5py.File(h5f)
    for k in name2id.keys():
        h5f['meta/name2id/'+k]=np.string_(name2id[k])
    for k in idx2name.keys():
        h5f['meta/idx2name/'+k]=np.string_(idx2name[k])
    for k in id2name.keys():
        h5f['meta/id2name/'+k]=np.string_(id2name[k])
    for k in name2idx.keys():
        h5f['meta/name2idx/'+k]=np.string_(name2idx[k])

    for l in lines:
        s = l.split(' ')
        h5f['db/val/'+s[0]]=np.string_(s[1])

def vrd_meta():

    objectListN = sio.loadmat(os.path.join('/media/zawlin/ssd/data_vrd/vrd/', 'objectListN.mat'))
    objectListN = objectListN['objectListN'].squeeze()

    classes = ('__background__',)  # always index 0
    for i in range(100):
        classes += (str(objectListN[i][0]),)
    h5f = '/home/zawlin/Dropbox/proj/sg_vrd_meta.h5'
    h5f = h5py.File(h5f)
    for i in range(101):
        h5f['meta/cls/name2idx/'+classes[i]] = str(i)
        h5f['meta/cls/idx2name/'+str(i)] = classes[i]


    predicate = sio.loadmat(os.path.join('/media/zawlin/ssd/data_vrd/vrd/', 'predicate.mat'))
    predicate = predicate['predicate'].squeeze()

    predicates=()
    for i in range(70):
        predicates += (str(predicate[i][0]),)

    for i in range(70):
        h5f['meta/pre/name2idx/'+predicates[i]] = str(i)
        h5f['meta/pre/idx2name/'+str(i)] = predicates[i]
    root = '/media/zawlin/ssd/data_vrd/vrd/sg/Data/sg_train_images'

    for path, subdirs, files in os.walk(root):
        for name in files:
            fpath = os.path.join(path, name)
            im_id = name.split('.')[0]
            im = cv2.imread(fpath)
            h5f['train/'+im_id+'/h']=im.shape[0]
            h5f['train/'+im_id+'/w']=im.shape[1]

def vrd_meta_add_predicate_types():
    h5f = '/home/zawlin/Dropbox/proj/sg_vrd_meta.h5'
    h5f = h5py.File(h5f)

    lines = [line.strip() for line in open('/media/zawlin/ssd/Dropbox/cvpr17/_relation_mappings/vrd_predicates.txt')]
    type_mappings={}
    for l in lines:
        ls = [i.strip() for i in l.split(',') if i.strip() != '']
        type_mappings[ls[0]]=ls[1]
    print type_mappings
    for k in h5f['meta/pre/name2idx/']:
        h5f['meta/pre/name2idx/'+k].attrs['type']=type_mappings[k]

def vrd_generate_type_idx():
    h5f = '/home/zawlin/Dropbox/proj/sg_vrd_meta.h5'
    h5f = h5py.File(h5f)
    v = []
    p =[]
    s = []
    c=[]
    for k in h5f['meta/pre/name2idx/']:
        if h5f['meta/pre/name2idx/'+k].attrs['type']=='v':
            v.append(str(h5f['meta/pre/name2idx/'+k][...]))
        if h5f['meta/pre/name2idx/'+k].attrs['type']=='p':
            p.append(str(h5f['meta/pre/name2idx/'+k][...]))
        if h5f['meta/pre/name2idx/'+k].attrs['type']=='s':
            s.append(str(h5f['meta/pre/name2idx/'+k][...]))
        if h5f['meta/pre/name2idx/'+k].attrs['type']=='c':
            c.append(str(h5f['meta/pre/name2idx/'+k][...]))
    print 'v= ' +str(v)
    print 'p= ' +str(p)
    print 's= ' +str(s)
    print 'c= ' +str(c)

def prep_train_data():
    'sg_vrd_2016_train.hdf5'
    vgg_data = h5py.File('output/sg_vrd_2016_train.hdf5')
    for im_id in vgg_data.keys():

        boxes = vgg_data[im_id]['boxes']
        scores = vgg_data[im_id]['scores']
        score_max = np.argmax(scores, axis=1)
        max_boxes = []
        for ii in xrange(len(score_max)):
            j = score_max[ii]
            max_boxes.append(boxes[ii, j * 4:(j + 1) * 4])
        max_boxes=np.array(max_boxes).flatten()
        vgg_data.create_dataset(im_id + '/max_boxes',dtype='short', data=max_boxes.astype(np.short))
        #max_boxes.reshape((-1,4))
def add_maxbox():
    'sg_vrd_2016_train.hdf5'
    vgg_data = h5py.File('output/sg_vrd_2016_test.hdf5')
    for im_id in vgg_data.keys():

        boxes = vgg_data[im_id]['boxes']
        scores = vgg_data[im_id]['scores']
        score_max = np.argmax(scores, axis=1)
        max_boxes = []
        for ii in xrange(len(score_max)):
            j = score_max[ii]
            max_boxes.append(boxes[ii, j * 4:(j + 1) * 4])
        max_boxes=np.array(max_boxes).flatten()
        vgg_data.create_dataset(im_id + '/max_boxes',dtype='short', data=max_boxes.astype(np.short))
        #max_boxes.reshape((-1,4))
def vr_make_meta_visual_phrase():
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5','r',driver='core')
    h5f  = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_vp_meta.h5')

    triplets = {}
    cnt = 0
    zl.tick()
    for k in m['gt/train'].keys():
        if cnt %1000==0:
            print cnt,zl.tock()
            zl.tick()
        cnt+=1
        # sub_boxes = m['gt/train/%s/sub_boxes'%k][...]
        # obj_boxes = m['gt/train/%s/obj_boxes'%k][...]
        rlp_labels = m['gt/train/%s/rlp_labels'%k][...]
        for i in xrange(rlp_labels.shape[0]):
            rlp_label = rlp_labels[i]

            s_lbl = zl.idx2name_cls(m,rlp_label[0])
            o_lbl = zl.idx2name_cls(m,rlp_label[2])
            p_lbl = zl.idx2name_pre(m,rlp_label[1])

            spo = '%s_%s_%s'%(s_lbl,p_lbl,o_lbl)
            # spo = '%d_%d_%d'%(rlp_label[0],rlp_label[1],rlp_label[2])
            if spo not in triplets:
                triplets[spo]=0
            triplets[spo]+=1
    zl.save('output/pkl/triplets_train.pkl',triplets)
    triplets_sorted = zl.sort_dict_by_val(triplets)

    triplets_ok = []

    for k,v in triplets_sorted:
        triplets_ok.append(k)
        print k,v
    triplets_ok = sorted(triplets_ok)
    triplets_ok = ['__background__']+triplets_ok
    for i in xrange(len(triplets_ok)):
        h5f['meta/tri/idx2name/%d'%i] = triplets_ok[i]
        h5f['meta/tri/name2idx/%s'%triplets_ok[i]] = i
    print len(triplets_ok)

def vr_make_meta_gt_visual_phrase():
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5','r',driver='core')
    h5f  = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_vp_meta.h5')

    triplets = {}
    cnt = 0
    zl.tick()
    for k in m['gt/train'].keys():
        if cnt %1000==0:

            print cnt,zl.tock()
            zl.tick()
        cnt+=1
        gt_boxes = []
        gt_labels = []
        sub_boxes = m['gt/train/%s/sub_boxes'%k][...]
        obj_boxes = m['gt/train/%s/obj_boxes'%k][...]
        rlp_labels = m['gt/train/%s/rlp_labels'%k][...]
        for i in xrange(rlp_labels.shape[0]):
            sub_box = sub_boxes[i]
            obj_box = obj_boxes[i]
            rlp_label = rlp_labels[i]
            joint_box = [min(sub_box[0],obj_box[0]), min(sub_box[1],obj_box[1]),max(sub_box[2],obj_box[2]),max(sub_box[3],obj_box[3])]
            s_lbl = zl.idx2name_cls(m,rlp_label[0])
            o_lbl = zl.idx2name_cls(m,rlp_label[2])
            p_lbl = zl.idx2name_pre(m,rlp_label[1])
            spo = '%s_%s_%s'%(s_lbl,p_lbl,o_lbl)
            lbl = zl.name2idx_tri(h5f,spo)
            gt_boxes.append(joint_box)
            gt_labels.append(lbl)
        h5f.create_dataset('gt/train/%s/labels'%k,data = np.array(gt_labels).astype(np.int16))
        h5f.create_dataset('gt/train/%s/boxes'%k,data = np.array(gt_boxes).astype(np.int16))

def vr_vphrase_make_voc_format(split_type):
    if split_type !='train' and split_type!='test':
        print 'error'
        exit(0)
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5')
    m_vp = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_vphrase_meta.h5')
    root = '/home/zawlin/data/data_vrd/vrd/sg_vp/'
    anno_root= root+'Annotations/'+split_type+'/'
    data_root= root+'Data/'+split_type+'/'
    zl.make_dirs(anno_root)
    zl.make_dirs(data_root)
    cnt = 0
    zl.tick()
    for k in m_vp['gt/%s'%split_type].keys():
        if cnt%1000==0:
            print cnt,zl.tock()
            zl.tick()
        cnt+=1
        # todo for vg
        # im_data= db.image_data.find_one({'image_id':imid})
        # im_path_full = im_data['url'].replace('https://cs.stanford.edu/people/rak248/','')
        # im_path_folder = im_path_full.split('/')[0]
        # im_path_file = im_path_full.split('/')[1]
        # im_src_path = vr_root+'%s/%s'%(im_path_folder,im_path_file)
        # im_dst_path = data_root+'%s/%s'%(im_path_folder,im_path_file)
        # zl.copy_file(im_src_path,im_dst_path)
        voc_datum = {"folder": '',
                     "source": {"database":"sg vrd visual phrase"},
                     "filename":k+'.jpg'
                     }
        m['train/%s/w'%k][...]
        w, h = int(m['train/%s/w'%k][...]),int(m['train/%s/h'%k][...])
        voc_datum['size']={'width':w,'height':h}

        objs = []
        gt_boxes = m_vp['gt/%s/%s/boxes'%(split_type,k)][...]
        gt_labels = m_vp['gt/%s/%s/labels'%(split_type,k)][...]
        for i in xrange(gt_boxes.shape[0]):
            gt_box = gt_boxes[i]
            gt_label = gt_labels[i]
            ymin, ymax, xmin, xmax = gt_box[1],gt_box[3],gt_box[0],gt_box[2]
            bbox = {'ymin':ymin,'ymax':ymax,'xmin':xmin,'xmax':xmax}
            name = zl.idx2name_tri(m_vp,gt_label)
            obj = {'name':name,
                   'bndbox':bbox}
            objs.append(obj)

        voc_datum['object']=objs
        #write to xml
        dst_path = os.path.join(anno_root,voc_datum["folder"], voc_datum["filename"][:voc_datum["filename"].rfind('.')]+'.xml')
        voc_datum={'annotation':voc_datum}
        f = open(dst_path,'w')
        f.write(dict2xml(voc_datum)+'\n')
        f.close()
    print 'images with annotation=%d\n'%cnt

def vr_make_meta_for_obj_evaluation():
    from numpy.core.records import fromarrays
    from scipy.io import savemat
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5')
    SG_VRD_ID = []
    WNID = []
    name = []
    description = []
    for i in xrange(1,101):
        n = str(m['meta/cls/idx2name/%d'%i][...])
        SG_VRD_ID.append(i)
        WNID.append(n)
        name.append(n)
        description.append(n)
    meta_synset = fromarrays([SG_VRD_ID,WNID,name,description], names=['SG_VRD_ID', 'WNID', 'name', 'description'])
    savemat('/home/zawlin/Dropbox/proj/sg_vrd_meta.mat', {'synsets': meta_synset})

vr_make_meta_for_obj_evaluation()
#add_maxbox()
#count_taller_occurrence()
#prep_test_data()
#vrd_meta()
#prep_train()
#prep_test()
#vrd_meta_add_predicate_types()
#vrd_generate_type_idx()
#prep_train_data_only()
#gen_image_sets()
# vr_make_meta_visual_phrase()
# vr_make_meta_gt_visual_phrase()
#vr_vphrase_make_voc_format('train')

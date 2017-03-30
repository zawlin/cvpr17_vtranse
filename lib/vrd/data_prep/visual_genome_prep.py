import _init_paths
import cv2

from pymongo import MongoClient
import time
import operator
import numpy as np
import utils.zl_utils as zl
from nltk.stem import PorterStemmer, WordNetLemmatizer

from textblob import TextBlob as tb
from textblob_aptagger import PerceptronTagger
import nltk
from nltk.tag.stanford import StanfordPOSTagger
import enchant
from nltk.metrics import edit_distance
from nltk.corpus import wordnet
import cv2
import numpy as np
import os
import scipy.io as sio
from dict2xml import dict2xml
import h5py
import socket

import random
def pos_tag_stanford(text):
    TCP_IP = '127.0.0.1'
    TCP_PORT = 2020
    MESSAGE = text + '\n'

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    s.send(MESSAGE)
    data = ''
    while True:
        data_frag = s.recv(1024)
        data += data_frag
        if len(data_frag) < 1024:
            break
    s.close()
    tags = []
    # print data.split(' ')
    for d in data.split(' '):
        if d != '': tags.append(tuple(d.split('_')))
    return tags

class SpellingReplacer(object):
    def __init__(self, dict_name='en_US', max_dist=2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = 2

    def replace(self, word):
        if self.spell_dict.check(word):
            return word
        if len(word) <= 3:
            return word
        suggestions = self.spell_dict.suggest(word)

        if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
            return suggestions[0]
        else:
            return word

def hanwang_help_region_descriptions():
    client = MongoClient("mongodb://localhost:27017")
    db = client.visual_genome
    db_results = db.region_descriptions.find(no_cursor_timeout=True)

    cnt = 0
    start = time.time()
    last = time.time()

    subjects = {}
    objects = {}
    objects_all = {}
    predicate = {}
    cnt = 0
    for doc in db_results:
        id = doc['id']
        cnt += 1
        if cnt > 6:
            exit(0)
        if cnt % 10000 == 0:
            print cnt
        img_path = db.image_data.find({"id": id})[0]['url'].replace('https://cs.stanford.edu/people/rak248/', '')
        img_path = '/media/zawlin/ssd/data_vrd/vg/' + img_path
        im = cv2.imread(img_path)
        rcnt = 0
        cv2.imwrite('/home/zawlin/hw/' + str(id) + '.jpg', im)
        for r in doc['regions']:
            rcnt += 1
            imdraw = im.copy()
            # c = (np.random.randint(40,255),np.random.randint(40,255),np.random.randint(40,255))
            cv2.rectangle(imdraw, (r['x'], r['y']),
                          (r['x'] + r['width'], r['y'] + r['height']), (0, 255, 0), 2)

            cv2.rectangle(imdraw, (0, 0),
                          (imdraw.shape[1], 20), (0, 0, 0), -1)
            cv2.putText(imdraw, r['phrase'], (15, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

            cv2.imwrite('/home/zawlin/hw/' + str(id) + '_' + str(rcnt) + '.jpg', imdraw)

def vg_data_vis():
    client = MongoClient("mongodb://localhost:27017")
    db = client.visual_genome
    db_results = db.relationships.find(no_cursor_timeout=True)

    cnt = 0
    start = time.time()
    last = time.time()

    subjects = {}
    objects = {}
    objects_all = {}
    predicate = {}
    cnt = 0
    for doc in db_results:
        id = doc['id']
        cnt += 1
        if cnt % 10000 == 0:
            print cnt
        img_path = db.image_data.find({"id": id})[0]['url'].replace('https://cs.stanford.edu/people/rak248/', '')
        img_path = '/media/zawlin/ssd/data_vrd/vg/' + img_path
        im = cv2.imread(img_path)
        rcnt = 0
        cv2.imwrite('/home/zawlin/hw/' + str(cnt) + '.jpg', im)
        for r in doc['relationships']:
            rcnt += 1
            imdraw = im.copy()
            # c = (np.random.randint(40,255),np.random.randint(40,255),np.random.randint(40,255))
            cv2.rectangle(imdraw, (r['object']['x'], r['object']['y']),
                          (r['object']['x'] + r['object']['w'], r['object']['y'] + r['object']['h']), (0, 255, 0), 2)
            cv2.rectangle(imdraw, (r['subject']['x'], r['subject']['y']),
                          (r['subject']['x'] + r['subject']['w'], r['subject']['y'] + r['subject']['h']), (0, 0, 255)
                          , 2)
            cv2.putText(imdraw, r['predicate'], (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(imdraw, r['object']['name'], (r['object']['x'] + 10, r['object']['y'] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(imdraw, r['subject']['name'], (r['subject']['x'] + 10, r['subject']['y'] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imwrite('/home/zawlin/hw/' + str(cnt) + '_' + str(rcnt) + '.jpg', imdraw)
            if r['predicate'] not in predicate:
                predicate[r['predicate']] = 1
            else:
                predicate[r['predicate']] += 1
            if r['subject']['name'] not in subjects:
                subjects[r['subject']['name']] = 1
            else:
                subjects[r['subject']['name']] += 1
            if r['object']['name'] not in objects:
                objects[r['object']['name']] = 1
            else:
                objects[r['object']['name']] += 1
            if r['subject']['name'] not in objects_all:
                objects_all[r['subject']['name']] = 1
            else:
                objects_all[r['subject']['name']] += 1
            if r['object']['name'] not in objects_all:
                objects_all[r['object']['name']] = 1
            else:
                objects_all[r['object']['name']] += 1
            cv2.imshow('imdraw', imdraw)
            cv2.waitKey(0)
        cv2.imshow('im', im)
        cv2.waitKey(0)

    objects_all_sorted = sorted(objects_all.items(), key=operator.itemgetter(1))
    objects_sorted = sorted(objects.items(), key=operator.itemgetter(1))
    subjects_sorted = sorted(subjects.items(), key=operator.itemgetter(1))
    predicate_sorted = sorted(predicate.items(), key=operator.itemgetter(1))
    print predicate_sorted[:100]

def vg_count_top():
    client = MongoClient("mongodb://localhost:27017")
    db = client.visual_genome_1_2
    db_results = db.relationships_cannon.find(no_cursor_timeout=True)

    cnt = 0
    start = time.time()
    last = time.time()

    subjects = {}
    objects = {}
    objects_all = {}
    predicate = {}
    cnt = 0
    for doc in db_results:
        imid = doc['image_id']
        cnt += 1
        if cnt % 10000 == 0:
            print cnt
        rcnt = 0
        for r in doc['relationships']:
            if r['predicate'] == 'C_FAILED': continue
            if r['subject']['name'] == 'C_FAILED': continue
            if r['object']['name'] == 'C_FAILED': continue

            rcnt += 1
            if r['predicate'] not in predicate:
                predicate[r['predicate']] = 1
            else:
                predicate[r['predicate']] += 1
            if r['subject']['name'] not in subjects:
                subjects[r['subject']['name']] = 1
            else:
                subjects[r['subject']['name']] += 1
            if r['object']['name'] not in objects:
                objects[r['object']['name']] = 1
            else:
                objects[r['object']['name']] += 1
            if r['subject']['name'] not in objects_all:
                objects_all[r['subject']['name']] = 1
            else:
                objects_all[r['subject']['name']] += 1
            if r['object']['name'] not in objects_all:
                objects_all[r['object']['name']] = 1
            else:
                objects_all[r['object']['name']] += 1
    zl.save('/home/zawlin/g/py-faster-rcnn/output/objects_all.pkl', objects_all)
    zl.save('/home/zawlin/g/py-faster-rcnn/output/objects.pkl', objects)
    zl.save('/home/zawlin/g/py-faster-rcnn/output/subjects.pkl', subjects)
    zl.save('/home/zawlin/g/py-faster-rcnn/output/predicate.pkl', predicate)
    objects_all_sorted = sorted(objects_all.items(), key=operator.itemgetter(1))
    objects_sorted = sorted(objects.items(), key=operator.itemgetter(1))
    subjects_sorted = sorted(subjects.items(), key=operator.itemgetter(1))
    predicate_sorted = sorted(predicate.items(), key=operator.itemgetter(1))

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def cannonicalize_so(text, wnl, spl):
    try:

        k = text.lower()
        text_arr = nltk.word_tokenize(k)
        text_arr = [spl.replace(i) for i in text_arr]
        # zl.tick()
        fixed_test = ''
        for t in text_arr:
            fixed_test += t + ' '
        fixed_test = fixed_test.strip()
        tags = pos_tag_stanford(fixed_test)

        # final = [wnl.lemmatize(i[0]) for i in tags if 'NN' in i[1] or 'V' in i[1]]
        final = [wnl.lemmatize(i[0]) for i in tags if
                 i[0] == 'can' or i[0] == 'building' or 'NN' in i[1] or ('DT' not in i[1] and 'RP' not in i[1]) and len(
                     wordnet.synsets(i[0], 'n')) > 0]
        res = ""
        for f in final:
            res = res + f + ' '
        res = res.strip()
        if res == "":
            return "C_FAILED"
        return res
    except Exception as ex:
        print ex
        return "C_FAILED"

def cannonicalize_relationship(text, wnl, spl):
    try:
        k = text.lower()
        text_arr = nltk.word_tokenize(k)
        text_arr = [spl.replace(i) for i in text_arr]
        # zl.tick()
        fixed_test = ''
        for t in text_arr:
            fixed_test += t + ' '
        fixed_test = fixed_test.strip()
        tags = pos_tag_stanford(fixed_test)
        final = [wnl.lemmatize(i[0], get_wordnet_pos(i[1])) for i in tags if
                 'DT' not in i[1] and 'RP' not in i[1] or len(wordnet.synsets(i[0], 'v')) > 0]
        res = ""
        for f in final:
            res = res + f + ' '
        res = res.strip()
        if res == "":
            return "C_FAILED"
        return res
    except Exception as ex:
        print ex
        return "C_FAILED"

def vg_data_prep2():
    objects_all = zl.load('/home/zawlin/g/py-faster-rcnn/output/objects_all.pkl')
    objects = zl.load('/home/zawlin/g/py-faster-rcnn/output/objects.pkl')
    subjects = zl.load('/home/zawlin/g/py-faster-rcnn/output/subjects.pkl')
    predicate = zl.load('/home/zawlin/g/py-faster-rcnn/output/predicate.pkl')

    objects_all_sorted = sorted(objects_all.items(), key=operator.itemgetter(1), reverse=True)
    objects_sorted = sorted(objects.items(), key=operator.itemgetter(1), reverse=True)
    subjects_sorted = sorted(subjects.items(), key=operator.itemgetter(1), reverse=True)
    predicate_sorted = sorted(predicate.items(), key=operator.itemgetter(1), reverse=True)

    wnl = WordNetLemmatizer()
    spl = SpellingReplacer()
    cnt = 0
    print objects_all_sorted[:100]
    # for k, v in predicate_sorted[:100]:
    #        print k

def vg_choose_final_set():
    objects_all = zl.load('/home/zawlin/g/py-faster-rcnn/output/objects_all.pkl')
    objects = zl.load('/home/zawlin/g/py-faster-rcnn/output/objects.pkl')
    subjects = zl.load('/home/zawlin/g/py-faster-rcnn/output/subjects.pkl')
    predicate = zl.load('/home/zawlin/g/py-faster-rcnn/output/predicate.pkl')

    objects_all_sorted = sorted(objects_all.items(), key=operator.itemgetter(1), reverse=True)
    objects_sorted = sorted(objects.items(), key=operator.itemgetter(1), reverse=True)
    subjects_sorted = sorted(subjects.items(), key=operator.itemgetter(1), reverse=True)
    predicate_sorted = sorted(predicate.items(), key=operator.itemgetter(1), reverse=True)
    # for pre in predicate_sorted:
        # print pre[0],pre[1]
    # exit(0)
    #objects_all_sorted = objects_all_sorted[:300]
    #predicate_sorted = predicate_sorted[:100]
    objects_final = {i[0]:i[1] for i in objects_all_sorted[:202]}
    del objects_final['woman']
    del objects_final['background']
    # print len(sorted(objects_final.keys()))
    # exit(0)
    #predicate_final = {i[0]:i[1] for i in predicate_sorted[:100]}
    predicate_final = {}
    mappings_p = make_p_mappings()
    for k in mappings_p.keys():
        if mappings_p[k] in predicate:
            predicate_final[mappings_p[k]] = predicate[mappings_p[k]]
    client = MongoClient("mongodb://localhost:27017")
    db = client.visual_genome_1_2
    db_results = db.relationships_cannon.find(no_cursor_timeout=True)

    cnt = 0
    start = time.time()
    last = time.time()

    subjects = {}
    objects = {}
    objects_all = {}
    predicate = {}
    cnt = 0
    docs = []
    for doc in db_results:
        if cnt % 10000 == 0:
            print cnt
        docs.append(doc)
        cnt += 1
    cnt =0
    random.shuffle(docs)
    for doc in docs:
        cnt += 1
        if cnt % 10000 == 0:
            print cnt
        rcnt = 0
        doc_object = dict(doc)
        doc_relationship = dict(doc)
        doc_object.pop('relationships',None)
        doc_relationship.pop('relationships',None)
        doc_object['objects'] = []
        doc_relationship['relationships'] = []
        for r in doc['relationships']:
            #doc_object['objects'].append()
            pre = r['predicate']

            obj = r['object']['name']
            sub = r['subject']['name']
            if sub in objects_final: doc_object['objects'].append(r['subject'])
            if obj in objects_final: doc_object['objects'].append(r['object'])
            if sub in objects_final and obj in objects_final and pre in predicate_final:
                doc_relationship['relationships'].append(r)
        if cnt<=80000:
            db.relationships_all_train.insert(doc_relationship)
            db.relationships_objects_train.insert(doc_object)
        else:
            db.relationships_all_test.insert(doc_relationship)
            db.relationships_objects_test.insert(doc_object)

def make_mappings():
    lines = [line.strip() for line in open('/media/zawlin/ssd/data_vrd/vg_1.2/objects.txt')]
    obj_mappings = {}
    for l in lines:
        ls = [i.strip() for i in l.split(',') if i.strip() != '']
        for ll in ls:
            obj_mappings[ll] = ls[0]
    return obj_mappings

def make_p_mappings():
    lines = [line.strip() for line in open('/media/zawlin/ssd/data_vrd/vg_1.2/vg_predicates.txt')]
    p_mappings = {}
    for l in lines:
        if ':' in l:continue
        ls = [i.strip() for i in l.split(',') if i.strip() != '']
        for i in xrange(len(ls)-1):
            p_mappings[ls[i]] = ls[0]
    return p_mappings

def vg_cannonicalize():
    client = MongoClient("mongodb://localhost:27017")
    db = client.visual_genome_1_2
    db_results = db.relationships.find(no_cursor_timeout=True)
    cnt = 0
    mappings = make_mappings()
    mappings_p = make_p_mappings()
    wnl = WordNetLemmatizer()
    spl = SpellingReplacer()
    zl.tick()
    for doc in db_results:
        id = doc['image_id']
        cnt += 1
        if cnt % 1000 == 0:
            print cnt, zl.tock()
            zl.tick()
        rcnt = 0
        for r in doc['relationships']:
            pre = r['predicate']

            sub_name = r['subject']['name']
            obj_name = r['object']['name']
            #if pre == '
            if sub_name in mappings:
                sub_name = mappings[sub_name]
            if obj_name in mappings:
                obj_name = mappings[obj_name]
            r['predicate_orig'] = pre
            r['object']['name_orig'] = obj_name
            r['subject']['name_orig'] = sub_name

            pre_canon = cannonicalize_relationship(pre, wnl, spl)
            obj_canon = cannonicalize_so(obj_name, wnl, spl)
            sub_canon  = cannonicalize_so(sub_name, wnl, spl)

            if pre_canon in mappings_p:
                pre_canon = mappings_p[pre_canon]
            if pre_canon == 'short than':
                pre_canon = 'tall than'
                sub_doc = r['subject']
                obj_doc = r['object']
                r['subject'],r['object'] = obj_doc,sub_doc
                r['subject_orig'],r['object_orig'] = sub_doc,obj_doc
                sub_canon,obj_canon = obj_canon,sub_canon
            if pre_canon == 'large than':
                pre_canon = 'small than'
                sub_doc = r['subject']
                obj_doc = r['object']
                r['subject'],r['object'] = obj_doc,sub_doc
                r['subject_orig'],r['object_orig'] = sub_doc,obj_doc
                sub_canon,obj_canon = obj_canon,sub_canon
            r['predicate'] = pre_canon
            r['object']['name'] =obj_canon
            r['subject']['name'] =sub_canon
        db.relationships_cannon.insert(doc)
        # exit(0)

def vg_make_voc_imageset(split_type):
    client = MongoClient("mongodb://localhost:27017")
    blacklist =[


            ]
    db = client.visual_genome_1_2
    if split_type !='train' and split_type!='test':
        print 'error'
        exit(0)
    vg_root = '/media/zawlin/ssd/data_vrd/vg_1.2/'
    imageset_root= '/media/zawlin/ssd/data_vrd/vg_1.2/voc_format/ImageSets/'+split_type+'.txt'
    cnt = 1
    # preload image data
    imdatas = {}
    for imdata in  db.image_data.find(no_cursor_timeout=True):
        imid =imdata['image_id']
        imdatas[imid] = imdata
    if split_type=='train':
        db_objs = db.relationships_objects_train.find(no_cursor_timeout=True)
    else:
        db_objs = db.relationships_objects_test.find(no_cursor_timeout=True)

    output = open(imageset_root,'w')
    mini_selection = {}
    for db_obj in db_objs:
        if len(db_obj['objects'])<=0:
            continue
        if cnt%1000==0:
            print cnt,zl.tock()
            zl.tick()
        imid = db_obj['image_id']

        im_data = imdatas[imid]
        if im_data['width']<100 or im_data['height']<100:continue
        im_path_full = im_data['url'].replace('https://cs.stanford.edu/people/rak248/','')
        im_path_folder = im_path_full.split('/')[0]
        im_path_file = im_path_full.split('/')[1]
        if '.jpg' not in im_path_file:
            print 'not a jpg image %s\n'%im_path_file
            exit(0)
        im_index = im_path_folder+'/'+im_path_file.replace('.jpg','')

        if im_index in blacklist:continue

        if split_type =='train':
            for o in db_obj['objects']:
                name = o['name']
                if name not in mini_selection:
                    mini_selection[name] = []
                if len(mini_selection[name])<3 and im_index not in mini_selection[name]:
                    mini_selection[name].append(im_index)

        output.write( '%s %d\n'%(im_index,cnt))
        cnt += 1
    output.close()

    if split_type=='train':
        imageset_root= '/media/zawlin/ssd/data_vrd/vg_1.2/voc_format/ImageSets/mini.txt'
        cnt = 1
        imageset_content=''
        for k in mini_selection.keys():
            for f in mini_selection[k]:
                imageset_content += '%s %d\n'%(f,cnt)
                cnt+= 1
        output = open(imageset_root,'w')
        output.write(imageset_content)
        output.close()

def vg_make_voc_format(split_type):
    if split_type !='train' and split_type!='test':
        print 'error'
        exit(0)
    vg_root = '/media/zawlin/ssd/data_vrd/vg_1.2/'
    anno_root= '/media/zawlin/ssd/data_vrd/vg_1.2/voc_format/Annotations/'+split_type+'/'
    data_root= '/media/zawlin/ssd/data_vrd/vg_1.2/voc_format/Data/'+split_type+'/'

    zl.make_dirs(anno_root+'VG_100K_2')
    zl.make_dirs(anno_root+'VG_100K')
    zl.make_dirs(data_root+'VG_100K_2')
    zl.make_dirs(data_root+'VG_100K')
    client = MongoClient("mongodb://localhost:27017")
    db = client.visual_genome_1_2
    if split_type=='train':
        db_objs = db.relationships_objects_train.find(no_cursor_timeout=True)
    else:
        db_objs = db.relationships_objects_test.find(no_cursor_timeout=True)

    cnt = 0
    zl.tick()
    for db_obj in db_objs:
        if cnt%1000==0:
            print cnt,zl.tock()
            zl.tick()
        cnt+=1
        imid = db_obj['image_id']
        im_data= db.image_data.find_one({'image_id':imid})
        im_path_full = im_data['url'].replace('https://cs.stanford.edu/people/rak248/','')
        im_path_folder = im_path_full.split('/')[0]
        im_path_file = im_path_full.split('/')[1]
        im_src_path = vg_root+'%s/%s'%(im_path_folder,im_path_file)
        im_dst_path = data_root+'%s/%s'%(im_path_folder,im_path_file)
        zl.copy_file(im_src_path,im_dst_path)
        voc_datum = {"folder": im_path_folder,
                     "source": {"database":"visual genome 1.2"},
                     "filename":im_path_file
                     }

        w, h = im_data['width'],im_data['height']
        voc_datum['size']={'width':w,'height':h}

        objs = []
        for o in db_obj['objects']:
            ymin, ymax, xmin, xmax = o['y'],o['y']+o['h'],o['x'],o['x']+o['w']
            bbox = {'ymin':ymin,'ymax':ymax,'xmin':xmin,'xmax':xmax}
            obj = {'name':o['name'],
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

def pre_output():
    h5f = '/home/zawlin/Dropbox/proj/sg_vrd_meta.h5'
    h5f = h5py.File(h5f)
    for k in  h5f['meta/pre/name2idx'].keys():
        print k+(',spatial' if len(h5f['meta/pre/name2idx/'+k].attrs)>0 else ',')
    pass

def vg_stats_predicate():
    client = MongoClient("mongodb://localhost:27017")
    db = client.visual_genome_1_2
    db_results = db.relationships_cannon.find(no_cursor_timeout=True)
    cnt = 0
    mappings = make_mappings()
    mappings_p = make_p_mappings()
    wnl = WordNetLemmatizer()
    spl = SpellingReplacer()
    sub_obj_info = {}
    zl.tick()
    for doc in db_results:
        id = doc['image_id']
        cnt += 1
        if cnt % 1000 == 0:
            print cnt, zl.tock()
            zl.tick()
        rcnt = 0
        for r in doc['relationships']:
            pre = r['predicate']
            sub_name = r['subject']['name']
            obj_name = r['object']['name']
            so_pair = sub_name + '_'+ obj_name
            if so_pair not in sub_obj_info:
                so_info = {'total':0,'predicates':[]}

            else:
                so_info = sub_obj_info[so_pair]
            so_info['total']+=1
            if pre not in so_info['predicates']:
                so_info['predicates'].append(pre)
    zl.save('output/sub_obj_info.pkl',sub_obj_info)
    #total_pairs = len(sub_obj_info.keys())+0.0
    total_pairs = 0.0
    total_of_averages = 0.0
    for k in sub_obj_info.keys():
        so_info = sub_obj_info[k]
        total_predicates = len(so_info['predicates'])+0.0
        if so_info['total']<2:continue
        total_pairs += 1
        total_annotated_pairs = so_info['total']+0.0
        avg_predicates_for_this_pair = total_predicates/total_annotated_pairs
        total_of_averages+=avg_predicates_for_this_pair
    total_of_averages /= total_pairs
    print 'total_pairs = %d'%total_pairs
    print 'total_of_averages = %d'%total_of_averages

def vg_check_pre_stats():
    client = MongoClient("mongodb://localhost:27017")
    db = client.visual_genome_1_2
    db_rel_train_all = db.relationships_all_train.find(no_cursor_timeout=True)
    db_rel_test_all= db.relationships_all_test.find(no_cursor_timeout=True)
    train_stats = {}
    test_stats = {}
    cnt = 0
    zl.tick()
    for db_rel in db_rel_train_all:
        if cnt%1000==0:
            print cnt,zl.tock()
            zl.tick()
        cnt += 1
        for r in db_rel['relationships']:
            name = r['predicate']
            if name not in train_stats:
                train_stats[name] = 0
            train_stats[name] += 1

    for db_rel in db_rel_test_all:
        if cnt%1000==0:
            print cnt,zl.tock()
            zl.tick()
        cnt += 1
        for r in db_rel['relationships']:
            name = r['predicate']
            if name not in test_stats:
                test_stats[name] = 0
            test_stats[name] += 1
    zl.save('output/train_pre_stats.pkl',train_stats)
    zl.save('output/test_pre_stats.pkl',test_stats)
    print zl.sort_dict_by_val(train_stats)
    print zl.sort_dict_by_val(test_stats)

def vg_check_obj_stats():
    client = MongoClient("mongodb://localhost:27017")
    db = client.visual_genome_1_2
    db_obj_train_all = db.relationships_objects_train.find(no_cursor_timeout=True)
    db_obj_test_all= db.relationships_objects_test.find(no_cursor_timeout=True)
    train_stats = {}
    test_stats = {}
    cnt = 0
    zl.tick()
    for db_obj in db_obj_train_all:
        if cnt%1000==0:
            print cnt,zl.tock()
            zl.tick()
        cnt += 1
        for o in db_obj['objects']:
            name = o['name']
            if name not in train_stats:
                train_stats[name] = 0
            train_stats[name] += 1

    for db_obj in db_obj_test_all:
        if cnt%1000==0:
            print cnt,zl.tock()
            zl.tick()
        cnt += 1
        for o in db_obj['objects']:
            name = o['name']
            if name not in test_stats:
                test_stats[name] = 0
            test_stats[name] += 1
    zl.save('output/train_stats.pkl',train_stats)
    zl.save('output/test_stats.pkl',test_stats)
    print zl.sort_dict_by_val(train_stats)
    print zl.sort_dict_by_val(test_stats)

def vg_make_meta_visual_phrase():
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5','r',driver='core')

    h5f  = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_vp_meta.h5')

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
            # sub_box = sub_boxes[i]
            # obj_box = obj_boxes[i]
            rlp_label = rlp_labels[i]
            # joint_bbox = [min(sub_bbox[0],obj_bbox[0]), min(sub_bbox[1],obj_bbox[1]),max(sub_bbox[2],obj_bbox[2]),max(sub_bbox[3],obj_bbox[3])]

            s_lbl = zl.idx2name_cls(m,rlp_label[0])
            o_lbl = zl.idx2name_cls(m,rlp_label[2])
            p_lbl = zl.idx2name_pre(m,rlp_label[1])

            spo = '%s_%s_%s'%(s_lbl,p_lbl,o_lbl)
            # spo = '%d_%d_%d'%(rlp_label[0],rlp_label[1],rlp_label[2])
            if spo not in triplets:
                triplets[spo]=0
            triplets[spo]+=1
    zl.save('output/pkl/triplets_train_vp.pkl',triplets)
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

def vg_make_meta_gt_visual_phrase():
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5','r',driver='core')
    h5f  = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_vp_meta.h5')

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

def vg_make_meta_hdf():
    train_stats = zl.load('output/train_stats.pkl')
    test_stats = zl.load('output/test_stats.pkl')
    #train_stats = zl.sort_dict_by_val(train_stats)
    obj_names_sorted =  sorted(train_stats.keys())
    classes = ('__background__',)  # always index 0
    for i in xrange(len(obj_names_sorted)):
        classes += (obj_names_sorted[i],)

    train_pre_stats = zl.load('output/train_pre_stats.pkl')
    test_pre_stats = zl.load('output/test_pre_stats.pkl')

    pre_names_sorted =  sorted(train_pre_stats.keys())
    predicates=()
    for i in range(len(pre_names_sorted)):
        predicates += (pre_names_sorted[i],)

    h5f = '/home/zawlin/Dropbox/proj/vg1_2_meta.h5'
    h5f = h5py.File(h5f)
    for i in range(201):
        h5f['meta/cls/name2idx/'+classes[i]] = str(i)
        h5f['meta/cls/idx2name/'+str(i)] = classes[i]

    for i in range(100):
        h5f['meta/pre/name2idx/'+predicates[i]] = str(i)
        h5f['meta/pre/idx2name/'+str(i)] = predicates[i]

def vg_db_stats():
    client = MongoClient("mongodb://localhost:27017")
    db = client.visual_genome_1_2
    db_rel_train_all = db.relationships_all_train.find(no_cursor_timeout=True)
    db_rel_test_all= db.relationships_all_test.find(no_cursor_timeout=True)
    train_stats = {}
    test_stats = {}
    cnt = 0
    zl.tick()
    for db_rel in db_rel_train_all:
        if cnt%1000==0:
            print cnt,zl.tock()
            zl.tick()
        cnt += 1
        for r in db_rel['relationships']:
            name = r['predicate']
            if name not in train_stats:
                train_stats[name] = 0
            train_stats[name] += 1

    for db_rel in db_rel_test_all:
        if cnt%1000==0:
            print cnt,zl.tock()
            zl.tick()
        cnt += 1
        for r in db_rel['relationships']:
            name = r['predicate']
            if name not in test_stats:
                test_stats[name] = 0
            test_stats[name] += 1

def vr_count_only_one_triplet():
    m = h5py.File('/media/zawlin/ssd/Dropbox/proj/sg_vrd_meta.h5')
    triplets = {}
    for k in m['gt/train/'].keys():
        rlp_labels = m['gt/train/'+k+'/rlp_labels']
        for i in xrange(rlp_labels.shape[0]):
            triplet = str(rlp_labels[i][0])+'_'+str(rlp_labels[i][1])+'_'+str(rlp_labels[i][2])
            if triplet not in triplets:
                triplets[triplet] = 0
            triplets[triplet] +=1

    for k in m['gt/test/'].keys():
        rlp_labels = m['gt/test/'+k+'/rlp_labels']
        for i in xrange(rlp_labels.shape[0]):
            triplet = str(rlp_labels[i][0])+'_'+str(rlp_labels[i][1])+'_'+str(rlp_labels[i][2])
            if triplet not in triplets:
                triplets[triplet] = 0
            triplets[triplet] +=1
    total_spo = len(triplets.keys())+0.0
    one_count = 0
    for k in triplets.keys():
        if triplets[k] == 1:
           one_count += 1
    print total_spo,one_count

def vg_total_annotation_count(spo_list):

    client = MongoClient("mongodb://localhost:27017")
    db = client.visual_genome_1_2
    db_results = db.relationships_all_train.find(no_cursor_timeout=True)
    cnt = 0
    rcnt = 0
    zl.tick()
    total_train_cnt = 0
    for doc in db_results:
        id = doc['image_id']
        cnt += 1
        if cnt % 1000 == 0:
            print cnt, zl.tock()
            zl.tick()
        ok = False
        for r in doc['relationships']:
            pre = r['predicate']
            sub_name = r['subject']['name']
            obj_name = r['object']['name']
            spo = sub_name+'_'+pre+'_'+obj_name
            if spo in spo_list:
                rcnt+=1
                ok = True
        if ok:
            total_train_cnt+=1



    db_results_2 = db.relationships_all_test.find(no_cursor_timeout=True)
    total_test_cnt = 0
    for doc in db_results_2:
        id = doc['image_id']
        cnt += 1
        if cnt % 1000 == 0:
            print cnt, zl.tock()
            zl.tick()
        ok = False
        for r in doc['relationships']:
            pre = r['predicate']
            sub_name = r['subject']['name']
            obj_name = r['object']['name']
            spo = sub_name+'_'+pre+'_'+obj_name
            if spo in spo_list:
                rcnt+=1
                ok = True
        if ok:
            total_test_cnt += 1
    print rcnt,total_train_cnt,total_test_cnt

def vg_count_predicate_per_object():
    client = MongoClient("mongodb://localhost:27017")
    db = client.visual_genome_1_2
    db_results = db.relationships_all_train.find(no_cursor_timeout=True)
    cnt = 0
    spo_infos = {}
    zl.tick()
    for doc in db_results:
        id = doc['image_id']
        cnt += 1
        if cnt % 1000 == 0:
            print cnt, zl.tock()
            zl.tick()
        rcnt = 0
        for r in doc['relationships']:
            pre = r['predicate']
            sub_name = r['subject']['name']
            obj_name = r['object']['name']
            if obj_name not in spo_infos:
                spo_info = {'predicates':[]}
                spo_infos[obj_name] = spo_info

            if sub_name not in spo_infos:
                spo_info = {'predicates':[]}
                spo_infos[sub_name] = spo_info
            sub_spo_info = spo_infos[sub_name]
            obj_spo_info = spo_infos[obj_name]
            if pre not in sub_spo_info['predicates']:
                sub_spo_info['predicates'].append(pre)
            if pre not in obj_spo_info['predicates']:
                obj_spo_info['predicates'].append(pre)


    db_results_2 = db.relationships_all_test.find(no_cursor_timeout=True)

    for doc in db_results_2:
        id = doc['image_id']
        cnt += 1
        if cnt % 1000 == 0:
            print cnt, zl.tock()
            zl.tick()
        rcnt = 0
        for r in doc['relationships']:
            pre = r['predicate']
            sub_name = r['subject']['name']
            obj_name = r['object']['name']
            if obj_name not in spo_infos:
                spo_info = {'predicates':[]}
                spo_infos[obj_name] = spo_info
            if sub_name not in spo_infos:
                spo_info = {'predicates':[]}
                spo_infos[sub_name] = spo_info
            sub_spo_info = spo_infos[sub_name]
            obj_spo_info = spo_infos[obj_name]
            if pre not in sub_spo_info['predicates']:
                sub_spo_info['predicates'].append(pre)
            if pre not in obj_spo_info['predicates']:
                obj_spo_info['predicates'].append(pre)
    total_predicates = 0
    for k in spo_infos.keys():
        spo_info = spo_infos[k]
        print len(spo_info['predicates'])
        total_predicates+= len(spo_info['predicates'])
    print total_predicates/200.

def vg_count_only_one_triplet():
    client = MongoClient("mongodb://localhost:27017")
    db = client.visual_genome_1_2
    db_results = db.relationships_all_train.find(no_cursor_timeout=True)
    cnt = 0
    spo_info = {}
    spo_list = []
    zl.tick()
    for doc in db_results:
        id = doc['image_id']
        cnt += 1
        if cnt % 1000 == 0:
            print cnt, zl.tock()
            zl.tick()
        rcnt = 0
        for r in doc['relationships']:
            pre = r['predicate']
            sub_name = r['subject']['name']
            obj_name = r['object']['name']
            spo = sub_name+'_'+pre+'_'+obj_name
            if spo not in spo_info:
                spo_info[spo]= 0
            spo_info[spo]+=1

    db_results_2 = db.relationships_all_test.find(no_cursor_timeout=True)

    for doc in db_results_2:
        id = doc['image_id']
        cnt += 1
        if cnt % 1000 == 0:
            print cnt, zl.tock()
            zl.tick()
        rcnt = 0
        for r in doc['relationships']:
            pre = r['predicate']
            sub_name = r['subject']['name']
            obj_name = r['object']['name']
            spo = sub_name+'_'+pre+'_'+obj_name
            if spo not in spo_info:
                spo_info[spo]= 0
            spo_info[spo]+=1
    zl.save('output/spo_info_vg.pkl',spo_info)
    #total_pairs = len(sub_obj_info.keys())+0.0
    total_spo = len(spo_info.keys())+0.0
    one_count = 0
    for k in spo_info.keys():
        if spo_info[k]>=5:
            spo_list.append(k)
            one_count += 1
    #print total_spo,one_count
    vg_total_annotation_count(spo_list)

def relation_make_meta_add_imid2path():
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5')

    client = MongoClient("mongodb://localhost:27017")
    db = client.visual_genome_1_2
    imdatas = {}
    for imdata in  db.image_data.find(no_cursor_timeout=True):
        imid = imdata['image_id']
        imdatas[imid] = imdata

    for k in imdatas.keys():
        im_data = imdatas[k]
        im_path_full = im_data['url'].replace('https://cs.stanford.edu/people/rak248/','')
        im_path_folder = im_path_full.split('/')[0]
        im_path_file = im_path_full.split('/')[1]
        im_path_relative = im_path_folder+'/'+im_path_file
        m['meta/imid2path/%s'%k] = im_path_relative

def relation_make_meta():
    spo_info = zl.load('output/spo_info_vg.pkl')
    spo_list = []
    spo_dict = {}
    for k in spo_info.keys():
        if spo_info[k]>=5:
            spo_list.append(k)
    for spo in spo_list:
        spo_dict[spo] = 0
    spo_list = spo_dict
    blacklist =[
            'VG_100K/2363098',
            'VG_100K_2/2402233',
            'VG_100K/2365839',
            'VG_100K_2/2398948',
            'VG_100K/2315697',
            'VG_100K_2/2403354',
            ]

    client = MongoClient("mongodb://localhost:27017")
    db = client.visual_genome_1_2
    imdatas = {}
    for imdata in  db.image_data.find(no_cursor_timeout=True):
        imid =imdata['image_id']
        imdatas[imid] = imdata
    db_results_train = db.relationships_all_train.find(no_cursor_timeout=True)
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5')
    cnt = 0
    for doc in db_results_train:
        if cnt %1000 ==0:
            print cnt
        cnt += 1
        imid = doc['image_id']
        im_data = imdatas[imid]
        if im_data['width']<100 or im_data['height']<100:continue
        im_path_full = im_data['url'].replace('https://cs.stanford.edu/people/rak248/','')
        im_path_folder = im_path_full.split('/')[0]
        im_path_file = im_path_full.split('/')[1]
        im_index = im_path_folder+'/'+im_path_file.replace('.jpg','')
        if im_index in blacklist:continue
        obj_boxes = []
        sub_boxes = []
        rlp_labels = []
        imid = doc['image_id']
        imdata = imdatas[imid]
        for r in doc['relationships']:
            pre = r['predicate']
            sub_name = r['subject']['name']
            obj_name = r['object']['name']
            spo = sub_name+'_'+pre+'_'+obj_name
            if spo in spo_list:
                sidx = zl.name2idx_cls(m,sub_name)
                oidx = zl.name2idx_cls(m,obj_name)
                pidx = zl.name2idx_pre(m,pre)
                ox1,oy1,ow,oh = r['object']['x'],r['object']['y'],r['object']['w'],r['object']['h']
                sx1,sy1,sw,sh = r['subject']['x'],r['subject']['y'],r['subject']['w'],r['subject']['h']
                ox2,oy2 = ox1+ow,oy1+oh
                sx2,sy2 = sx1+sw,sy1+sh
                rlp_labels.append([sidx,pidx,oidx])
                sub_boxes.append([sx1,sy1,sx2,sy2])
                obj_boxes.append([ox1,oy1,ox2,oy2])
        m.create_dataset('gt/train/%s/rlp_labels'%imid,data=np.array(rlp_labels).astype(np.int16))
        m.create_dataset('gt/train/%s/sub_boxes'%imid,data=np.array(sub_boxes).astype(np.int16))
        m.create_dataset('gt/train/%s/obj_boxes'%imid,data=np.array(obj_boxes).astype(np.int16))

    db_results_test = db.relationships_all_test.find(no_cursor_timeout=True)
    for doc in db_results_test:
        if cnt %1000 ==0:
            print cnt
        cnt += 1
        imid = doc['image_id']
        im_data = imdatas[imid]
        if im_data['width']<100 or im_data['height']<100:continue
        im_path_full = im_data['url'].replace('https://cs.stanford.edu/people/rak248/','')
        im_path_folder = im_path_full.split('/')[0]
        im_path_file = im_path_full.split('/')[1]
        im_index = im_path_folder+'/'+im_path_file.replace('.jpg','')
        if im_index in blacklist:continue
        obj_boxes = []
        sub_boxes = []
        rlp_labels = []
        imid = doc['image_id']
        imdata = imdatas[imid]
        for r in doc['relationships']:
            pre = r['predicate']
            sub_name = r['subject']['name']
            obj_name = r['object']['name']
            spo = sub_name+'_'+pre+'_'+obj_name
            if spo in spo_list:
                sidx = zl.name2idx_cls(m,sub_name)
                oidx = zl.name2idx_cls(m,obj_name)
                pidx = zl.name2idx_pre(m,pre)
                ox1,oy1,ow,oh = r['object']['x'],r['object']['y'],r['object']['w'],r['object']['h']
                sx1,sy1,sw,sh = r['subject']['x'],r['subject']['y'],r['subject']['w'],r['subject']['h']
                ox2,oy2 = ox1+ow,oy1+oh
                sx2,sy2 = sx1+sw,sy1+sh
                rlp_labels.append([sidx,pidx,oidx])
                sub_boxes.append([sx1,sy1,sx2,sy2])
                obj_boxes.append([ox1,oy1,ox2,oy2])
        m.create_dataset('gt/test/%s/rlp_labels'%imid,data=np.array(rlp_labels).astype(np.int16))
        m.create_dataset('gt/test/%s/sub_boxes'%imid,data=np.array(sub_boxes).astype(np.int16))
        m.create_dataset('gt/test/%s/obj_boxes'%imid,data=np.array(obj_boxes).astype(np.int16))

def remove_empty_from_metadata():
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta (3rd copy).h5')
    del_cnt =0
    cnt = 0
    # print len(m['gt/train'].keys())
    # exit(0)
    for k in m['gt/test'].keys():
        if cnt%1000==0:
            print cnt
        cnt +=1
        rlp_labels = m['gt/test/%s/sub_boxes'%k][...]
        if rlp_labels.shape[0]==0:
            del m['gt/test/%s'%k]
            del_cnt+=1

    for k in m['gt/train'].keys():
        if cnt%1000==0:
            print cnt
        cnt +=1
        rlp_labels = m['gt/train/%s/sub_boxes'%k][...]
        if rlp_labels.shape[0]==0:
            del m['gt/train/%s'%k]
            del_cnt+=1

def vg_vphrase_make_voc_format(split_type):
    if split_type !='train' and split_type!='test':
        print 'error'
        exit(0)
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5')
    m_vp = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_vp_meta.h5')
    vg_root = '/media/zawlin/ssd/data_vrd/vg_1.2/'
    root = '/media/zawlin/ssd/data_vrd/vg_1.2/voc_format_vp/'
    anno_root= root+'Annotations/'+split_type+'/'
    data_root= root+'Data/'+split_type+'/'
    zl.make_dirs(anno_root+'VG_100K_2')
    zl.make_dirs(anno_root+'VG_100K')
    zl.make_dirs(data_root+'VG_100K_2')
    zl.make_dirs(data_root+'VG_100K')

    client = MongoClient("mongodb://localhost:27017")
    db = client.visual_genome_1_2
    imdatas = {}
    for imdata in  db.image_data.find(no_cursor_timeout=True):
        imid =str(imdata['image_id'])
        imdatas[imid] = imdata
    imid2path = {}
    for k in m['meta/imid2path'].keys():
        imid2path[k] = str(m['meta/imid2path/%s'%k][...])

    cnt = 0
    zl.tick()
    for k in m_vp['gt/%s'%split_type].keys():
        if cnt%1000==0:
            print cnt,zl.tock()
            zl.tick()
        cnt+=1
        # todo for vg
        im_path = imid2path[k]
        im_src_path = vg_root+im_path
        im_dst_path = data_root+im_path
        zl.copy_file(im_src_path,im_dst_path)
        voc_datum = {"folder": im_path.split('/')[0],
                     "source": {"database":"sg vrd visual phrase"},
                     "filename":im_path.split('/')[1]
                     }
        #todo,remove mongodb from this processing stage
        imdata = imdatas[k]
        w, h =imdata['width'],imdata['height']
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

def vg_vphrase_make_imagesets():
    imageset_root= '/media/zawlin/ssd/data_vrd/vg_1.2/voc_format_vp/ImageSets/train.txt'
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5')
    imid2path = {}
    for k in m['meta/imid2path'].keys():
        imid2path[k] = str(m['meta/imid2path/%s'%k][...])
    m.close()
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_vp_meta.h5')
    out = open(imageset_root,'w')
    cnt = 1
    for k in m['gt/train'].keys():
        out.write('%s %d\n'%(imid2path[k].replace('.jpg',''),cnt))
        cnt+=1

def vg_make_meta_for_obj_evaluation():
    from numpy.core.records import fromarrays
    from scipy.io import savemat
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5')
    VG1_2_ID = []
    WNID = []
    name = []
    description = []
    for i in xrange(1,201):
        n = str(m['meta/cls/idx2name/%d'%i][...])
        VG1_2_ID.append(i)
        WNID.append(n)
        name.append(n)
        description.append(n)
    meta_synset = fromarrays([VG1_2_ID,WNID,name,description], names=['VG1_2_ID', 'WNID', 'name', 'description'])
    savemat('/home/zawlin/Dropbox/proj/vg1_2_meta.mat', {'synsets': meta_synset})

def vg_make_relation_gt_for_evaluation():
    from numpy.core.records import fromarrays
    from scipy.io import savemat
    h5f = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5')
    cnt = 1
    gt_obj_bboxes = []
    gt_sub_bboxes = []
    gt_rlp_labels = []
    keys_sorted = sorted(h5f['gt/test'].keys())
    imid2path = {}
    for k in h5f['meta/imid2path'].keys():
        imid2path[k] = str(h5f['meta/imid2path/%s'%k][...])
    ccnt=0
    imagePath = []
    for k in keys_sorted:
        if cnt%1000==0:
            print cnt
        cnt+=1
        obj_boxes = h5f['gt/test/%s/obj_boxes'%k][...].astype(np.float64)
        rlp_labels = h5f['gt/test/%s/rlp_labels'%k][...].astype(np.float64)
        sub_boxes = h5f['gt/test/%s/sub_boxes'%k][...].astype(np.float64)
        # for i in xrange(rlp_labels.shape[0]):
            # if rlp_labels[i][1] == 76 or rlp_labels[i][1]==85:
                # ccnt += 1
        gt_obj_bboxes.append(obj_boxes)
        gt_sub_bboxes.append(sub_boxes)
        gt_rlp_labels.append(rlp_labels)
        imagePath.append(imid2path[k])
    # print ccnt
    savemat('/home/zawlin/Dropbox/proj/vg_gt.mat',{'gt_obj_bboxes':np.array(gt_obj_bboxes),'gt_sub_bboxes':np.array(gt_sub_bboxes),'gt_rlp_labels':np.array(gt_rlp_labels),'imagePath':np.array(imagePath,dtype=np.object)})

def vg_meta_add_predicate_types():
    h5f = '/home/zawlin/Dropbox/proj/vg1_2_meta.h5'
    h5f = h5py.File(h5f)

    lines = [line.strip() for line in open('/media/zawlin/ssd/Dropbox/cvpr17/_relation_mappings/vg_predicates_for_processing.txt')]
    type_mappings={}
    for l in lines:
        ls = [i.strip() for i in l.split(',') if i.strip() != '']
        type_mappings[ls[0]]=ls[-1]
    #print type_mappings
    for k in h5f['meta/pre/name2idx/']:
        h5f['meta/pre/name2idx/'+k].attrs['type']=type_mappings[k]


def vg_generate_type_idx():
    h5f = '/home/zawlin/Dropbox/proj/vg1_2_meta.h5'
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

def vg_vp_meta_add_predicate_types():
    h5f = '/home/zawlin/Dropbox/proj/vg1_2_vp_meta.h5'
    h5f = h5py.File(h5f)

    lines = [line.strip() for line in open('/home/zawlin/Dropbox/cvpr17/_relation_mappings/vg_predicates_for_processing.txt')]
    type_mappings={}
    for l in lines:
        ls = [i.strip() for i in l.split(',') if i.strip() != '']
        type_mappings[ls[0]]=ls[-1]
    for k in h5f['meta/tri/name2idx/']:
        pre_orig = k
        try:
            if pre_orig != '__background__':
                pre = pre_orig.split('_')[1]
                # if 'tall' in pre_orig:
                    # print pre_orig

                # if 'short' in pre_orig:
                    # print pre_orig
                # if type_mappings[pre]=='c':
                    # print pre_orig
                h5f['meta/tri/name2idx/'+k].attrs['type']=type_mappings[pre]
        except:
            print pre_orig
            exit(0)
def check_c_type_img_in_train():
    h5f = '/home/zawlin/Dropbox/proj/vg1_2_meta.h5'
    h5f = h5py.File(h5f)
    for k in h5f['meta/pre/name2idx/']:
        if k !='__background__':
            if h5f['meta/pre/name2idx/%s'%k].attrs['type']=='c':
                print str(h5f['meta/pre/name2idx/%s'%k][...])
                print k

    for k in h5f['gt/test']:
        #print h5f['gt/train'][k].keys()
        p_labels = h5f['gt/test'][k]['rlp_labels'][...][:,1]
        # 76 == small than, 85==tall than
        if np.any(np.in1d(p_labels,np.array([76,85]))):
            # h5f.create_dataset('gt/test/%s/obj_boxes'%k,data = h5f['gt/train'][k]['obj_boxes'][...].astype(np.short))
            # h5f.create_dataset('gt/test/%s/sub_boxes'%k,data = h5f['gt/train'][k]['sub_boxes'][...].astype(np.short))
            # h5f.create_dataset('gt/test/%s/rlp_labels'%k,data = h5f['gt/train'][k]['rlp_labels'][...].astype(np.short))
            print k
        # print p_labels
        # print 38 in p_labels
        # exit(0)
        # if 2794 in h5f['gt/train'][k]['labels'][...]:
            # print k

def check_imid():
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5')
    for k in m['meta/imid2path'].keys():
        impath = str(m['meta/imid2path/%s'%k][...])
        impath = impath.split('/')[1].replace('.jpg','')
        if impath!=k:
            print impath
    pass

def gen_vg_predicates():
    m = h5py.File('/home/zawlin/Dropbox/proj/vg1_2_meta.h5')
    for k in m['meta/pre/name2idx']:
        print k,',',m['meta/pre/name2idx'][k].attrs['type']

def gen_vr_predicates():
    m = h5py.File('/home/zawlin/Dropbox/proj/sg_vrd_meta.h5')
    for k in m['meta/cls/name2idx']:
        print k
# vg_make_voc_imageset('train')
# vg_make_voc_imageset('test')
#relation_make_meta_add_imid2path()
#vg_make_meta_visual_phrase()
# vg_make_meta_gt_visual_phrase()
#vg_vphrase_make_voc_format('train')
#vg_vphrase_make_imagesets()
#vg_make_relation_gt_for_evaluation()
#check_imid()
#check_c_type_img_in_train()
# a = np.array([1,2,3])
# b = np.array([2,3])
# print np.any(np.in1d(a,b))
gen_vr_predicates()

This implements "Visual Translation Embedding Network for Visual Relation Detection,Hanwang Zhang, Zawlin Kyaw, Shih-Fu Chang, Tat-Seng Chua (CVPR2017)"

#### What's inside?
* object detectors training and data preprocessor for vrd and vg.
* vg canonicalization
* two-stage relation training code for vrd and vg.
* evaluation code for vg+vrd adapted from Lu (https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection)

#### Download links
* annotations in hdf5 format(Since quite a few have asked, for vrd, this is the same as the original mat-based data format. For vg, this is our own cleaned up set with 200 object categories and 100 predicates. The data is organized into train and test split for both datasets). If you come here for the dataset, this is the one.
    * https://www.dropbox.com/s/tae51mr75nd9qft/sg_vrd_meta.h5?dl=0
    * https://www.dropbox.com/s/ujd4247m2tduuj2/vg1_2_meta.h5?dl=0
* voc format for obj detector
    * https://www.dropbox.com/s/wsoqj8iczkgxgzs/sg_vrd_voc.tar.gz?dl=0
    * https://www.dropbox.com/s/8d38zjnirdg8xzw/vg_1.2_voc.tar.gz?dl=0
* object detectors
    * https://www.dropbox.com/s/m971saz2xue6evp/vg_obj_model.caffemodel?dl=0
    * https://www.dropbox.com/s/dp6r22olfaf96j5/vr_obj_model.caffemodel?dl=0
* relation model
    * https://www.dropbox.com/s/29pnw7hyoo1fvd8/vr_rel_model.caffemodel?dl=0

#### Coming soon
* instructions for running training
* demo code

#### Setup
##### Object Detector
Ensure data folder looks like this. 

    zawlin@zlgpu:~/g/cvpr17_vtranse/data$ tree -l -L 4 -d
    .
    ├── demo
    ├── scripts
    ├── sg_vrd_2016 -> /media/zawlin/ssd/data/vrd/vrd/sg
    │   ├── Annotations
    │   │   ├── sg_test_images
    │   │   └── sg_train_images
    │   ├── Data
    │   │   ├── sg_test_images
    │   │   └── sg_train_images
    │   ├── devkit
    │   │   ├── data
    │   │   │   └── ilsvrc_det_sample
    │   │   └── evaluation
    │   └── ImageSets
    └── vg1_2_2016 -> /media/zawlin/ssd/data/vrd/vg_1.2/voc_format
        ├── Annotations
        │   ├── test
        │   │   ├── VG_100K
        │   │   └── VG_100K_2
        │   └── train
        │       ├── VG_100K
        │       └── VG_100K_2
        ├── Data
        │   ├── test
        │   │   ├── VG_100K
        │   │   └── VG_100K_2
        │   └── train
        │       ├── VG_100K
        │       └── VG_100K_2
        ├── devkit
        │   ├── data
        │   │   └── ilsvrc_det_sample
        │   └── evaluation
        └── ImageSets

#### Citation

If you're using this code in a scientific publication please cite:
```
@inproceedings{Zhang_2017_CVPR,
  author    = {Hanwang Zhang, Zawlin Kyaw, Shih-Fu Chang, Tat-Seng Chua},
  title     = {Visual Translation Embedding Network for Visual Relation Detection},
  booktitle = {CVPR},
  year      = {2017},
}
```

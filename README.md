This implements "Visual Translation Embedding Network for Visual Relation Detection,Hanwang Zhang, Zawlin Kyaw, Shih-Fu Chang, Tat-Seng Chua (CVPR2017)"

#### What's inside?
* object detectors training and data preprocessor for vrd and vg.
* vg canonicalization
* relation training code for vrd and vg.

#### Coming soon
* setup instructions
* pretrained models

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

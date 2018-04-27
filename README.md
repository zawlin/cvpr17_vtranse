This implements "Visual Translation Embedding Network for Visual Relation Detection,Hanwang Zhang, Zawlin Kyaw, Shih-Fu Chang, Tat-Seng Chua (CVPR2017)"

Recently there is also a tensorflow adaption provided by yangxuntu, which obtain significant improvement on vg dataset. You can find the code here(https://github.com/yangxuntu/vtranse)
#### What's inside?
* object detectors training and data preprocessor for vrd and vg.
* vg canonicalization
* two-stage relation training code for vrd and vg.
* evaluation code for vg+vrd adapted from Lu (https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection)

#### Download links
The files are in google drive. The direct link to the folder is at https://drive.google.com/open?id=1BvtjCnlORMg4l92kNgZ2g1YaHYj9Dy3X

* annotations in hdf5 format(Since quite a few have asked, for vrd, this is the same as the original mat-based data format. For vg, this is our own cleaned up set with 200 object categories and 100 predicates. The data is organized into train and test split for both datasets). If you come here for the dataset, this is the one.
    * https://drive.google.com/open?id=1BzP8DN2MAz76IvQTlpNOYla_bNC9gQuN
    * https://drive.google.com/open?id=1C6MDiqWQupMrPOgk4T12zWiAJAZmY1aa
* voc format for obj detector
    * https://drive.google.com/open?id=1BzdHWwU7lBLzDGIQfqWUEIOfyWZbeSqD
    * https://drive.google.com/open?id=1CDsj8hsdKP8vtXhWXPNZFy2CaJ5ef-Zo
* object detectors
    * https://drive.google.com/open?id=1CJ8MeAo6Gy1Mf441ODutflHwm6XSwqrw
    * https://drive.google.com/open?id=1CL31Pg1Td2AEiSq-KKC13bM3l6AWxgQ7
* relation model
    * https://drive.google.com/open?id=1CV1sKAK2IZyT_8z_fsj5GZlLMRUDik3f

#### Coming soon
* end to end training instructions and code
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
### Training And Evaluation Instructions
I am using ubuntu 16.04  with gcc 5.4. If you run into protobuf errors, usually recompiling protobuf from source will eliminate the errors. When I refer to folders, it is with respect to the root github source folder.

The steps below are for vrd dataset. For vg, the steps are similar, you will just need to change the some folder or file paths to point to vg directory or scripts.

* First clone the repo
	* `git clone git@github.com:zawlin/cvpr17_vtranse.git`
	* `git submodule update --recursive`
* cd into caffe-fast-rcnn folder to build caffe. This step is the same as building py-faster-rcnn. You also need to copy  `Makefile.config.ubuntu16` to `Makefile.config` before you run `make` command.
* Prepare data by creating symbolic links under data folder as described in the previous section. You will also need to copy the dataset hdf5 files to `data` folder and the object detector models into the `model` folder.
* Make nms module.
	* `cd lib;make`
	* cython is required for this step. you can install it by `pip install cython --user`.
* A working object detector is required at this stage, please refer to py-faster-rcnn training instructions. Please note that the number of anchors needs to be modified to successfuly train the detector.
* Next is to generate visual features for training relation model.
	* `python lib/vrd/save_cache.py`
	* After this step `sg_vrd_2016_test.hdf5` and `sg_vrd_2016_train.hdf5` should be generated under `output` folder if you are training for vrd dataset.
* Run the visual relation training.
	* `python lib/vrd/run_train_relation.py`
* Generate results in matlab mat format for evaluation. 
	* `python lib/vrd/run_relation_test.py`
	* this will generate a file under  output, e.g., sg_vrd_2016_result_all_50000.mat.
* Copy the generated mat file to `relation_evaluation/vr/data/vtranse_results.mat`
	* `cp output/sg_vrd_2016_result_all_50000.mat relation_evaluation/vr/data/vtranse_results.mat`
* Run the matlab evaluation script at `relation_evaluation/vr/eval_vtranse.m`

### Citation

If you're using this code in a scientific publication please cite:
```
@inproceedings{Zhang_2017_CVPR,
  author    = {Hanwang Zhang, Zawlin Kyaw, Shih-Fu Chang, Tat-Seng Chua},
  title     = {Visual Translation Embedding Network for Visual Relation Detection},
  booktitle = {CVPR},
  year      = {2017},
}
```

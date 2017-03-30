#!/bin/bash
nohup python -u tools/train_net.py --gpu 1 --solver models/sg_vrd/vgg16/faster_rcnn_end2end/solver.prototxt --imdb  sg_vrd_2016_train   --iters 700000   --cfg experiments/cfgs/faster_rcnn_end2end.yml 2>&1 >>sg_vrd_obj.out&
	

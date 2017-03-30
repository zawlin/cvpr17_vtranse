#!/bin/bash
nohup python -u tools/train_net.py --gpu 1 --solver models/vg1_2/vgg16/faster_rcnn_end2end/solver.prototxt --imdb  vg1_2_2016_train --iters 700000   --cfg experiments/cfgs/faster_rcnn_end2end.yml 2>&1 >>output/logs/vg1_2_obj.out&
	

%   This file is for Predicting <subject, predicate, object> phrase and relationship

%   Distribution code Version 1.0 -- Copyright 2016, AI lab @ Stanford University.
%   
%   The Code is created based on the method described in the following paper 
%   [1] "Visual Relationship Detection with Language Priors",
%   Cewu Lu*, Ranjay Krishna*, Michael Bernstein, Li Fei-Fei, European Conference on Computer Vision, 
%   (ECCV 2016), 2016(oral). (* = indicates equal contribution)
%  
%   The code and the algorithm are for non-comercial use only.

%% data loading
addpath('evaluation');
load('data/objectListN.mat'); 
% given a object category index and ouput the name of it.

load('data/Wb.mat');
load('data/vtranse_results.mat');
%% computing Phrase Det. and Relationship Det. accuracy

fprintf('\n');
fprintf('#######  Top recall results  ####### \n');
recall100P = top_recall_Phrase(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
recall50P = top_recall_Phrase(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours); 
fprintf('Phrase Det. R@100: %0.2f \n', 100*recall100P);
fprintf('Phrase Det. R@50: %0.2f \n', 100*recall50P);

recall100R = top_recall_Relationship(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
recall50R = top_recall_Relationship(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
fprintf('Relationship Det. R@100: %0.2f \n', 100*recall100R);
fprintf('Relationship Det. R@50: %0.2f \n', 100*recall50R);

fprintf('\n');
fprintf('#######  Zero-shot results  ####### \n');
zeroShot100P = zeroShot_top_recall_Phrase(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
zeroShot50P = zeroShot_top_recall_Phrase(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
fprintf('zero-shot Phrase Det. R@100: %0.2f \n', 100*zeroShot100P);
fprintf('zero-shot Phrase Det. R@50: %0.2f \n', 100*zeroShot50P);

zeroShot100R = zeroShot_top_recall_Relationship(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
zeroShot50R = zeroShot_top_recall_Relationship(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
fprintf('zero-shot Relationship Det. R@100: %0.2f \n', 100*zeroShot100R);
fprintf('zero-shot Relationship Det. R@50: %0.2f \n', 100*zeroShot50R);
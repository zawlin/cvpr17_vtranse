
%% computing Phrase Det. and Relationship Det. accuracy

addpath('evaluation');
fprintf('\n');
% recall100P = top_recall_Phrase(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
% recall50P = top_recall_Phrase(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours); 
% fprintf('%0.2f \n', 100*recall100P);
% fprintf('%0.2f \n', 100*recall50P);
% 
recall100R = top_recall_Phrase_vp(100, rlp_confs_ours, rlp_labels_ours, bboxes_ours);
recall50R = top_recall_Phrase_vp(50, rlp_confs_ours, rlp_labels_ours, bboxes_ours);
fprintf('%0.2f \n', 100*recall100R);
fprintf('%0.2f \n', 100*recall50R);

% recall100R = top_recall_Relationship(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
% recall50R = top_recall_Relationship(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
% fprintf('%0.2f \n', 100*recall100R);
% fprintf('%0.2f \n', 100*recall50R);
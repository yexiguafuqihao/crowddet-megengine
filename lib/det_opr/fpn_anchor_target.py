import megengine as mge
import megengine.random as rand
import megengine.functional as F
import numpy as np
from config import config
from det_opr.bbox_opr import box_overlap_opr, bbox_transform_opr
import pdb
def fpn_rpn_reshape(pred_cls_score_list, pred_bbox_offsets_list):

    final_pred_bbox_offsets_list = []
    final_pred_cls_score_list = []
    batch_per_gpu = pred_cls_score_list[0].shape[0]
    for bid in range(batch_per_gpu):
        batch_pred_bbox_offsets_list = []
        batch_pred_cls_score_list = []
        for i in range(len(pred_cls_score_list)):
            pred_cls_score_perlvl = pred_cls_score_list[i][bid] \
                .transpose(1, 2, 0).reshape(-1, 2)
            pred_bbox_offsets_perlvl = pred_bbox_offsets_list[i][bid] \
                .transpose(1, 2, 0).reshape(-1, 4)
            batch_pred_cls_score_list.append(pred_cls_score_perlvl)
            batch_pred_bbox_offsets_list.append(pred_bbox_offsets_perlvl)
        batch_pred_cls_score = F.concat(batch_pred_cls_score_list, axis=0)
        batch_pred_bbox_offsets = F.concat(batch_pred_bbox_offsets_list, axis=0)
        final_pred_cls_score_list.append(batch_pred_cls_score)
        final_pred_bbox_offsets_list.append(batch_pred_bbox_offsets)

    final_pred_cls_score = F.concat(final_pred_cls_score_list, axis=0)
    final_pred_bbox_offsets = F.concat(final_pred_bbox_offsets_list, axis=0)
    
    return final_pred_cls_score, final_pred_bbox_offsets

def fpn_anchor_target_opr_core_impl(
        gt_boxes, im_info, anchors, allow_low_quality_matches=True):
    
    ignore_label = config.ignore_label
    # get the gt boxes
    gtboxes = gt_boxes[:im_info[5].astype(np.int32)]
    ignore_mask = F.equal(gtboxes[:, 4], config.ignore_label)

    # find the valid gtboxes
    _, index = F.cond_take(1 - ignore_mask > 0, ignore_mask)
    valid_gt_boxes = gtboxes[index.astype(np.int32)]
    
    # compute the iou matrix
    overlaps = box_overlap_opr(anchors, valid_gt_boxes[:, :4])
    # match the dtboxes
    a_shp0 = anchors.shape[0]
    argmax_overlaps = F.argmax(overlaps, axis=1)
    max_overlaps = F.nn.indexing_one_hot(overlaps, argmax_overlaps.astype(np.int32), 1)
    
    labels = F.ones(a_shp0).astype(np.int32) * ignore_label
    # set negative ones
    labels = labels * (max_overlaps >= config.rpn_negative_overlap).astype(np.float32)

    # set positive ones
    fg_mask = (max_overlaps >= config.rpn_positive_overlap)
    const_one = mge.tensor(1.0)
     
    if allow_low_quality_matches:

        # match the max gt
        gt_max_overlaps = F.max(overlaps, axis=0)
        gt_argmax_overlaps = F.argmax(overlaps, axis=0)
        gt_argmax_overlaps = gt_argmax_overlaps.astype(np.int32)
        
        max_overlaps[gt_argmax_overlaps] = 1.
        m = gt_max_overlaps.shape[0]
        argmax_overlaps[gt_argmax_overlaps] = F.linspace(0, m - 1, m).astype(np.int32)
        fg_mask = (max_overlaps >= config.rpn_positive_overlap)
        
    labels[fg_mask] = 1
    # compute the bbox targets
    bbox_targets = bbox_transform_opr(
        anchors, valid_gt_boxes[argmax_overlaps, :4])
    if config.rpn_bbox_normalize_targets:

        std_opr = mge.tensor(config.bbox_normalize_stds[None, :]).to(anchors.device)
        mean_opr = mge.tensor(config.bbox_normalize_means[None, :]).to(anchors.device)
        minus_opr = mean_opr / std_opr
        bbox_targets = bbox_targets / std_opr - minus_opr
    return labels, bbox_targets

def fpn_anchor_target(boxes, im_info, all_anchors_list):
    final_labels_list = []
    final_bbox_targets_list = []
    batch_per_gpu = boxes.shape[0]
    for bid in range(batch_per_gpu):
        batch_labels_list = []
        batch_bbox_targets_list = []
        for i in range(len(all_anchors_list)):
            anchors_perlvl = all_anchors_list[i]
            rpn_labels_perlvl, rpn_bbox_targets_perlvl = fpn_anchor_target_opr_core_impl(
                boxes[bid], im_info[bid], anchors_perlvl)
            batch_labels_list.append(rpn_labels_perlvl)
            batch_bbox_targets_list.append(rpn_bbox_targets_perlvl)
        # here we samples the rpn_labels
        concated_batch_labels = F.concat(batch_labels_list, axis=0)
        concated_batch_bbox_targets = F.concat(batch_bbox_targets_list, axis=0)
        # sample labels
        num_positive = config.num_sample_anchors * config.positive_anchor_ratio
        concated_batch_labels = _bernoulli_sample_labels(concated_batch_labels,
                num_positive, 1, config.ignore_label)
        num_positive = F.equal(concated_batch_labels, 1).sum()
        num_negative = config.num_sample_anchors - num_positive
        concated_batch_labels = _bernoulli_sample_labels(concated_batch_labels,
                num_negative, 0, config.ignore_label)

        final_labels_list.append(concated_batch_labels)
        final_bbox_targets_list.append(concated_batch_bbox_targets)
    final_labels = F.concat(final_labels_list, axis=0)
    final_bbox_targets = F.concat(final_bbox_targets_list, axis=0)
    bbox_targets, labels = final_bbox_targets.detach(), final_labels.detach()
    return labels, bbox_targets

def _bernoulli_sample_labels(
        labels, num_samples, sample_value, ignore_label=-1):
    """ Using the bernoulli sampling method"""
    sample_label_mask = F.equal(labels, sample_value)
    num_mask = sample_label_mask.sum()
    num_final_samples = F.minimum(num_mask, num_samples)
    # here, we use the bernoulli probability to sample the anchors
    sample_prob = num_final_samples / num_mask
    uniform_rng = rand.uniform(0, 1, sample_label_mask.shape)
    disable_mask = (uniform_rng >= sample_prob) * sample_label_mask
    #TODO check cudaerror: illegal memory access was encountered
    labels = labels * (1 - disable_mask) + disable_mask * ignore_label

    return labels


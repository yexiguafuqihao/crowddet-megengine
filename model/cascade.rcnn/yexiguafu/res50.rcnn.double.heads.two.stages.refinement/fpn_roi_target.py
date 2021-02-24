# -*- coding: utf-8 -*-
import megengine as mge
import megengine.random as rand
import megengine.functional as F

import numpy as np
from config import config
# from det_opr.utils import mask_to_inds
from bbox_opr import box_overlap_opr, bbox_transform_opr, box_overlap_ignore_opr
import pdb

def fpn_roi_target(rpn_rois, im_info, gt_boxes, fg_threshold = config.fg_threshold, top_k=1):
    # return_rois = []
    # return_labels = []

    return_rois, return_labels = [], []
    return_bbox_targets = []
    # get per image proposals and gt_boxes
    batch_per_gpu = im_info.shape[0]
    #sampling = fg_threshold <= 0.5
    sampling = True
    # is_sample = True if top_k < 2 else False
    for bid in range(batch_per_gpu):

        gt_boxes_perimg = gt_boxes[bid, :im_info[bid, 5].astype(np.int32), :]
        dummy_gt = F.ones([1, gt_boxes_perimg.shape[1]])

        batch_inds = F.ones((gt_boxes_perimg.shape[0], 1)) * bid
        #if config.proposal_append_gt:
        gt_rois = F.concat([batch_inds, gt_boxes_perimg[:, :4]], axis=1)
        batch_rois_mask = F.equal(rpn_rois[:, 0], bid) > 0
        _, batch_rois_index = F.cond_take(batch_rois_mask, batch_rois_mask)
        
        # batch_roi_mask = rpn_rois[:, 0] == bid
        # batch_roi_inds = mask_to_inds(batch_roi_mask)
        all_rois= F.concat([rpn_rois[batch_rois_index], gt_rois], axis=0) if sampling \
            else rpn_rois[batch_rois_index]
        # all_rois = F.concat([rpn_rois.ai[batch_roi_inds], gt_rois], axis=0)

        gt_boxes_perimg = F.concat([gt_boxes_perimg, dummy_gt],axis=0)
        overlaps_normal, overlaps_ignore = box_overlap_ignore_opr(
                all_rois[:, 1:5], gt_boxes_perimg)

        # overlaps_normal, overlaps_normal_indices = F.argsort(overlaps_normal, descending=True)
        # overlaps_ignore, overlaps_ignore_indices = F.argsort(overlaps_ignore, descending=True)
        overlaps_normal_indices = F.argsort(overlaps_normal, descending=True)
        overlaps_normal = F.gather(overlaps_normal, 1, overlaps_normal_indices)
        # overlaps_normal = F.nn.indexing_one_hot(overlaps_normal, overlaps_normal_indices, 1)
        overlaps_ignore_indices = F.argsort(overlaps_ignore, descending = True)
        overlaps_ignore = F.gather(overlaps_ignore, 1, overlaps_ignore_indices)
        # overlaps_ignore = F.nn.indexing_one_hot(overlaps_ignore, overlaps_ignore_indices, 1)


        # gt max and indices, ignore max and indices
        max_overlaps_normal = overlaps_normal[:, :top_k].flatten()
        gt_assignment_normal = overlaps_normal_indices[:, :top_k].flatten()
        max_overlaps_ignore = overlaps_ignore[:, :top_k].flatten()
        gt_assignment_ignore = overlaps_ignore_indices[:, :top_k].flatten()
        # cons masks
        
        ignore_assign_mask = (max_overlaps_normal < fg_threshold).astype(np.float32) * (
                max_overlaps_ignore > max_overlaps_normal).astype(np.float32)
        max_overlaps = max_overlaps_normal * (1 - ignore_assign_mask).astype(np.float32) + \
                max_overlaps_ignore * ignore_assign_mask
        

        gt_assignment = gt_assignment_normal * (1- ignore_assign_mask) + \
                gt_assignment_ignore * ignore_assign_mask
        
        gt_assignment = gt_assignment.astype(np.int32)

        labels = gt_boxes_perimg[gt_assignment, 4]
        fg_mask = (max_overlaps >= fg_threshold).astype(np.float32) * (1 - F.equal(labels, config.ignore_label))
        bg_mask = (max_overlaps < config.bg_threshold_high).astype(np.float32) * (
                max_overlaps >= config.bg_threshold_low).astype(np.float32)

        fg_mask = fg_mask.reshape(-1, top_k)
        bg_mask = bg_mask.reshape(-1, top_k)
        pos_max = config.num_rois * config.fg_ratio
        fg_inds_mask = _bernoulli_sample_masks(fg_mask[:, 0], pos_max, 1) if sampling else F.equal(fg_mask[:, 0], 0)
        neg_max = config.num_rois - fg_inds_mask.sum()
        bg_inds_mask = _bernoulli_sample_masks(bg_mask[:, 0], neg_max, 1) if sampling else F.equal(bg_mask[:, 0], 0)
        labels = labels * fg_mask.reshape(-1)

        keep_mask = fg_inds_mask + bg_inds_mask
        keep_mask = keep_mask + F.equal(keep_mask.sum(), 0)
        # keep_inds = mask_to_inds(keep_mask)
        _, keep_inds = F.cond_take(keep_mask > 0, keep_mask)
        #keep_inds = keep_inds[:F.minimum(config.num_rois, keep_inds.shapeof()[0])]
        # labels
        labels = labels.reshape(-1, top_k)[keep_inds]
        gt_assignment = gt_assignment.reshape(-1, top_k)[keep_inds].reshape(-1).astype(np.int32)
        target_boxes = gt_boxes_perimg[gt_assignment, :4]
        # rois = all_rois.ai[keep_inds]
        rois = all_rois[keep_inds]
        # target_shape = (rois.shapeof()[0], top_k, rois.shapeof()[-1])
        n, c = rois.shape[0], rois.shape[1]
        target_rois = F.broadcast_to(F.expand_dims(rois, 1), (n, top_k, c)).reshape(-1, c)
        # target_rois = F.add_axis(rois, 1).broadcast(target_shape).reshape(-1, rois.shapeof()[-1])
        bbox_targets = bbox_transform_opr(target_rois[:, 1:5], target_boxes[:, :4])
        if config.rcnn_bbox_normalize_targets:
            std_opr = mge.tensor(config.bbox_normalize_stds[None, :]).to(rois.device)
            mean_opr = mge.tensor(config.bbox_normalize_means[None, :]).to(rois.device)
            minus_opr = mean_opr / std_opr
            bbox_targets = bbox_targets / std_opr - minus_opr
        bbox_targets = bbox_targets.reshape(-1, top_k * 4)
        return_rois.append(rois)
        return_labels.append(labels)
        return_bbox_targets.append(bbox_targets)
    if config.batch_per_gpu == 1:
        rois, labels, bbox_targets = rois.detach(), labels.detach(), bbox_targets.detach()
        return rois, labels, bbox_targets
        # return F.zero_grad(rois), F.zero_grad(labels), F.zero_grad(bbox_targets)
    else:
        return_rois = F.concat(return_rois, axis=0)
        return_labels = F.concat(return_labels, axis=0)
        return_bbox_targets = F.concat(return_bbox_targets, axis=0)

        return_rois = return_rois.detach()
        return_labels = return_labels.detach()
        return_bbox_targets = return_bbox_targets.detach()
        return return_rois, return_labels, return_bbox_targets
        # rois, labels, bbox_targets = return_rois.detach(), return_labels.detach(), return_bbox_targets.detach()
        # return rois, labels, bbox_targets
        # return F.zero_grad(return_rois), F.zero_grad(return_labels), F.zero_grad(return_bbox_targets)

def _bernoulli_sample_masks(masks, num_samples, sample_value):
    """ Using the bernoulli sampling method"""
    sample_mask = F.equal(masks, sample_value)
    num_mask = sample_mask.sum()
    num_final_samples = F.minimum(num_mask, num_samples)
    # here, we use the bernoulli probability to sample the anchors
    sample_prob = num_final_samples / num_mask
    # uniform_rng = rand.uniform(sample_mask.shapeof()[0])
    uniform_rng = rand.uniform(0, 1, sample_mask.shape)
    after_sampled_mask = (uniform_rng <= sample_prob) * sample_mask
    return after_sampled_mask


import megengine as mge
import megengine.functional as F
from megengine import tensor
import numpy as np
from megengine.functional.nn import nms
from config import config
from det_opr.bbox_opr import bbox_transform_inv_opr, clip_boxes_opr, \
    filter_boxes_opr, box_overlap_opr
# from bbox_opr import box_overlap_opr
import pdb
def find_top_rpn_proposals(is_train, rpn_bbox_offsets_list, rpn_cls_prob_list,
        all_anchors_list, im_info):
    prev_nms_top_n = config.train_prev_nms_top_n \
        if is_train else config.test_prev_nms_top_n
    post_nms_top_n = config.train_post_nms_top_n \
        if is_train else config.test_post_nms_top_n
    batch_per_gpu = config.batch_per_gpu if is_train else 1
    nms_threshold = config.rpn_nms_threshold
    box_min_size = config.rpn_min_box_size
    bbox_normalize_targets = config.rpn_bbox_normalize_targets
    bbox_normalize_means = config.bbox_normalize_means
    bbox_normalize_stds = config.bbox_normalize_stds

    list_size = len(rpn_bbox_offsets_list)

    return_rois, return_probs = [], []
    batch_per_gpu = rpn_cls_prob_list[0].shape[0]
    for bid in range(batch_per_gpu):
        batch_proposals_list = []
        batch_probs_list = []
        for l in range(list_size):
            # get proposals and probs
            offsets = rpn_bbox_offsets_list[l][bid] \
                .transpose(1, 2, 0).reshape(-1, 4)
            if bbox_normalize_targets:
                std_opr = tensor(config.bbox_normalize_stds[None, :])
                mean_opr = tensor(config.bbox_normalize_means[None, :])
                pred_offsets = pred_offsets * std_opr
                pred_offsets = pred_offsets + mean_opr
            all_anchors = all_anchors_list[l]

            proposals = bbox_transform_inv_opr(all_anchors, offsets)
            if config.anchor_within_border:
                proposals = clip_boxes_opr(proposals, im_info[bid, :])
            probs = rpn_cls_prob_list[l][bid] \
                    .transpose(1,2,0).reshape(-1, 2)
            probs = F.softmax(probs)[:, 1]
            # gather the proposals and probs
            batch_proposals_list.append(proposals)
            batch_probs_list.append(probs)

        
        batch_proposals = F.concat(batch_proposals_list, axis=0)
        batch_probs = F.concat(batch_probs_list, axis=0)
        # filter the boxes with small size.
        wh = batch_proposals[:, 2:4] - batch_proposals[:, :2] + 1
        thresh = box_min_size * im_info[bid, 2]
        keep_mask = F.prod((wh >= thresh), axis=1)
        keep_mask = keep_mask + F.equal(keep_mask.sum(), 0)
        keep_mask, inds = F.cond_take(keep_mask > 0, keep_mask)

        inds = inds.astype(np.int32)
        # batch_proposals = F.nn.indexing_one_hot(batch_proposals, inds, 0)
        # batch_probs = F.nn.indexing_one_hot(batch_probs, inds, 0)
        batch_proposals, batch_probs = batch_proposals[inds], batch_probs[inds]

        # prev_nms_top_n
        num_proposals = F.minimum(prev_nms_top_n, batch_proposals.shape[0])
        idx = F.argsort(batch_probs, descending=True)
        topk_idx = idx[:num_proposals].reshape(-1)
        batch_proposals = batch_proposals[topk_idx].detach()
        batch_probs = batch_probs[topk_idx].detach()
        
        # For each image, run a total-level NMS, and choose topk results.
        keep_inds = nms(batch_proposals, batch_probs, nms_threshold, max_output = 2000)
        # num = F.minimum(post_nms_top_n, keep_inds.shape[0])
        # keep_inds = keep_inds[:num]

        batch_rois, batch_probs = batch_proposals[keep_inds], batch_probs[keep_inds]

        # cons the rois
        batch_inds = F.ones((batch_rois.shape[0], 1)) * bid
        batch_rois = F.concat([batch_inds, batch_rois[:, :4]], axis=1)
        return_rois.append(batch_rois)
        return_probs.append(batch_probs)

    if batch_per_gpu == 1:
        return batch_rois, batch_probs
    else:
        concated_rois = F.concat(return_rois, axis=0)
        concated_probs = F.concat(return_probs, axis=0)
        return concated_rois, concated_probs

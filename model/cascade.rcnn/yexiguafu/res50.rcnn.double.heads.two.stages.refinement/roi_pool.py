import math
import numpy as np
import megengine as mge
import megengine.functional as F
import pdb
def roi_pool(rpn_fms, rois, stride, pool_shape, roi_type='roi_align', 
             labels=None, bbox_targets=None):

    assert len(stride) == len(rpn_fms)
    canonical_level = 4
    canonical_box_size = 224
    min_level = math.log2(stride[0])
    max_level = math.log2(stride[-1])

    num_fms = len(rpn_fms)
    box_sizes = F.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    level_assignments = F.floor(
	canonical_level + F.log(box_sizes / canonical_box_size) / np.log(2)
    )
    level_assignments = F.minimum(level_assignments, max_level)
    level_assignments = F.maximum(level_assignments, min_level)
    level_assignments = level_assignments - min_level
    available_masks = F.concat(
        [F.ones(level_assignments.shape[0]), F.zeros(num_fms)], axis=0)
    level_assignments = F.concat([level_assignments, mge.tensor(np.arange(num_fms, dtype=np.int32))], axis=0)
    rois = F.concat([rois, F.zeros((num_fms, rois.shape[-1]))], axis=0)

    if labels is not None and bbox_targets is not None:
        labels = F.concat([labels, F.ones((num_fms, labels.shape[-1]))], axis=0)
        bbox_targets = F.concat([bbox_targets, F.zeros((num_fms, bbox_targets.shape[-1]))], axis=0)
    
    pool_list, inds_list = [], []
    for i in range(len(rpn_fms)):
        # mask = level_assignments == i
        # inds = mask_to_inds(mask)
        mask = F.equal(level_assignments, i)
        _, inds = F.cond_take(mask > 0, mask)
        rois_fm = rois[inds.astype(np.int32)]
        if roi_type == 'roi_pool':
            pool_fm = F.nn.roi_pooling(
                    rpn_fms[i], rois_fm, pool_shape, mode='max', scale=1.0/stride[i])
        elif roi_type == 'roi_align':
            pool_fm = F.nn.roi_align(
                    rpn_fms[i], rois_fm, pool_shape, mode='average', 
                    spatial_scale=1.0/stride[i], sample_points=2, aligned=True)
        pool_list.append(pool_fm)
        inds_list.append(inds)

    fm_order = F.concat(inds_list, axis=0)
    pool_feature = F.concat(pool_list, axis=0)

    ordered_available_masks = available_masks[fm_order]
    # available_inds = mask_to_inds(ordered_available_masks)
    _, available_inds = F.cond_take(ordered_available_masks > 0, ordered_available_masks)
    available_inds = available_inds.astype(np.int32)
    pool_feature = pool_feature[available_inds.astype(np.int32)]
    rois = rois[fm_order, :][available_inds.astype(np.int32)]
    if labels is not None:
        labels = labels[fm_order][available_inds]
        bbox_targets = bbox_targets[fm_order][available_inds]
        return pool_feature, rois, labels.detach(), bbox_targets.detach()
    else:
        return pool_feature, rois, None, None


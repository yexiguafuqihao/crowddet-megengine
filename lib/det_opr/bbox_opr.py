import math
import megengine.functional as F
import numpy as np
from megengine import Tensor
import pdb

def restore_bbox(rois, deltas, unnormalize=True):

    assert deltas.ndim == 3
    if unnormalize:
        std_opr = mge.tensor(config.bbox_normalize_stds.reshape(1, 1, -1))
        mean_opr = mge.tensor(config.bbox_normalize_means.reshape(1, 1, -1))
        deltas = deltas * std_opr
        deltas = deltas + mean_opr

    # n = deltas.shape[1]
    n, c = deltas.shape[0], deltas.shape[1]
    all_rois = F.broadcast_to(F.expand_dims(rois, 1), (n, c, rois.shape[1])).reshape(-1, rois.shape[1])
    deltas = deltas.reshape(-1, deltas.shape[2])
    pred_bbox = bbox_transform_inv_opr(all_rois, deltas)
    pred_bbox = pred_bbox.reshape(-1, c, pred_bbox.shape[1])
    return pred_bbox

def filter_boxes_opr(boxes, min_size):

    """Remove all boxes with any side smaller than min_size."""
    wh = boxes[:, 2:4] - boxes[:, 0:2] + 1
    keep_mask = F.prod(wh >= min_size, axis = 1).astype(np.float32)
    keep_mask = keep_mask + F.equal(keep_mask.sum(), 0).astype(np.float32)
    return keep

def clip_boxes_opr(boxes, im_info):
    """ Clip the boxes into the image region."""
    w = im_info[1] - 1
    h = im_info[0] - 1
    boxes[:, 0::4] = boxes[:, 0::4].clamp(min=0, max=w)
    boxes[:, 1::4] = boxes[:, 1::4].clamp(min=0, max=h)
    boxes[:, 2::4] = boxes[:, 2::4].clamp(min=0, max=w)
    boxes[:, 3::4] = boxes[:, 3::4].clamp(min=0, max=h)
    return boxes

def bbox_transform_inv_opr(bbox, deltas):
    max_delta = math.log(1000.0 / 16)
    """ Transforms the learned deltas to the final bbox coordinates, the axis is 1"""
    bbox_width = bbox[:, 2] - bbox[:, 0] + 1
    bbox_height = bbox[:, 3] - bbox[:, 1] + 1
    bbox_ctr_x = bbox[:, 0] + 0.5 * bbox_width
    bbox_ctr_y = bbox[:, 1] + 0.5 * bbox_height
    pred_ctr_x = bbox_ctr_x + deltas[:, 0] * bbox_width
    pred_ctr_y = bbox_ctr_y + deltas[:, 1] * bbox_height

    dw = deltas[:, 2]
    dh = deltas[:, 3]
    dw = F.minimum(dw, max_delta)
    dh = F.minimum(dh, max_delta)
    pred_width = bbox_width * F.exp(dw)
    pred_height = bbox_height * F.exp(dh)

    pred_x1 = pred_ctr_x - 0.5 * pred_width
    pred_y1 = pred_ctr_y - 0.5 * pred_height
    pred_x2 = pred_ctr_x + 0.5 * pred_width
    pred_y2 = pred_ctr_y + 0.5 * pred_height
    # pred_boxes = F.concat((pred_x1.reshape(-1, 1), pred_y1.reshape(-1, 1),
    #                         pred_x2.reshape(-1, 1), pred_y2.reshape(-1, 1)), axis=1)
    pred_boxes = F.stack([pred_x1, pred_y1, pred_x2, pred_y2], axis = 1)
    return pred_boxes

def bbox_transform_opr(bbox, gt):
    """ Transform the bounding box and ground truth to the loss targets.
    The 4 box coordinates are in axis 1"""
    bbox_width = bbox[:, 2] - bbox[:, 0] + 1
    bbox_height = bbox[:, 3] - bbox[:, 1] + 1
    bbox_ctr_x = bbox[:, 0] + 0.5 * bbox_width
    bbox_ctr_y = bbox[:, 1] + 0.5 * bbox_height

    gt_width = gt[:, 2] - gt[:, 0] + 1
    gt_height = gt[:, 3] - gt[:, 1] + 1
    gt_ctr_x = gt[:, 0] + 0.5 * gt_width
    gt_ctr_y = gt[:, 1] + 0.5 * gt_height

    target_dx = (gt_ctr_x - bbox_ctr_x) / bbox_width
    target_dy = (gt_ctr_y - bbox_ctr_y) / bbox_height
    target_dw = F.log(gt_width / bbox_width)
    target_dh = F.log(gt_height / bbox_height)
    target = F.stack([target_dx, target_dy, target_dw, target_dh], axis=1)
    return target

def box_overlap_opr(box: Tensor, gt: Tensor) -> Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    # box = boxes1
    # gt = boxes2
    # target_shape = (boxes1.shape[0], boxes2.shape[0], 4)

    N, K = box.shape[0], gt.shape[0]
    b_box = F.broadcast_to(F.expand_dims(box, 1),(N, K, box.shape[1]))
    b_gt = F.broadcast_to(F.expand_dims(gt, 0), (N, K, gt.shape[1]))
    # b_gt = F.expand_dims(gt, 0).broadcast_to(N, K, gt.shape[1])

    # b_box = F.expand_dims(boxes1, 1).broadcast(*target_shape)
    # b_gt = F.expand_dims(boxes2, 0).broadcast(*target_shape)

    iw = F.minimum(b_box[:, :, 2], b_gt[:, :, 2]) - F.maximum(
        b_box[:, :, 0], b_gt[:, :, 0]
    )
    ih = F.minimum(b_box[:, :, 3], b_gt[:, :, 3]) - F.maximum(
        b_box[:, :, 1], b_gt[:, :, 1]
    )
    inter = F.maximum(iw, 0) * F.maximum(ih, 0)

    area_box = F.maximum(box[:, 2] - box[:, 0], 0) * F.maximum(box[:, 3] - box[:, 1], 0)
    area_gt = F.maximum(gt[:, 2] - gt[:, 0], 0) * F.maximum(gt[:, 3] - gt[:, 1], 0)

    # area_target_shape = (box.shape[0], gt.shapeof()[0])
    b_area_box = F.broadcast_to(F.expand_dims(area_box, 1), (N, K))
    b_area_gt = F.broadcast_to(F.expand_dims(area_gt, 0), (N, K))
    # b_area_box = F.expand_dims(area_box, 1).broadcast_to(N, K)
    # b_area_gt = F.expand_dims(area_gt, 0).broadcast_to(N, K)
    # b_area_box = F.add_axis(area_box, 1).broadcast(*area_target_shape)
    # b_area_gt = F.add_axis(area_gt, 0).broadcast(*area_target_shape)

    union = b_area_box + b_area_gt - inter
    overlaps = F.maximum(inter / union, 0)

    return overlaps

def box_overlap_ignore_opr(box: Tensor, gt: Tensor, ignore_label=-1) -> Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    # box = boxes1
    # gt = boxes2
    # target_shape = (boxes1.shapeof()[0], boxes2.shapeof()[0], 4)
    eps = 1e-5
    N, K = box.shape[0], gt.shape[0]
    b_box = F.broadcast_to(F.expand_dims(box, 1), (N, K, box.shape[1]))
    b_gt = F.broadcast_to(F.expand_dims(gt, 0), (N, K, gt.shape[1]))

    # b_box = F.add_axis(boxes1, 1).broadcast(*target_shape)
    # b_gt = F.add_axis(boxes2[:, :4], 0).broadcast(*target_shape)

    iw = F.minimum(b_box[:, :, 2], b_gt[:, :, 2]) - F.maximum(
        b_box[:, :, 0], b_gt[:, :, 0]
    )
    ih = F.minimum(b_box[:, :, 3], b_gt[:, :, 3]) - F.maximum(
        b_box[:, :, 1], b_gt[:, :, 1]
    )
    inter = F.maximum(iw, 0) * F.maximum(ih, 0)

    area_box = F.maximum(box[:, 2] - box[:, 0], 0) * F.maximum(box[:, 3] - box[:, 1], 0)
    area_gt = F.maximum(gt[:, 2] - gt[:, 0], 0) * F.maximum(gt[:, 3] - gt[:, 1], 0)
    # area_target_shape = (box.shapeof()[0], gt.shapeof()[0])
    # b_area_box = F.add_axis(area_box, 1).broadcast(*area_target_shape)
    # b_area_gt = F.add_axis(area_gt, 0).broadcast(*area_target_shape)
    b_area_box = F.broadcast_to(F.expand_dims(area_box, 1), (N, K)) + eps
    b_area_gt = F.broadcast_to(F.expand_dims(area_gt, 0), (N, K))
    union = b_area_box + b_area_gt - inter + eps

    overlaps_normal = F.maximum(inter / union, 0)
    overlaps_ignore = F.maximum(inter / b_area_box, 0)
    overlaps = F.maximum(inter / union, 0)

    # gt_ignore_mask = F.add_axis(F.equal(gt[:, 4], ignore_label), 0).broadcast(*area_target_shape)
    ignore_mask = F.equal(gt[:, 4], ignore_label)
    gt_ignore_mask = F.expand_dims(ignore_mask, 0)
    overlaps_normal *= (1 - gt_ignore_mask)
    overlaps_ignore *= gt_ignore_mask
    return overlaps_normal, overlaps_ignore


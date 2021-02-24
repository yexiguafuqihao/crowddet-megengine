import megengine as mge
import megengine.functional as F
import numpy as np
from megengine import Tensor
import pdb
def softmax_loss(pred, label, ignore_label=-1):

    max_pred = pred.max(axis=1, keepdims=True).detach()
    pred -= max_pred
    log_prob = pred - F.log(F.exp(pred).sum(axis=1, keepdims=True))
    mask = 1 - F.equal(label, ignore_label)
    vlabel = label * mask.astype(np.float32)
    loss = -(F.nn.indexing_one_hot(log_prob, vlabel.astype(np.int32), 1).flatten() * mask)
    loss = loss.sum() / F.maximum(mask.sum(), 1)
    return loss

def softmax_loss_opr(pred, label, ignore_label=-1):

    max_pred = pred.max(axis=1, keepdims=True).detach()
    pred -= max_pred
    log_prob = pred - F.log(F.exp(pred).sum(axis=1, keepdims=True))
    mask = 1 - F.equal(label, ignore_label)
    vlabel = label * mask.astype(np.float32)
    loss = -(F.nn.indexing_one_hot(log_prob, vlabel.astype(np.int32), 1).flatten() * mask)
    return loss

def _smooth_l1_base(pred, gt, sigma):

    sigma2 = sigma ** 2
    cond_point = 1 / sigma2
    x = pred - gt
    abs_x = F.abs(x)
    in_mask = abs_x < cond_point
    out_mask = 1 - in_mask.astype(np.float32)
    in_value = 0.5 * (sigma * x) ** 2
    out_value = abs_x - 0.5 / sigma2
    value = in_value * in_mask.astype(np.float32) + out_value * out_mask
    return value

def _get_mask_of_label(label, background, ignore_label):
    
    mask_fg = 1 - F.equal(label, background).astype(np.float32)
    mask_ig = 1 - F.equal(label, ignore_label).astype(np.float32)
    mask = mask_fg * mask_ig
    return mask, mask_ig

def smooth_l1_loss_rcnn_opr(
        pred, gt, label, sigma = 1, background=0, ignore_label=-1):
    """
        pred    : (minibatch, class_num, 4)
        gt      : (minibatch, 4)
        label   : (minibatch,  )
    """
    broadcast_label = F.broadcast_to(label.reshape(-1, 1), (label.shape[0], pred.shape[-1]))
    broadcast_mask, broadcast_mask_ig = _get_mask_of_label(
        broadcast_label, background, ignore_label)
    vlabel = broadcast_label * broadcast_mask
    pred_corr = F.nn.indexing_one_hot(pred, vlabel.astype(np.int32), 1)
    value = _smooth_l1_base(pred_corr, gt, sigma)
    loss = (value * broadcast_mask).sum(axis=1)
    return loss

def smooth_l1_loss_rpn(pred, gt, label, sigma=1, 
    background=0, ignore_label=-1, axis=1):
    
    value = _smooth_l1_base(pred, gt, sigma)
    mask, mask_ig = _get_mask_of_label(label, background, ignore_label)
    loss = (value.sum(axis = axis) * mask).sum() / F.maximum(mask_ig.sum(), 1)
    return loss

def smooth_l1_loss_rcnn(
        pred, gt, label, sigma = 1, background=0, ignore_label=-1):
    """
        pred    : (minibatch, class_num, 4)
        gt      : (minibatch, 4)
        label   : (minibatch,  )
    """
    broadcast_label = F.broadcast_to(label.reshape(-1, 1), (label.shape[0], pred.shape[-1]))
    broadcast_mask, broadcast_mask_ig = _get_mask_of_label(
        broadcast_label, background, ignore_label)
    vlabel = broadcast_label * broadcast_mask
    pred_corr = F.nn.indexing_one_hot(pred, vlabel.astype(np.int32), 1)
    value = _smooth_l1_base(pred_corr, gt, sigma)
    loss = (value * broadcast_mask).sum() / F.maximum((label>0).sum(), 1)
    return loss

def smooth_l1_loss(pred, target, beta: float):

    abs_x = F.abs(pred - target)
    in_mask = abs_x < beta
    out_mask = 1 - in_mask.astype(np.float32)
    in_loss = 0.5 * abs_x ** 2 / beta
    out_loss = abs_x - 0.5 * beta
    loss = in_loss * in_mask.astype(np.float32) + out_loss * out_mask
    return loss.sum(axis=1)


import numpy as np
import megengine.functional as F
import megengine.module as M
from config import config
from .anchors_generator import AnchorGenerator
from .find_top_rpn_proposals import find_top_rpn_proposals
from .fpn_anchor_target import fpn_anchor_target, fpn_rpn_reshape
from det_opr.loss_opr import softmax_loss, smooth_l1_loss_rpn
import pdb
class RPN(M.Module):
    def __init__(self, rpn_channel=256):
        
        super().__init__()
        self.anchors_generator = AnchorGenerator(
            config.anchor_base_size,
            config.anchor_aspect_ratios,
            config.anchor_base_scale)
        self.rpn_conv = M.Conv2d(256, rpn_channel, kernel_size=3, stride=1, padding=1)
        self.rpn_cls_score = M.Conv2d(rpn_channel, config.num_cell_anchors * 2, kernel_size=1, stride=1)
        self.rpn_bbox_offsets = M.Conv2d(rpn_channel, config.num_cell_anchors * 4, kernel_size=1, stride=1)

        for l in [self.rpn_conv, self.rpn_cls_score, self.rpn_bbox_offsets]:
            M.init.normal_(l.weight, std=0.01)
            M.init.fill_(l.bias, 0)

    def forward(self, features, im_info, boxes=None):
        # prediction
        pred_cls_score_list = []
        pred_bbox_offsets_list = []
        for x in features:
            t = F.relu(self.rpn_conv(x))
            pred_cls_score_list.append(self.rpn_cls_score(t))
            pred_bbox_offsets_list.append(self.rpn_bbox_offsets(t))
        # get anchors
        all_anchors_list = []
        fm_stride = 2 ** (len(features) + 1)
        for fm in features:
            layer_anchors = self.anchors_generator(fm, fm_stride)
            fm_stride = fm_stride // 2
            all_anchors_list.append(layer_anchors)
        # sample from the predictions
        rpn_rois, rpn_probs = find_top_rpn_proposals(
                self.training, pred_bbox_offsets_list, pred_cls_score_list,
                all_anchors_list, im_info)

        if self.training:
            rpn_labels, rpn_bbox_targets = fpn_anchor_target(
                    boxes, im_info, all_anchors_list)
            #rpn_labels = rpn_labels.astype(np.int32)
            pred_cls_score, pred_bbox_offsets = fpn_rpn_reshape(
                pred_cls_score_list, pred_bbox_offsets_list)

            # rpn loss
            rpn_cls_loss = softmax_loss(pred_cls_score, rpn_labels)
            rpn_bbox_loss = smooth_l1_loss_rpn(pred_bbox_offsets, rpn_bbox_targets,   \
                rpn_labels, config.rpn_smooth_l1_beta)

            loss_dict = {}
            loss_dict['loss_rpn_cls'] = rpn_cls_loss
            loss_dict['loss_rpn_loc'] = rpn_bbox_loss
            return rpn_rois, loss_dict

        else:
            return rpn_rois


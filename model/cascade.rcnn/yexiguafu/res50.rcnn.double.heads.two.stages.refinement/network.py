import os, sys
import numpy as np
import megengine as mge
import megengine.functional as F
import megengine.module as M
from config import config
from backbone.resnet50 import ResNet50
from module.rpn import RPN
from layers.roi_pool import roi_pool as roi_pooler
from det_opr.fpn_roi_target import fpn_roi_target
from det_opr.loss_opr import softmax_loss_opr, smooth_l1_loss_rcnn_opr, softmax_loss
from det_opr.utils import get_padded_tensor
from det_opr.bbox_opr import restore_bbox, bbox_transform_inv_opr
import pdb

class FPN(M.Module):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """
    def __init__(self, bottom_up):
        super(FPN, self).__init__()
        in_channels = [256, 512, 1024, 2048]
        fpn_dim = 256
        use_bias =True

        lateral_convs, output_convs = [], []
        for idx, in_channels in enumerate(in_channels):
            lateral_conv = M.Conv2d(
                in_channels, fpn_dim, kernel_size=1, bias=use_bias)
            output_conv = M.Conv2d(
                fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1, bias=use_bias)
            M.init.msra_normal_(lateral_conv.weight, mode="fan_in")
            M.init.msra_normal_(output_conv.weight, mode="fan_in")
            if use_bias:
                M.init.fill_(lateral_conv.bias, 0)
                M.init.fill_(output_conv.bias, 0)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.bottom_up = bottom_up

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        bottom_up_features = bottom_up_features[::-1]
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
            bottom_up_features[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            fh, fw = features.shape[2:]
            top_down_features = F.nn.interpolate(
                prev_features, size = (fh, fw), mode="BILINEAR")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            results.append(output_conv(prev_features))
        # p6
        last_p6 = F.max_pool2d(results[0], kernel_size=1, stride=2, padding=0)
        results.insert(0, last_p6)
        return results

class Network(M.Module):
    def __init__(self):
        
        super().__init__()
        # ----------------------- build the backbone ------------------------ #
        self.resnet50 = ResNet50()
        # ------------ freeze the weights of resnet stage1 and stage 2 ------ #
        if config.backbone_freeze_at >= 1:
            for p in self.resnet50.conv1.parameters():
                p = p.detach()
        if config.backbone_freeze_at >= 2:
            for p in self.resnet50.layer1.parameters():
                p = p.detach()

        # -------------------------- build the FPN -------------------------- #
        self.backbone = FPN(self.resnet50)
        # -------------------------- build the RPN -------------------------- #
        self.RPN = RPN(config.rpn_channel)
        # ----------------------- build the RCNN head ----------------------- #
        self.RCNN = RCNN()
        # -------------------------- input Tensor --------------------------- #
        self.inputs = {
            "image": mge.tensor(
                np.random.random([2, 3, 224, 224]).astype(np.float32), dtype="float32",
            ),
            "im_info": mge.tensor(
                np.random.random([2, 5]).astype(np.float32), dtype="float32",
            ),
            "gt_boxes": mge.tensor(
                np.random.random([2, 100, 5]).astype(np.float32), dtype="float32",
            ),
        }
    
    def pre_process(self, images):

        mean = config.image_mean.reshape(1, 3, 1, 1).astype(np.float32)
        std = config.image_std.reshape(1, 3, 1, 1).astype(np.float32)
        mean = mge.tensor(mean).to(images.device)
        std = mge.tensor(std).to(images.device)
        normed_images = (images - mean) / std
        normed_images = get_padded_tensor(normed_images, 64)
        return normed_images

    def forward(self, inputs):

        im_info = inputs['im_info']
        gt_boxes = inputs['gt_boxes']
        
        # process the images
        images = self.pre_process(inputs['image'])
        del inputs
        if self.training:
            return self._forward_train(images, im_info, gt_boxes)
        else:
            return self._forward_test(images, im_info)

    def _forward_train(self, image, im_info, gt_boxes):

        loss_dict = {}
        # stride: 64,32,16,8,4, p6->p2
        fpn_fms = self.backbone(image)
        rpn_rois, loss_dict_rpn = \
            self.RPN(fpn_fms, im_info, gt_boxes)
        
        loss_dict_rcnn = self.RCNN(
                fpn_fms, rpn_rois, gt_boxes, im_info)

        loss_dict.update(loss_dict_rpn)
        loss_dict.update(loss_dict_rcnn)
        return loss_dict

    def _forward_test(self, image, im_info):

        fpn_fms = self.backbone(image)
        
        rpn_rois = self.RPN(fpn_fms, im_info)
        
        pred_bbox = self.RCNN(fpn_fms, rpn_rois)
        return pred_bbox

class CascadeRCNN(M.Module):

    def __init__(self, iou_thresh, nheads, stage):

        super().__init__()

        assert iou_thresh >= 0.5 and nheads > 0
        self.iou_thresh = iou_thresh
        self.nheads = nheads
        self.n = config.num_classes
        self.name = 'cascade_stage_{}'.format(stage)
        self.refinement = True if nheads > 1 else False

        self.fc1 = M.Linear(256 * 7 * 7, 1024)
        self.fc2 = M.Linear(1024, 1024)
        self.fc3 = M.Linear(1054, 1024) if self.refinement else None

        self.relu = M.ReLU()

        self.n = config.num_classes
        self.p = M.Linear(1024, 5 * self.n * nheads)
        self.q = M.Linear(1024, 5 * self.n) if self.refinement else None
        self.r = M.Linear(1024, 5 * self.n) if self.refinement else None
        self._init_weights()

    def _init_weights(self):

        for l in [self.fc1, self.fc2]:
            # M.init.normal_(l.weight, std=0.01)
            M.init.msra_uniform_(l.weight, a=1)
            M.init.fill_(l.bias, 0)

        for l in [self.p]:
            M.init.normal_(l.weight, std=0.01)
            M.init.fill_(l.bias, 0)

        if self.refinement: #and self.fc3 is not None and self.q is not None:
            # self.refinement
            M.init.msra_uniform_(self.fc3.weight, a=1)
            M.init.fill_(self.fc3.bias, 0)
            
            a = np.random.normal(0, 0.001, [1024, 4 * self.n])
            b = np.random.normal(0, 0.01, [1024, self.n])
            c = np.hstack([a, b]).transpose()
            
            for l in [self.q, self.r]:
                M.init.fill_(l.weight, c)
                M.init.fill_(l.bias, 0)
    
    def refinement_module(self, prob, fc2):
        
        m = prob.reshape(-1, 5*self.n)
        offsets, scores = m[:, :-self.n], m[:, -self.n:]
        n = offsets.shape[0]
        offsets = offsets.reshape(-1, self.n, 4)
        cls_scores = F.expand_dims(F.softmax(scores, axis=1), axis=2)
        pred_boxes = F.concat([offsets, cls_scores], axis=2)[:, 1]
        n, c = pred_boxes.shape
        pred_boxes = F.broadcast_to(F.expand_dims(pred_boxes, axis=1), (n, 6, c)).reshape(n,-1)

        n, c = fc2.shape
        fc3 = F.broadcast_to(F.expand_dims(fc2, axis=1), (n, 2, c)).reshape(-1, c)
        fc3 = F.concat([fc3, pred_boxes], axis=1)
        fc3 = self.relu(self.fc3(fc3))
        fc3 = fc3.reshape(n, 2, -1).transpose(1, 0, 2)

        a = self.q(fc3[0])
        b = self.r(fc3[1])
        prob = F.stack([a, b], axis=1).reshape(-1, 10*self.n)
        return prob

    def forward(self, fpn_fms, rois, gtboxes=None, im_info = None):

        rpn_fms = fpn_fms[1:]
        rpn_fms.reverse()
        rcnn_rois = rois
        stride = [4, 8, 16, 32]
        if self.training:
            rcnn_rois, labels, bbox_targets = fpn_roi_target(rois, im_info, gtboxes, 
                self.iou_thresh, top_k=self.nheads)

            pool5, rcnn_rois, labels, bbox_targets = roi_pooler(
                rpn_fms, rcnn_rois, stride, (7, 7), 'roi_align',  \
                labels, bbox_targets)
        else:
            pool5, rcnn_rois, _, _ = roi_pooler(rpn_fms, rcnn_rois, stride, (7, 7),
                'roi_align')
        
        pool5 = F.flatten(pool5, start_axis=1)
        fc1 = self.relu(self.fc1(pool5))
        fc2 = self.relu(self.fc2(fc1))
        prob = self.p(fc2)

        if self.refinement:
            final_pred = self.refinement_module(prob, fc2)
            
        loss = {}
        if self.training:
            
            # compute the loss function and then return 
            bbox_targets = bbox_targets.reshape(-1, 4) if self.nheads > 1 else bbox_targets
            labels = labels.reshape(-1)
            loss = self.compute_regular_loss(prob, bbox_targets, labels) if self.nheads < 2 else \
                self.compute_gemini_loss_opr(prob, bbox_targets, labels)
            pred_bboxes = self.recover_pred_boxes(rcnn_rois, prob, self.nheads)

            if self.refinement:
                auxi_loss = self.compute_gemini_loss_opr(final_pred, bbox_targets, labels)
                pred_boxes = self.recover_pred_boxes(rcnn_rois, final_pred, self.nheads)
                loss.update(auxi_loss)

            return loss, pred_bboxes
        
        else:

            # return the detection boxes and their scores
            pred_boxes = self.recover_pred_boxes(rcnn_rois, prob, self.nheads)
            if self.refinement:
                pred_boxes = self.recover_pred_boxes(rcnn_rois, final_pred, self.nheads)
            
            return pred_boxes
    
    def recover_pred_boxes(self, rcnn_rois, prob, nhead):

        n = prob.shape[0]
        prob = prob.reshape(n, nhead, -1)
        prob = prob.reshape(-1, prob.shape[2])

        cls_score, bbox_pred = prob[:, -self.n:], prob[:, :-self.n]
        cls_prob = F.softmax(cls_score, axis=1)
        m, c = rcnn_rois.shape
        rois = F.broadcast_to(F.expand_dims(rcnn_rois, axis = 1), (m, nhead, c)).reshape(-1, c)
        bbox_pred = bbox_pred.reshape(n * nhead, -1, 4)
        
        pred_boxes = restore_bbox(rois[:, 1:5], bbox_pred, config = config)
        cls_prob = F.expand_dims(cls_prob, axis=2)
        pred_boxes = F.concat([pred_boxes, cls_prob], axis=2)
        n, c = bbox_pred.shape[:2]
        bid = F.broadcast_to(F.expand_dims(rois[:, :1], axis=1), (n, c, 1))
        pred_boxes = F.concat([pred_boxes, bid], axis = 2)

        return pred_boxes.detach()

    def compute_emd_loss_opr(self, a, b, bbox_targets, labels):
        
        labels = labels.flatten()
        c = a.shape[1]
        prob = F.stack([a, b], axis=1).reshape(-1, c)
        offsets, cls_score = prob[:, :-self.n], prob[:,-self.n:]

        cls_loss = softmax_loss_opr(cls_score, labels)
        n = offsets.shape[0]
        offsets = offsets.reshape(n, -1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)
        reg_loss = smooth_l1_loss_rcnn_opr(offsets, bbox_targets,
            labels, sigma = config.rcnn_smooth_l1_beta)

        vlabel = 1 - ((labels < 0).reshape(-1, 2).sum(axis=1) > 1)
        loss = (cls_loss + 1 * reg_loss).reshape(-1, 2).sum(axis=1) * vlabel
        return loss

    def compute_gemini_loss_opr(self, prob, bbox_targets, labels):

        prob = prob.reshape(prob.shape[0], 2, -1)
        n, _, c = prob.shape
        prob = prob.transpose(1, 0, 2)
        a, b = prob[0], prob[1]
        loss0 = self.compute_emd_loss_opr(a, b, bbox_targets, labels)
        loss1 = self.compute_emd_loss_opr(b, a, bbox_targets, labels)
        loss = F.stack([loss0, loss1], axis = 1)
        emd_loss = loss.min(axis=1).sum() / F.maximum(loss.shape[0], 1)
        loss = {'rcnn_emd_loss': emd_loss}
        return loss

    def compute_regular_loss(self, prob, bbox_targets, labels):

        offsets, cls_scores = prob[:,:-self.n], prob[:, -self.n:]
        n = offsets.shape[0]
        offsets = offsets.reshape(n, -1, 4)
        cls_loss = softmax_loss(cls_scores, labels)
        
        bbox_loss = smooth_l1_loss_rcnn_opr(offsets, bbox_targets,
            labels, config.rcnn_smooth_l1_beta)

        bbox_loss = bbox_loss.sum() / F.maximum((labels > 0).sum(), 1)
        loss = {}
        loss['{}_cls_loss'.format(self.name)] = cls_loss
        loss['{}_bbox_loss'.format(self.name)] = bbox_loss
        return loss

class RCNN(M.Module):

    def __init__(self, iou_thrs = [config.fg_threshold, 0.6], nheads = [1, 2]):
        super().__init__()
        # roi head
        assert len(iou_thrs) > 0 and len(nheads)
        self.iou_thrs = iou_thrs
        self.nheads = nheads
        self.n = config.num_classes
        assert len(iou_thrs) == len(nheads)
        self.subnets = []

        for i, iou_thresh in enumerate(iou_thrs):

            head = CascadeRCNN(iou_thresh, self.nheads[i], i + 1)
            self.subnets.append(head)

    def forward(self, fpn_fms, rcnn_rois, gt_boxes=None, im_info=None):
       
        if self.training:
            
            loss = {}
            for i, _ in enumerate(self.iou_thrs):
                loss_dict, prob = self.subnets[i](fpn_fms, rcnn_rois, gt_boxes, im_info)
                rois = prob[:,1]
                rcnn_list = []
                for bid in range(config.batch_per_gpu):
                    mask = F.equal(rois[:,5], bid)
                    _, index = F.cond_take(mask>0, mask)
                    batch_id = bid * F.ones([mask.sum(), 1])
                    m = F.concat([batch_id, rois[index, :4]],axis=1)
                    rcnn_list.append(m)

                rcnn_rois = F.concat(rcnn_list,axis=0)
                loss.update(loss_dict)

            return loss
        
        else:

            # boxes_pred = self._forward_test(fpn_fms, rcnn_rois)
            for i, _ in enumerate(self.iou_thrs):
                prob = self.subnets[i](fpn_fms, rcnn_rois)
                rois = prob[:,1]
                rcnn_list = []
                for bid in range(1):
                    mask = F.equal(rois[:,5], bid)
                    _, index = F.cond_take(mask>0, mask)
                    batch_id = bid * F.ones([mask.sum(), 1])
                    m = F.concat([batch_id, rois[index, :4]],axis=1)
                    rcnn_list.append(m)

                rcnn_rois = F.concat(rcnn_list,axis=0)
            return prob[:,:,:5]

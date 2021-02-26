import numpy as np
import megengine as mge
import megengine.functional as F
import megengine.module as M
from config import config
from backbone.resnet50 import ResNet50
from module.rpn import RPN
from layers.roi_pool import roi_pool
from det_opr.bbox_opr import bbox_transform_inv_opr, restore_bbox
from det_opr.fpn_roi_target import fpn_roi_target
from det_opr.loss_opr import softmax_loss_opr, smooth_l1_loss_rcnn_opr
from det_opr.utils import get_padded_tensor
import pdb
class Network(M.Module):
    def __init__(self):
        super().__init__()
        # ----------------------- build the backbone ------------------------ #
        self.resnet50 = ResNet50()
        # ------------ freeze the weights of resnet stage1 and stage 2 ------ #
        if config.backbone_freeze_at >= 1:
            for p in self.resnet50.conv1.parameters():
                # p.requires_grad = False
                p = p.detach()
        if config.backbone_freeze_at >= 2:
            for p in self.resnet50.layer1.parameters():
                # p.requires_grad = False
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

        images = inputs['image']
        im_info = inputs['im_info']
        gt_boxes = inputs['gt_boxes']
        #del images
        # process the images
        normed_images = self.pre_process(images)
        if self.training:
            return self._forward_train(normed_images, im_info, gt_boxes)
        else:
            return self._forward_test(normed_images, im_info)

    def _forward_train(self, image, im_info, gt_boxes):

        loss_dict = {}
        # stride: 64,32,16,8,4, p6->p2
        fpn_fms = self.backbone(image)
        rpn_rois, loss_dict_rpn = \
            self.RPN(fpn_fms, im_info, gt_boxes)
        rcnn_rois, rcnn_labels, rcnn_bbox_targets = fpn_roi_target(
            rpn_rois, im_info, gt_boxes, top_k=2)
        loss_dict_rcnn = self.RCNN(
                fpn_fms, rcnn_rois, rcnn_labels, rcnn_bbox_targets)
        loss_dict.update(loss_dict_rpn)
        loss_dict.update(loss_dict_rcnn)
        return loss_dict

    def _forward_test(self, image, im_info):

        fpn_fms = self.backbone(image)
        
        rpn_rois = self.RPN(fpn_fms, im_info)
        
        pred_bbox = self.RCNN(fpn_fms, rpn_rois)
        return pred_bbox

class RCNN(M.Module):
    def __init__(self):
        super().__init__()
        # roi head
        self.refinement = True
        self.fc1 = M.Linear(256*7*7, 1024)
        self.fc2 = M.Linear(1024, 1024)
        self.fc3 = M.Linear(1054, 1024) if self.refinement else None

        self.relu = M.ReLU()

        self.n = config.num_classes
        self.a = M.Linear(1024, 5 * self.n)
        self.b = M.Linear(1024, 5 * self.n)

        self.q = M.Linear(1024, 5 * self.n) if self.refinement else None
        self.r = M.Linear(1024, 5 * self.n) if self.refinement else None
        self._init_weights()

    def _init_weights(self,):
        
        for l in [self.fc1, self.fc2, self.a, self.b]:
            M.init.normal_(l.weight, std=0.01)
            M.init.fill_(l.bias, 0)

        if self.refinement:
            for l in [self.q, self.r, self.fc3]:
                M.init.normal_(l.weight, std=0.01)
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
        prob = F.stack([a, b], axis=1).reshape(-1, a.shape[1])
        return prob

    def forward(self, fpn_fms, rcnn_rois, labels=None, bbox_targets=None):
        # stride: 64,32,16,8,4 -> 4, 8, 16, 32
        fpn_fms = fpn_fms[1:]
        fpn_fms.reverse()
        stride = [4, 8, 16, 32]
        poo5, rcnn_rois, labels, bbox_targets = roi_pool(
                fpn_fms, rcnn_rois, stride, (7, 7), 'roi_align',
                labels, bbox_targets)
        poo5 = F.flatten(poo5, start_axis=1)
        fc1 = F.relu(self.fc1(poo5))
        fc2 = F.relu(self.fc2(fc1))

        a = self.a(fc2)
        b = self.b(fc2)
        prob = F.stack([a, b], axis=1).reshape(-1, a.shape[1])
        
        if self.refinement:
            final_prob = self.refinement_module(prob, fc2)
        
        if self.training:
           
            emd_loss = self.compute_gemini_loss(prob, bbox_targets, labels)
            loss_dict = {}
            loss_dict['loss_rcnn_emd'] = emd_loss
            if self.refinement_module:
                final_emd_loss = self.compute_gemini_loss(final_prob, bbox_targets, labels)
                loss_dict['final_rcnn_emd'] = final_emd_loss
            return loss_dict
        else:

            offsets, cls_scores = prob[:, :-self.n], prob[:, -self.n:]
            pred_bbox = offsets.reshape(-1, self.n, 4)
            cls_prob = F.softmax(cls_scores, axis=1)
            n = rcnn_rois.shape[0]
            rois = F.broadcast_to(F.expand_dims(rcnn_rois[:, 1:5], axis=1), (n, 2, 4)).reshape(-1, 4)
            normalized = config.rcnn_bbox_normalize_targets
            pred_boxes = restore_bbox(rois, pred_bbox, normalized, config)
            pred_bbox = F.concat([pred_boxes, F.expand_dims(cls_prob, axis=2)], axis=2)
            return pred_bbox
    
    def compute_emd_loss(self, a, b, bbox_targets, labels):

        c = a.shape[1]
        prob = F.stack([a, b], axis = 1).reshape(-1, c)
        pred_bbox, cls_scores = prob[:,:-self.n], prob[:,-self.n:]
        n, c = bbox_targets.shape[0], bbox_targets.shape[1]
        bbox_targets, labels = bbox_targets.reshape(-1, 4), labels.flatten()

        cls_loss = softmax_loss_opr(cls_scores, labels)
        pred_bbox = pred_bbox.reshape(-1, self.n, 4)
        rcnn_bbox_loss = smooth_l1_loss_rcnn_opr(pred_bbox, bbox_targets, labels,
            config.rcnn_smooth_l1_beta)
        loss = cls_loss + rcnn_bbox_loss
        loss = loss.reshape(-1, 2).sum(axis=1)
        return loss

    def compute_gemini_loss(self, prob, bbox_targets, labels):

        c = prob.shape[1]
        prob = prob.reshape(-1, 2, c).transpose(1, 0, 2)
        a, b = prob[0], prob[1]
        loss0 = self.compute_emd_loss(a, b, bbox_targets, labels)
        loss1 = self.compute_emd_loss(b, a, bbox_targets, labels)
        loss = F.stack([loss0, loss1], axis=1)
        vlabel = (labels > -1).reshape(-1, 2).sum(axis=1) > 1
        emd_loss = loss.min(axis=1).sum() / F.maximum(vlabel.sum(), 1)
        return emd_loss

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

        # lateral_convs = list()
        # output_convs = list()
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
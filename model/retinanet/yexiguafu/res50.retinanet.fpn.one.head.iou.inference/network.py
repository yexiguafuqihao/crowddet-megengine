import numpy as np
import megengine as mge
import megengine.functional as F
import megengine.module as M
import math
from config import config
from backbone.resnet50 import ResNet50
from module.generate_anchors import generate_anchors
from det_opr.bbox_opr import bbox_transform_inv_opr, box_overlap_opr
from det_opr.utils import get_padded_tensor
from rpn_anchor_target_opr import rpn_anchor_target_opr
from det_opr.loss_opr import sigmoid_cross_entropy_retina, smooth_l1_loss_retina, iou_l1_loss
import pdb

class RetinaNetAnchorV2(M.Module):
    def __init__(self):
        super().__init__()
    
    def generate_anchors_opr(self, fm_3x3, fm_stride,
        anchor_scales=(8, 16, 32, 64, 128), 
        anchor_ratios=(1, 2, 3), base_size = 4):

        np_anchors = generate_anchors(
            base_size=base_size,
            ratios=np.array(anchor_ratios),
            scales=np.array(anchor_scales))
        device = fm_3x3.device
        anchors = mge.tensor(np_anchors).to(device)
        height, width = fm_3x3.shape[2], fm_3x3.shape[3]
        shift_x = F.linspace(0, width-1, width).to(device) * fm_stride
        shift_y = F.linspace(0, height -1, height).to(device) * fm_stride

        broad_shift_x = F.broadcast_to(shift_x.reshape(1, -1), (height, width)).flatten()
        broad_shift_y = F.broadcast_to(shift_y.reshape(-1, 1), (height, width)).flatten()
        shifts = F.stack([broad_shift_x, broad_shift_y, broad_shift_x, broad_shift_y], axis=1)

        c = anchors.shape[1]
        all_anchors = F.expand_dims(anchors, axis=0) + F.expand_dims(shifts, axis=1)
        all_anchors = all_anchors.reshape(-1, c).detach()
        return all_anchors

    def forward(self, fpn_fms):

        all_anchors_list = []
        fm_stride = [8, 16, 32, 64, 128]
        fm_stride.reverse()

        for i, fm_3x3 in enumerate(fpn_fms):
            
            anchor_scales = np.array(config.anchor_base_scale) * fm_stride[i]
            all_anchors = self.generate_anchors_opr(fm_3x3, fm_stride[i], anchor_scales,
                config.anchor_aspect_ratios, base_size = 4)
            all_anchors_list.append(all_anchors)
        return all_anchors_list

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
        # self.RPN = RPN(config.rpn_channel)
        self.head = RetinaNetHead()
        # -------------------------- buid the anchor generator -------------- #
        self.anchor_generator = RetinaNetAnchorV2()
        
        # -------------------------- buid the criteria ---------------------- #
        self.criteria = RetinaNetCriteriaV2()
        # -------------------------- input Tensor --------------------------- #
        self.inputs = {
            "image": mge.tensor(
                np.random.random([2, 3, 756, 1400]).astype(np.float32), dtype="float32",
            ),
            "im_info": mge.tensor(
                np.random.random([2, 6]).astype(np.float32), dtype="float32",
            ),
            "gt_boxes": mge.tensor(
                np.random.random([2, 500, 5]).astype(np.float32), dtype="float32",
            ),
        }
    
    def pre_process(self, images):

        mean = config.image_mean.reshape(1, -1, 1, 1).astype(np.float32)
        std = config.image_std.reshape(1, -1, 1, 1).astype(np.float32)
        mean = mge.tensor(mean).to(images.device)
        std = mge.tensor(std).to(images.device)
        normed_images = (images - mean) / std
        normed_images = get_padded_tensor(normed_images, 64)
        return normed_images

    def forward(self, inputs):

        im_info = inputs['im_info']
        # process the images
        normed_images = self.pre_process(inputs['image'])

        if self.training:
            gt_boxes = inputs['gt_boxes']
            return self._forward_train(normed_images, im_info, gt_boxes)
        else:
            return self._forward_test(normed_images, im_info)

    def _forward_train(self, image, im_info, gt_boxes):

        loss_dict = {}
        # stride: 128,64,32,16,8, p6->p2
        fpn_fms = self.backbone(image)
        pred_cls_list, rpn_num_prob_list, pred_reg_list, rpn_iou_list = self.head(fpn_fms)

        anchors_list = self.anchor_generator(fpn_fms)

        loss_dict = self.criteria(
                pred_cls_list, rpn_num_prob_list, pred_reg_list, anchors_list, 
                rpn_iou_list, gt_boxes, im_info)

        return loss_dict

    def _forward_test(self, image, im_info):

        fpn_fms = self.backbone(image)

        pred_cls_list, rpn_num_prob_list, pred_reg_list, rpn_iou_list = self.head(fpn_fms)

        anchors_list = self.anchor_generator(fpn_fms)

        pred_boxes = self._recover_dtboxes(anchors_list, pred_cls_list,
            pred_reg_list, rpn_iou_list)

        return pred_boxes

    def _recover_dtboxes(self, anchors_list, rpn_cls_list, rpn_bbox_list, rpn_iou_list):

        assert rpn_cls_list[0].shape[0] == 1
        all_anchors = F.concat(anchors_list, axis = 0)
        rpn_cls_scores_final = F.concat(rpn_cls_list, axis=1)[0]
        rpn_bbox_offsets_final = F.concat(rpn_bbox_list,axis=1)[0]
        rpn_iou_prob_final = F.concat(rpn_iou_list, axis=1)[0]

        rpn_bbox_offsets = rpn_bbox_offsets_final.reshape(-1, 4)
        rpn_cls_scores = rpn_cls_scores_final.reshape(-1, 1)
        rpn_iou_prob = rpn_iou_prob_final.reshape(-1, 1)

        n, c = all_anchors.shape[0], all_anchors.shape[1]
        anchors = F.broadcast_to(F.expand_dims(all_anchors, 1), (n, 1, c)).reshape(-1, c)
        rpn_bbox = bbox_transform_inv_opr(anchors, rpn_bbox_offsets)
        pred_boxes = F.concat([rpn_bbox, rpn_cls_scores, rpn_iou_prob], axis=1)
        return pred_boxes
        
class RetinaNetCriteriaV2(M.Module):

    def __init__(self):
        
        super().__init__()

    def anchor_iou_target_opr(self, boxes, im_info, all_anchors,
            rpn_bbox_offsets):

        n = rpn_bbox_offsets.shape[0]
        res = []
        for i in range(n):

            gtboxes = boxes[i, :im_info[i, 5].astype(np.int32)]
            offsets =  rpn_bbox_offsets[i].reshape(-1, 4).detach()
            m = offsets.shape[0]
            an, ac = all_anchors.shape[0], all_anchors.shape[1]
            anchors = F.broadcast_to(F.expand_dims(all_anchors, 1), (an, 1, ac)).reshape(-1, ac)
            dtboxes = bbox_transform_inv_opr(anchors[:,:4], offsets[:, :4])
            overlaps = box_overlap_opr(dtboxes, gtboxes[:, :4])
            ignore_mask = 1 - F.equal(gtboxes[:, 4], config.anchor_ignore_label).astype(np.float32)
            ignore_mask = F.expand_dims(ignore_mask, axis=0)
            overlaps = overlaps * ignore_mask
            
            index = F.argmax(overlaps, axis = 1)
            value = F.nn.indexing_one_hot(overlaps, index, 1)
            value = F.expand_dims(F.expand_dims(value, axis=1), axis=0)
            res.append(value)

        result = F.concat(res, 0)
        return result

    def forward(self, pred_cls_list, rpn_num_prob_list, pred_reg_list,
        anchors_list, rpn_iou_list, boxes, im_info):

        all_anchors_list = [F.concat([a, i*F.ones([a.shape[0], 1]).to(a.device)], axis=1) 
            for i, a in enumerate(anchors_list)]

        all_anchors_final = F.concat(all_anchors_list, axis = 0)
        
        rpn_bbox_offset_final = F.concat(pred_reg_list, axis = 1)
        rpn_cls_prob_final = F.concat(pred_cls_list, axis = 1)
        rpn_iou_prob_final = F.concat(rpn_iou_list, axis = 1)
        rpn_num_per_points_final = F.concat(rpn_num_prob_list, axis = 1)


        rpn_labels, rpn_target_boxes = rpn_anchor_target_opr(boxes, im_info, all_anchors_final)
        ious_target = self.anchor_iou_target_opr(boxes, im_info, all_anchors_final,
            rpn_bbox_offset_final)

        n = rpn_labels.shape[0]
        target_boxes =  rpn_target_boxes.reshape(n, -1, 2, 4).transpose(2, 0, 1, 3)
        rpn_cls_prob_final = rpn_cls_prob_final
        offsets_final = rpn_bbox_offset_final
        target_boxes = target_boxes[0]
        rpn_labels = rpn_labels.transpose(2, 0, 1)
        labels = rpn_labels[0]
        
        cls_loss = sigmoid_cross_entropy_retina(rpn_cls_prob_final, 
                labels, alpha = config.focal_loss_alpha, gamma = config.focal_loss_gamma)
        rpn_bbox_loss = smooth_l1_loss_retina(offsets_final, target_boxes, labels)

        rpn_labels = F.expand_dims(labels, axis=2)
        rpn_iou_loss = iou_l1_loss(rpn_iou_prob_final, ious_target, rpn_labels)
        
        loss_dict = {}
        loss_dict['rpn_cls_loss'] = cls_loss
        loss_dict['rpn_bbox_loss'] = 2 * rpn_bbox_loss
        loss_dict['rpn_iou_loss'] = 2 * rpn_iou_loss
        return loss_dict

class RetinaNetHead(M.Module):
    
    def __init__(self):
        super().__init__()
        num_convs = 4
        in_channels = 256
        cls_subnet, bbox_subnet = [], []
        for _ in range(num_convs):
            cls_subnet.append(
                M.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(M.ReLU())
            bbox_subnet.append(
                M.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(M.ReLU())
        self.cls_subnet = M.Sequential(*cls_subnet)
        self.bbox_subnet = M.Sequential(*bbox_subnet)
        # predictor
        self.cls_score = M.Conv2d(
            in_channels, config.num_cell_anchors * (config.num_classes-1) * 1,
            kernel_size=3, stride=1, padding=1)
        self.bbox_pred = M.Conv2d(
            in_channels, config.num_cell_anchors * 4 * 1,
            kernel_size=3, stride=1, padding=1)

        self.iou_pred = M.Conv2d(
            in_channels, config.num_cell_anchors * 1,
            kernel_size = 3, stride=1, padding = 1)

        self.num_pred = M.Conv2d(in_channels,
            config.num_cell_anchors * 1,
            kernel_size = 3, stride=1, padding = 1)
        self._init_weights()

    def _init_weights(self):

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.num_pred,
                self.cls_score, self.bbox_pred, self.iou_pred]:
            for layer in modules.modules():
                if isinstance(layer, M.Conv2d):
                    M.init.normal_(layer.weight, std=0.01)
                    M.init.fill_(layer.bias, 0)

        prior_prob = 0.01
        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        M.init.fill_(self.cls_score.bias, bias_value)

    def forward(self, features):
   
        cls_prob_list, rpn_num_prob_list, pred_bbox_list, rpn_iou_prob_list = [], [], [], []
        for feature in features:

            rpn_cls_conv = self.cls_subnet(feature)
            cls_score = self.cls_score(rpn_cls_conv)
            rpn_num_prob = self.num_pred(rpn_cls_conv)

            cls_prob = F.sigmoid(cls_score)

            rpn_box_conv = self.bbox_subnet(feature)
            offsets = self.bbox_pred(rpn_box_conv)
            rpn_iou_prob = self.iou_pred(rpn_box_conv)

            cls_prob_list.append(cls_prob)
            pred_bbox_list.append(offsets)
            rpn_iou_prob_list.append(rpn_iou_prob)
            rpn_num_prob_list.append(rpn_num_prob)


        assert cls_prob_list[0].ndim == 4
        pred_cls_list = [
            _.transpose(0, 2, 3, 1).reshape(_.shape[0], -1, (config.num_classes-1))
            for _ in cls_prob_list]
        pred_reg_list = [
            _.transpose(0, 2, 3, 1).reshape(_.shape[0], -1, 4)
            for _ in pred_bbox_list]
        rpn_iou_list = [
            _.transpose(0, 2, 3, 1).reshape(_.shape[0], -1, (config.num_classes-1))
            for _ in rpn_iou_prob_list]

        rpn_num_prob_list = [
            _.transpose(0, 2, 3, 1).reshape(_.shape[0], -1, (config.num_classes-1))
            for _ in rpn_num_prob_list]

        return pred_cls_list, rpn_num_prob_list, pred_reg_list, rpn_iou_list
class FPN(M.Module):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """
    def __init__(self, bottom_up):
        super(FPN, self).__init__()
        in_channels = [512, 1024, 2048]
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
        
        self.p6 = M.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.p7 = M.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.relu = M.ReLU()
        lateral_convs.reverse()
        output_convs.reverse()
        self.lateral_convs = lateral_convs
        self.output_convs = output_convs
        self.bottom_up = bottom_up

    def forward(self, x):

        bottom_up_features = self.bottom_up(x)
        # bottom_up_features = bottom_up_features[::-1]
        bottom_up_features.reverse()

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
        p6 = self.p6(results[0])
        results.insert(0, p6)
        p7 = self.p7(self.relu(p6))
        results.insert(0, p7)
        return results
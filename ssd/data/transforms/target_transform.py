# coding=utf-8 
import numpy as np
import torch

from ssd.utils import box_utils


class SSDTargetTransform:
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors # 构建的anchors  格式 ： center_x,center_y w,h
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors) #  格式 ： x,y w,h
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold) # 将真挚边框分配给 anchor
        boxes = box_utils.corner_form_to_center_form(boxes) # 转换成中心点格式
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
       
        return locations, labels

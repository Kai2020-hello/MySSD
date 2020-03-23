import torch
import math 

def convert_locations_to_boxes(locations, priors, center_variance,size_variance):
    '''
        转换 ssd 位置回归结果 格式是 (center_x,center_y,h,w)
        locations ： (batch_size, num_priors, 4)
        priors ： (num_priors, 4)
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.

        Returns:
        boxes:  priors: [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    '''
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        locations[...,:2] * center_variance * priors[...,2:] + priors[...,:2],
        torch.exp(locations[...,2:] * size_variance * priors[...,:2])
    ], dim=locations.dim() - 1)

# 中心点格式 转化成 角点格式
def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:] / 2,
                      locations[..., :2] + locations[..., 2:] / 2], locations.dim() - 1)

# 焦点格式转化成 中心点格式
def corner_form_to_center_form(boxes):
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)
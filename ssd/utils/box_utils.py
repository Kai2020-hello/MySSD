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

# 根据角点计算面积
def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


# 计算交并比 iou
def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


# 分配anchor给真值
def assign_priors(gt_boxes, gt_labels, corner_form_priors,iou_threshold):
    """
    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    #计算iou num_priors x num_targets 
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))

    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    # size: num_priors x num_targets
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    # size: num_priors 每个anchor对应的每个 真值
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets  目标框对应最好的anchor
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned 让每个目标框对应最好的anchor保留
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels

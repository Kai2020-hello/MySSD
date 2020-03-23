import torch.nn as nn
import torch.nn.functional as F
import torch

def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio
    loss[pos_mask] = -math.inf  # 使得正样本的 损失最小

    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


class MutiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio):
        # neg_pos_ratio 副正比例
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """
            confidence 预测置信度 (batch_   size, num_priors, num_classes)
            predicted_locations 预测的位置 相对于 anchor的 (batch_   size, num_priors, 4)
            labels,  标签 (batch_   size, num_priors)
            gt_locations 位置 (batch_   size, num_priors, 4)
        """
        num_classes = confidence.shape[2]

        # 背景占比非常高，需要从 背景中获取部分参与训练
        with torch.no_grad():
            # 计算背景类型的 loss
            loss = -F.log_softmax(confidence,dim=2)[:,:,0]
            mask = hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.view(-1, num_classes), labels[mask], reduction='sum')
        
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos
from  torch import nn 
from ssd.modeling.backbone import vgg
from ssd.modeling.box_head.box_head import SSDBoxHead

class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = vgg(cfg)
        self.box_head = SSDBoxHead(cfg)

    def forward(self, images, target = None):
        features = self.vgg(images)
        detections, detector_losses = self.box_head(features)

        if self.training:
            return detector_losses
        return detections

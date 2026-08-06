from __future__ import annotations
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_Weights


def build_fasterrcnn_model(num_classes: int, pretrained: bool, device: str):
    # build a faster r-cnn with an fpn backbone (stock torchvision anchors + full pretrained
    # detection weights, incl. RPN/box head - reverted from a custom small-object anchor
    # generator, which required weights=None (RPN/ROI heads training from scratch) and led to
    # much higher/noisier loss in early epochs. Recall gains for small objects should be
    # pursued via other levers (e.g. tile_size, score_thresh, training data) instead.)
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model.to(device)

def build_maskrcnn_model(num_classes: int, pretrained: bool, device: str):
    # build a mask r-cnn with an fpn backbone (stock torchvision anchors + full pretrained
    # detection weights - see build_fasterrcnn_model() for why the custom anchor generator was reverted)
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    # replace the box predictor to match class count
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # replace the mask predictor to match class count
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden, num_classes)
    return model.to(device)
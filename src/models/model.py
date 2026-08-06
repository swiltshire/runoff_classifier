from __future__ import annotations
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models import ResNet50_Weights


def _small_object_anchor_generator() -> AnchorGenerator:
    """Custom RPN anchor generator that ADDS smaller scales below torchvision's
    stock defaults (32/64/128/256/512), instead of replacing them, so the model
    stays able to detect large multi-class features (e.g. Culvert_Structure)
    while adding coverage for tiny features (e.g. Tile_Outlet, ~12-19px
    diagonal at the default tile_size/resize settings).

    One tuple of sizes/aspect_ratios per FPN level (P2-P6 = 5 levels).
    num_anchors_per_location must stay consistent across levels: 3 sizes x
    3 aspect ratios = 9 anchors/location at every level.
    """
    sizes = ((16, 32, 64), (32, 64, 128), (64, 128, 256), (128, 256, 512), (256, 512, 1024))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(sizes)
    return AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)


def build_fasterrcnn_model(num_classes: int, pretrained: bool, device: str):
    # build a faster r-cnn with an fpn backbone and a custom (additive) small-object anchor generator
    weights_backbone = ResNet50_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=weights_backbone,
        rpn_anchor_generator=_small_object_anchor_generator(),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model.to(device)

def build_maskrcnn_model(num_classes: int, pretrained: bool, device: str):
    # build a mask r-cnn with an fpn backbone and a custom (additive) small-object anchor generator
    weights_backbone = ResNet50_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=weights_backbone,
        rpn_anchor_generator=_small_object_anchor_generator(),
    )
    # replace the box predictor to match class count
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # replace the mask predictor to match class count
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden, num_classes)
    return model.to(device)
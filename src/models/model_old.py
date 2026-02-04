
# all variable names are snake_case and all comments are lowercase

from typing import List, Tuple
import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_fasterrcnn_model(num_classes: int,
                           pretrained: bool = True,
                           anchor_sizes: Tuple[Tuple[int, ...], ...] | None = None,
                           image_mean: List[float] | None = None,
                           image_std: List[float] | None = None,
                           device: str | None = None) -> nn.Module:
    """
    create a faster r-cnn model with a resnet50-fpn backbone.

    args:
        num_classes: number of classes including background (background = 0)
        pretrained: load imagenet-pretrained backbone and coco-trained head
        anchor_sizes: optional custom anchor sizes per fpn level
        image_mean: optional per-channel normalization mean
        image_std: optional per-channel normalization std
        device: optional device string (e.g., 'cuda' or 'cpu')

    returns:
        a torch nn module ready for training or inference
    """

    weights = 'DEFAULT' if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # replace the classifier head to match our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # set optional transforms parameters
    if image_mean is not None:
        model.transform.image_mean = image_mean
    if image_std is not None:
        model.transform.image_std = image_std

    # set optional custom anchors
    if anchor_sizes is not None:
        # anchor_sizes is a tuple of tuples, one per fpn level
        model.rpn.anchor_generator.sizes = anchor_sizes

    if device is not None:
        model.to(device)

    return model

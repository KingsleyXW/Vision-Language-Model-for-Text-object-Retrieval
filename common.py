from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction
import torchvision



def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5"
            }
        )

        dummy_out = self.backbone(torch.randn(2, 3, 224*4, 224*4))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        self.fpn_params = nn.ModuleDict()

        # Replace "pass" statement with your code
        i = 3
        for level_name, feature_shape in dummy_out_shapes:
            if i <= 5:
                self.fpn_params[f'c{i}_p{i}'] = nn.Conv2d(feature_shape[1], self.out_channels,1,1, 0)
            else:
                break
            i += 1


        self.fpn_params['p3_p3'] = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        self.fpn_params['p4_p4'] = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        self.fpn_params['p5_p5'] = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        self.fpn_params['p6_p6'] = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        self.fpn_params['p7_p7'] = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        
        self.fpn_params['p5_p6'] = nn.Conv2d(self.out_channels, self.out_channels, 3, 2, 1)
        self.fpn_params['p6_p7'] = nn.Conv2d(self.out_channels, self.out_channels, 3, 2, 1)


    @property
    def fpn_strides(self):

        return {"p3": 8, "p4": 16, "p5": 32, "p6": 64, "p7": 128}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None, "p6": None, "p7": None}

        c3 = backbone_feats['c3']
        c4 = backbone_feats['c4']
        c5 = backbone_feats['c5']

        p5 = self.fpn_params['c5_p5'](c5)
        p4 = self.fpn_params['c4_p4'](c4) + F.interpolate(p5, scale_factor=2)
        p3 = self.fpn_params['c3_p3'](c3) + F.interpolate(p4, scale_factor=2)
        
        p6 = self.fpn_params['p5_p6'](p5)
        p7 = self.fpn_params['p6_p7'](p6)

        # # Smooth
        p7 = self.fpn_params['p7_p7'](p7)
        p6 = self.fpn_params['p6_p6'](p6)
        p5 = self.fpn_params['p5_p5'](p5)
        p4 = self.fpn_params['p4_p4'](p4)
        p3 = self.fpn_params['p3_p3'](p3)
    
        fpn_feats["p3"] = p3
        fpn_feats["p4"] = p4
        fpn_feats["p5"] = p5
        fpn_feats["p6"] = p6
        fpn_feats["p7"] = p7

        fpn_feats_shapes = [(key, value.shape) for key, value in fpn_feats.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in fpn_feats_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "pass" statement with your code
        if type(feat_shape) == torch.Tensor:
           N, C, H, W = feat_shape.shape
        else:
            N, C, H, W = feat_shape

        location = torch.zeros(H*W, 2, device=device, dtype=dtype)
        location_x = torch.arange(H)*level_stride + level_stride/2
        location_y = torch.arange(W)*level_stride + level_stride/2
        for i in range(W):
            location[H*(i):H*(i+1),0] = location_x
            location[H*(i):H*(i+1),1] = location_y[i]*torch.ones(H,dtype = dtype)
        location_coords[level_name] = location
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shape (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = []
    keep = torchvision.ops.nms(boxes, scores, iou_threshold)


    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep

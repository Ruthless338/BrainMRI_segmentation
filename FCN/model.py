import torch
from torch import nn, Tensor
from backbone import resnet50, resnet101
from typing import Dict
from torch.nn import functional as F
from collections import OrderedDict


# 这个类的作用是从一个复杂模型中提取中间层的输出
#
class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layer = return_layers
        # 保证return_layers的键和值都是字符串类型
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 找到所有中间层
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            # 保证每个层只被顺序加入一次
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layer

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        # self.items是继承自nn.ModuleDict，在init函数里被初始化为layers
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FCN(nn.Module):
    # backbone:如ResNet
    # classifier:分类器，如FCNHead
    # aux_classifier:辅助分类器，结构与分类器类似
    def __init__(self, backbone, classifier, aux_classifier=None):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        # 双线性插值
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(FCNHead, self).__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels, out_channels, 1)
        )




def FCN_resnet50(aux, num_classes=21, pretrain_backbone=False):
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        backbone.load_state_dict(torch.load("resnet50.pth", map_location='cpu'))

    out_channels = 2048
    aux_channels = 1024

    return_layers = {'layer4': 'out'}

    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_channels, num_classes)

    classifier = FCNHead(out_channels, num_classes)

    model = FCN(backbone, classifier, aux_classifier)

    return model


def FCN_resnet101(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet101_imagenet': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
    # 'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth'
    backbone = resnet101(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet101 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet101.pth", map_location='cpu'))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN(backbone, classifier, aux_classifier)

    return model

'''
DeepLabV3模型的实现
1. ASPP模块
2. DeepLabV3模型
3. FCNHead模块
4. IntermediateLayerGetter模块
5. DeepLabHead模块
6. deeplabv3_resnet101函数
7. DeepLabV3_ResNet50函数
'''

from collections import OrderedDict
import torch
from torch import nn, Tensor
from backbone import resnet50, resnet101
from torch.nn import functional as F


class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class DeepLabV3(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        size = x.shape[-2:]
        features = self.backbone(x)
        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            result['aux'] = x
        return result


# 将全连接层替换成1x1的卷积层，输出种类概率
class FCNHead(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        mid_channels = in_channels // 4
        super(FCNHead, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        for model in self:
            x = model(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels = 256):
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        ]
        # 若干个ASPP的卷积层之后，还有一个ASPPPooling层（该层直接处理最初的输入）
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


def DeepLabV3_ResNet50(aux, num_classes, pretrain_backbone=False):
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        state_dict = torch.load("resnet50.pth", map_location='cpu')
        backbone.load_state_dict(state_dict)

    out_channels = 2048
    aux_channels = 1024

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    aux_classifier = None
    if aux:
        aux_classifier = FCNHead(aux_channels, num_classes)

    classifier = DeepLabHead(out_channels, num_classes)
    model = DeepLabV3(backbone, classifier, aux_classifier)
    return model


def DeepLabV3_ResNet101(aux, num_classes, pretrain_backbone=False):
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

    classifier = DeepLabHead(out_inplanes, num_classes)

    model = DeepLabV3(backbone, classifier, aux_classifier)

    return model

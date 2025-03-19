import torch
import torch.nn as nn

# 不同于其他几个网络，由于U2Net网络前向传播返回值是一个列表，所以diceLoss的计算方式不同
# 总的loss值等于列表中预测图片损失值的加权值
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, weights=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # 防止除0错误
        self.weights = weights if weights is not None else None

    def forward(self, preds, target):
        total_loss = 0
        if self.weights is None:
            weights = [1.0]+[0.4]*(len(preds)-1)
        else:  # 如果没有设置权重，则使用默认权重
            weights = self.weights
        for i,pred in enumerate(preds):
            total_loss += self.loss(pred, target) * weights[i]
        return total_loss

    def loss(self, pred, target):
        # preds: [batch_size, num_classes, height, width]
        # targets: [batch_size, num_classes, height, width]
        pred = torch.sigmoid(pred)

        intersection = torch.sum(pred * target)  # 计算交集
        union = torch.sum(pred) + torch.sum(target)  # 计算并集

        # Dice系数计算
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Dice损失
        dice_loss = 1 - dice
        return dice_loss


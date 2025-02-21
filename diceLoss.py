import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # 防止除0错误

    def forward(self, preds, targets):
        # preds: [batch_size, num_classes, height, width]
        # targets: [batch_size, num_classes, height, width]

        # 预测值和标签进行平滑
        preds = torch.sigmoid(preds)  # 由于使用的是二分类，使用sigmoid转换为[0, 1]之间的值
        intersection = torch.sum(preds * targets)  # 计算交集
        union = torch.sum(preds) + torch.sum(targets)  # 计算并集

        # Dice系数计算
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Dice损失
        dice_loss = 1 - dice
        return dice_loss


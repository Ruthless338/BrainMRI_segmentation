import torch
import torch.nn as nn
'''
由于输出只有一个通道，所以进行sigmoid激活而不用softmax激活
'''
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # 防止除0错误

    def forward(self, preds, targets):
        # preds: [batch_size, num_classes, height, width]
        # targets: [batch_size, num_classes, height, width]
        preds = torch.sigmoid(preds)

        intersection = torch.sum(preds * targets)  # 计算交集
        union = torch.sum(preds) + torch.sum(targets)  # 计算并集

        # Dice系数计算
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Dice损失
        dice_loss = 1 - dice
        return dice_loss


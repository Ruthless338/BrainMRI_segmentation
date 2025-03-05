import torch
import os
from model import FCN_resnet50
from myDatasets import myDataset
from torch.utils.data import DataLoader
from diceLoss import DiceLoss
from eval import evaluate_model
import torch.optim as optim
from torch.optim import lr_scheduler

# 配置参数
root = '../MRI'  # 替换为实际数据集路径
num_classes = 1  # 根据数据集类别数修改
batch_size = 8
num_epochs = 50
lr = 0.001
momentum = 0.9
weight_decay = 1e-4
pretrain_backbone = False  # 是否使用预训练骨干网络
aux = True  # 是否使用辅助分类器

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model = FCN_resnet50(aux=aux, num_classes=num_classes, pretrain_backbone=pretrain_backbone)
model = model.to(device)

# 数据加载
train_data = myDataset(root, train=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

test_data = myDataset(root, train=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

# 损失函数和优化器
criterion = DiceLoss()
ce_criterion = torch.nn.CrossEntropyLoss()  # 组合Dice Loss和交叉熵
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 学习率调度

best_miou = 0.0

# 训练循环
if __name__ == '__main__':
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (MRI, masks) in enumerate(train_loader):
            MRI = MRI.to(device)
            masks = masks.to(device).long()  # 确保masks是LongTensor
            optimizer.zero_grad()
            outputs = model(MRI)
            main_out = outputs["out"]
            main_out = torch.sigmoid(main_out)
            # 计算主损失
            dice_loss = criterion(main_out, masks)
            ce_loss = ce_criterion(main_out, masks)
            loss = dice_loss + ce_loss
            # 如果使用辅助分类器
            if aux:
                aux_out = outputs["aux"]
                aux_out = torch.sigmoid(aux_out)
                aux_dice = criterion(aux_out, masks)
                aux_ce = ce_criterion(aux_out, masks)
                loss += 0.5 * (aux_dice + aux_ce)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')

    state_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs
    }
    torch.save(state_dict, "state_dict.pth")

    testData = myDataset(root, False)
    test_loader = DataLoader(testData, batch_size=8, shuffle=False, collate_fn=myDataset.collate_fn)

    model.eval()
    results = []
    with torch.no_grad():
        for batch in test_loader:
            MRI, mask = batch
            MRI, mask = MRI.to(device), mask.to(device)
            output = model(MRI)
            main_out = output["out"]
            pred = (torch.sigmoid(main_out) > 0.5).float()  # 二值化预测结果
            metrics = evaluate_model(pred, mask)
            results.append(metrics)

    # 打印平均指标
    avg_metrics = {k: sum([r[k] for r in results]) / len(results) for k in results[0].keys()}
    print("Average Test Metrics:", avg_metrics)

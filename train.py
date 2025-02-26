from torch.utils.data import DataLoader, Dataset
from UNet import UNet
import torch
from myDatasets import myDataset
from diceLoss import DiceLoss
from eval import evaluate_model

# 超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = ''
model = UNet(3, 2, True, 64).to(device)
lr = 0.05
max_iterations = 5
optimizer = torch.optim.SGD(model.parameters(), lr)
criterion = DiceLoss()  # diceLoss

if __name__ == '__main__':

    trainData = myDataset(root, True)
    n = trainData.__len__()
    train_loader = DataLoader(trainData, batch_size=4, shuffle=True, collate_fn=myDataset.collate_fn)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(max_iterations):
        model.train()
        total_loss = 0
        for batch in train_loader:
            MRI, mask = batch
            MRI, mask = MRI.to(device), mask.to(device)  # 确保数据移动到正确的设备
            optimizer.zero_grad()
            output = model(MRI)
            output = torch.sigmoid(output)  # 如果使用 DiceLoss，通常需要对输出进行 sigmoid 激活
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')

    testData = myDataset(root, False)
    test_loader = DataLoader(testData, batch_size=4, shuffle=False, collate_fn=myDataset.collate_fn)

    model.eval()
    results = []
    with torch.no_grad():
        for batch in test_loader:
            MRI, mask = batch
            MRI, mask = MRI.to(device), mask.to(device)
            output = model(MRI)
            pred = (torch.sigmoid(output) > 0.5).float()  # 二值化预测结果
            metrics = evaluate_model(pred, mask)
            results.append(metrics)

    # 打印平均指标
    avg_metrics = {k: sum([r[k] for r in results]) / len(results) for k in results[0].keys()}
    print("Average Test Metrics:", avg_metrics)
